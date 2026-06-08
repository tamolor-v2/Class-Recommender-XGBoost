import sys
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ml_model")

# ─────────────────────────────────────────────────────────────────────────────
# Feature column lists — must stay in sync with feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
USER_NUMERIC_COLS = [
    "age", "account_age_days", "has_referral_code", "is_social_referral",
    "join_day_of_week", "join_hour", "join_month", "age_missing",
    "user_interaction_count",   # NEW — activity level signal
]

USER_CAT_COLS = [
    "usergender", "channel_type",
]

CLASS_NUMERIC_COLS = [
    "classcost_numeric",
    "class_duration_mins",
    "start_hour",
    "class_repeat_rate",
    "teacher_affinity_score",
    "location_affinity_score",
    "class_popularity",
    "class_popularity_rank",
]

CLASS_CAT_COLS = [
    "teacher_token", "location_token", "location_clean",
]

ALL_FEATURE_COLS = (
    USER_NUMERIC_COLS + USER_CAT_COLS +
    CLASS_NUMERIC_COLS + CLASS_CAT_COLS +
    ["content_sim"]
)

class MLModeling:
    """
    Hybrid recommendation model:

    ┌─────────────────────────────────────────────────────────┐
    │  XGBoost Ranker  (users WITH interaction history)       │
    │  Learns a score(user, class) from:                      │
    │    • User profile features  (demographics, trust, etc.) │
    │    • Class features         (cost, time, teacher, etc.) │
    │    • Content similarity     (TF-IDF cosine similarity)  │
    ├─────────────────────────────────────────────────────────┤
    │  Content-Based Fallback  (guests + cold-start users)    │
    │  TF-IDF cosine similarity on class 'content' strings    │
    │  + global popularity weighting                          │
    └─────────────────────────────────────────────────────────┘
    """

    def train_model(self, interactions: pd.DataFrame,
                    classes_df: pd.DataFrame,
                    users_df: pd.DataFrame = None,
                    xgb_training_df: pd.DataFrame = None):
        """
        Train the hybrid model.

        Parameters
        ----------
        interactions     : (user_id, class_id, score) — from FeatureEngineering
        classes_df       : preprocessed + label-encoded classes
        users_df         : preprocessed + label-encoded registered users
        xgb_training_df  : pre-built XGBoost feature matrix (from
                           FeatureEngineering.build_xgb_training_data).
                           If None, a warning is logged and the content-based
                           model is trained only.

        Returns
        -------
        dict with keys:
          xgb_model, tfidf, tfidf_matrix, class_index, classes_df,
          feature_cols, interaction_popularity
        """
        logger.info("=== Training XGBoost Hybrid Recommendation Model ===")

        # ── 1.  Content-based: TF-IDF on available classes ────
        logger.info("Building TF-IDF content model...")
        avail_classes = classes_df.copy()

        if "content" not in avail_classes.columns:
            logger.warning(
                "'content' column missing — building fallback from classname + classtag."
            )
            avail_classes["content"] = (
                avail_classes["classname"].fillna("") + " " +
                avail_classes["classtag"].fillna("")
            )

        if "is_available" in avail_classes.columns:
            n_before = len(avail_classes)
            avail_classes = avail_classes[
                avail_classes["is_available"] == 1
            ].reset_index(drop=True)
            logger.info(
                "Excluded %d unavailable classes; %d remain.",
                n_before - len(avail_classes), len(avail_classes)
            )

        # Optional demographic enrichment of content strings
        if users_df is not None:
            avail_classes = self._enrich_content_with_demographics(
                avail_classes, users_df, interactions
            )

        tfidf        = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = tfidf.fit_transform(avail_classes["content"])
        class_index  = avail_classes["classid"].astype(str).tolist()

        logger.info(
            "TF-IDF matrix: %d classes × %d features",
            tfidf_matrix.shape[0], tfidf_matrix.shape[1]
        )

        # ── 2.  Popularity lookup (normalised, used as fallback weight) ─────
        pop = (
            interactions.groupby("class_id")["score"].sum()
            .rename("popularity").reset_index()
        )
        max_pop = pop["popularity"].max()
        # max_pop = pop["popularity"].max() or 1.0
        if pd.isna(max_pop) or max_pop <= 0:
            max_pop = 1.0
        pop["popularity"] = (
                pop["popularity"] / max_pop
        )
        interaction_popularity = dict(zip(pop["class_id"], pop["popularity"]))

        # ── 3.  XGBoost model ───────
        xgb_model    = None
        feature_cols = []

        if xgb_training_df is not None and len(xgb_training_df) > 0:
            logger.info(
                "Training XGBoost ranker on %d samples "
                "(tightened hyperparameters to reduce mild overfitting)...",
                len(xgb_training_df)
            )

            feature_cols = [c for c in ALL_FEATURE_COLS if c in xgb_training_df.columns]
            X = (
                xgb_training_df[feature_cols]
                .fillna(0)
                .astype(float)
            )
            # X = xgb_training_df[feature_cols].astype(float)
            y = (
                xgb_training_df["label"]
                .fillna(0)
                .astype(float)
            )
            # y = xgb_training_df["label"].astype(float)

            # TRAIN / VALIDATION SPLIT
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
                # stratify removed — stratify is for classification with discrete labels.
                # XGBoost is trained as a regressor (objective="reg:squarederror") with
                # continuous float scores as labels. Every score value is nearly unique,
                # so sklearn cannot find 2 members per class — hence the ValueError.
            )

            logger.info("Train split: %d | Validation split: %d",len(X_train), len(X_val))

            # SAMPLE WEIGHTS
            sample_weight = np.where(
                y_train > 0.5,
                2.0,
                1.0
            )

            xgb_model = XGBRegressor(

                # reduced complexity
                n_estimators=220,
                max_depth=4,
                learning_rate=0.04,

                # stochastic regularisation
                subsample=0.7,
                colsample_bytree=0.7,

                # leaf constraints
                min_child_weight=5,

                # regularisation
                reg_alpha=0.5,
                reg_lambda=2.0,

                # objective
                objective="reg:squarederror",

                # metrics
                eval_metric="rmse",

                # infrastructure
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )

            xgb_model.fit(X_train, y_train,
                          sample_weight=sample_weight,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=False)

            logger.info("XGBoost training complete.")
            # FEATURE IMPORTANCE
            importance_df = pd.Series(
                xgb_model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)

            logger.info(
                "Top-10 feature importances:\n%s",
                importance_df.head(10).to_string()
            )
        else:
            logger.warning(
                "No XGBoost training data provided — "
                "model will use content-based scoring only."
            )

        return {
            "xgb_model":              xgb_model,
            "tfidf":                  tfidf,
            "tfidf_matrix":           tfidf_matrix,
            "class_index":            class_index,
            "classes_df":             avail_classes,
            "feature_cols":           feature_cols,
            "interaction_popularity": interaction_popularity,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Demographic enrichment (unchanged from original, adapted for XGBoost)
    # ─────────────────────────────────────────────────────────────────────────
    def _enrich_content_with_demographics(self, classes_df: pd.DataFrame,
                                          users_df: pd.DataFrame,
                                          interactions: pd.DataFrame
                                          ) -> pd.DataFrame:
        required_user_cols  = {"userid", "age_group", "generation", "channel_type"}
        required_inter_cols = {"user_id", "class_id", "score"}

        if not required_user_cols.issubset(users_df.columns):
            logger.warning("Skipping demographic enrichment — missing columns.")
            return classes_df
        if not required_inter_cols.issubset(interactions.columns):
            logger.warning("Skipping demographic enrichment — interactions incomplete.")
            return classes_df

        engaged = interactions[interactions["score"] > 1.0][["user_id", "class_id"]].copy()
        if engaged.empty:
            logger.info("Falling back to score > 0 for demographic enrichment.")
            engaged = interactions[interactions["score"] > 0][
                ["user_id", "class_id"]
            ].copy()

        engaged["user_id"] = engaged["user_id"].astype(str)

        profile = users_df[["userid", "age_group", "generation", "channel_type"]].copy()
        profile["userid"] = profile["userid"].astype(str)

        user_class = profile.merge(engaged, left_on="userid",
                                   right_on="user_id", how="inner")
        if user_class.empty:
            logger.warning("Demographic enrichment join returned no rows.")
            return classes_df

        def safe_mode(s):
            m = s.dropna().mode()
            return m.iloc[0] if not m.empty else ""

        class_demo = (
            user_class.groupby("class_id")
            .agg(
                dominant_age_group =("age_group",   safe_mode),
                dominant_generation=("generation",  safe_mode),
                dominant_channel   =("channel_type", safe_mode),
            )
            .reset_index()
        )
        class_demo["class_id"] = class_demo["class_id"].astype(classes_df["classid"].dtype)

        classes_df = classes_df.merge(
            class_demo, left_on="classid", right_on="class_id", how="left"
        )

        # Guard: ensure content is always a clean string after the merge
        classes_df["content"] = classes_df["content"].astype(str).fillna("")

        for col in ["dominant_age_group", "dominant_generation", "dominant_channel"]:
            classes_df["content"] += " " + classes_df[col].astype(str).fillna("")

        classes_df = classes_df.drop(
            columns=["class_id", "dominant_age_group",
                     "dominant_generation", "dominant_channel"],
            errors="ignore"
        )
        logger.info(
            "Demographic enrichment applied to %d / %d classes.",
            class_demo["class_id"].nunique(), len(classes_df)
        )
        return classes_df
