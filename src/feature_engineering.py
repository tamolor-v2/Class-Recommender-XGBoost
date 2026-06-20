import pandas as pd
import numpy as np
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
# import random

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("feature_engineering")

###############################################################################
# Numeric user features forwarded to the XGBoost feature matrix
###############################################################################
USER_NUMERIC_COLS = [
    "age", "account_age_days", "has_referral_code", "is_social_referral",
    "join_day_of_week", "join_hour", "join_month", "age_missing",
    "user_interaction_count",   # NEW — total interactions for user
]

# Numeric class features forwarded to the XGBoost feature matrix
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

# Label-encoded categorical columns (already encoded in preprocess.py)
USER_CAT_COLS = ["usergender", "channel_type"]

CLASS_CAT_COLS = ["teacher_token", "location_token", "location_clean"]


###############################################################################
# Feature Engineering Class
###############################################################################
class FeatureEngineering:

    ###########################################################################
    # 1.  Interaction-signal aggregation  (unchanged from original)
    ###########################################################################
    def feature_interactions(
            self,
            cart,
            enroll,
            order_map,
            transfer_map,
            enroll_map,
            sales_order=None,
            transfer_class=None,
            guest_data=None,
            users_df=None,
            classes_df=None
    ):

        logger.info("Extracting interaction features...")
        today = pd.Timestamp.now()

        ###############################################
        # SAFETY: CF TRUST
        ###############################################
        cf_trust_map = {}
        if users_df is not None and "cf_trust" in users_df.columns:
            cf_trust_map = dict(
                zip(users_df["userid"].astype(str),
                    users_df["cf_trust"].astype(float))
            )

        ###############################################
        # CLASS LOOKUPS
        ###############################################
        cost_weight_map = {}
        available_set = None

        if classes_df is not None:
            cdf = classes_df.copy()
            cdf["classid"] = cdf["classid"].astype(str)

            if "cost_weight" in cdf.columns:
                cost_weight_map = dict(zip(cdf["classid"], cdf["cost_weight"]))

            if "is_available" in cdf.columns:
                available_set = set(cdf[cdf["is_available"] == 1]["classid"])

            logger.info(
                "Availability filter: %d available / %d total",
                len(available_set or []), len(cdf)
            )

        def cost_w(cid):
            return cost_weight_map.get(str(cid), 1.0)

        def is_available(cid):
            return available_set is None or str(cid) in available_set

        def safe_days(dt):
            if pd.isna(dt):
                return 365
            return max((today - dt).days, 0)

        def recency_weight(dt):
            return 1 / (1 + safe_days(dt))

        ################################################
        # CART SIGNAL
        ################################################
        cart_df = cart[["shoppingcartitemstudentid", "shoppingcartitemclassid"]].copy()
        cart_df.columns = ["user_id", "class_id"]
        cart_df["user_id"] = cart_df["user_id"].astype(str)
        cart_df["class_id"] = cart_df["class_id"].astype(str)
        cart_df = cart_df[cart_df["class_id"].apply(is_available)]

        cart_df["score"] = cart_df["class_id"].apply(lambda c: 0.35 * cost_w(c))

        ################################################
        # ENROLLMENT SIGNAL
        ################################################
        enroll_df = enroll[
            ["enrollmentstudentid", "enrollmentclassid",
             "enrollmentstatus", "enrollmentcheckindate"]
        ].copy()

        enroll_df.columns = ["user_id", "class_id", "status", "date"]
        enroll_df["user_id"] = enroll_df["user_id"].astype(str)
        enroll_df["class_id"] = enroll_df["class_id"].astype(str)
        enroll_df["date"] = pd.to_datetime(enroll_df["date"], errors="coerce")

        enroll_df = enroll_df[enroll_df["class_id"].apply(is_available)]

        enroll_df["score"] = enroll_df["status"].map(enroll_map).fillna(1)
        enroll_df["days"] = (today - enroll_df["date"]).dt.days.fillna(365)
        enroll_df["recency"] = enroll_df["date"].apply(recency_weight)

        enroll_df["score"] = (
            enroll_df["score"] * enroll_df["recency"] * enroll_df["class_id"].apply(cost_w)
        )

        enroll_df = enroll_df[["user_id", "class_id", "score"]]

        ################################################
        # REPEAT ENROLLMENT SIGNAL
        ################################################
        repeat_enroll_df = pd.DataFrame()

        try:
            re = enroll.copy()

            # SAFETY: only keep required columns first
            required_cols = [
                "enrollmentstudentid",
                "enrollmentclassid",
                "enrollmentstatus",
                "enrollmentcheckindate"
            ]

            re = re[required_cols].copy()

            # NOW safe rename (no mismatch possible)
            re.rename(columns={
                "enrollmentstudentid": "user_id",
                "enrollmentclassid": "class_id",
                "enrollmentstatus": "status",
                "enrollmentcheckindate": "date"
            }, inplace=True)

            re["user_id"] = re["user_id"].astype(str)
            re["class_id"] = re["class_id"].astype(str)
            re["date"] = pd.to_datetime(re["date"], errors="coerce")

            re = re[re["class_id"].apply(is_available)]

            repeat_counts = (
                re.groupby(["user_id", "class_id"])
                .size()
                .reset_index(name="total_enrollments")
            )

            repeat_counts = repeat_counts[repeat_counts["total_enrollments"] > 1]

            if not repeat_counts.empty:
                repeat_counts["score"] = np.log1p(repeat_counts["total_enrollments"])
                repeat_enroll_df = repeat_counts[["user_id", "class_id", "score"]]

                logger.info(
                    "Repeat enrollment signal — %d pairs detected",
                    len(repeat_enroll_df)
                )

        except Exception as e:
            logger.warning("Repeat enrollment skipped safely: %s", e)

        ###############################################
        # TRANSFERS (FULL)
        ###############################################
        transfer_df = pd.DataFrame()

        if transfer_class is not None:
            t = transfer_class.copy()

            if "transferenrollmentid" in t.columns and "enrollment" in enroll.columns:
                enroll_lookup = enroll[
                    ["enrollmentid", "enrollmentstudentid", "enrollmentclassid"]
                ].copy()

                t = t.merge(enroll_lookup,
                            left_on="transferenrollmentid",
                            right_on="enrollmentid",
                            how="left")

                t = t.rename(columns={
                    "enrollmentstudentid": "user_id",
                    "enrollmentclassid": "class_id",
                    "transferstatus": "status",
                    "transferdaterequest": "date"
                })

                t["user_id"] = t["user_id"].astype(str)
                t["class_id"] = t["class_id"].astype(str)

                t = t[t["class_id"].apply(is_available)]
                t["score"] = t["status"].map(transfer_map).fillna(0)
                t["recency"] = t["date"].apply(recency_weight)
                t["score"] *= t["recency"]

                transfer_df = t[["user_id", "class_id", "score"]]

        ###############################################
        # SALES
        ###############################################
        sales_df = pd.DataFrame()

        if sales_order is not None:
            s = sales_order.copy()

            s["shoppingcartid"] = s["shoppingcartid"].astype(str)

            cart_ids = cart[
                ["shoppingcartid", "shoppingcartitemclassid"]
            ].drop_duplicates()

            cart_ids["shoppingcartid"] = cart_ids["shoppingcartid"].astype(str)

            s = s.merge(cart_ids, on="shoppingcartid", how="left")
            s = s.dropna(subset=["shoppingcartitemclassid"])

            s = s.rename(columns={
                "salesorderstudentid": "user_id",
                "shoppingcartitemclassid": "class_id",
                "creditcardapprovalstatus": "status",
                "orderdate": "date"
            })

            s["user_id"] = s["user_id"].astype(str)
            s["class_id"] = s["class_id"].astype(str)

            s = s[s["class_id"].apply(is_available)]
            s["score"] = s["status"].map(order_map).fillna(0)
            s["recency"] = s["date"].apply(recency_weight)

            s["score"] *= s["recency"] * s["class_id"].apply(cost_w)

            sales_df = s[["user_id", "class_id", "score"]]

        ################################################
        # GUEST INTERACTIONS
        ################################################
        guest_interactions_df = pd.DataFrame()

        if guest_data is not None:

            g = guest_data.copy()
            g["guestuserid"] = g["guestuserid"].astype(str)

            guest_cart = g.merge(
                cart[["shoppingcartitemstudentid", "shoppingcartitemclassid"]],
                left_on="guestuserid",
                right_on="shoppingcartitemstudentid",
                how="inner"
            )

            guest_cart = guest_cart.rename(columns={
                "guestuserid": "user_id",
                "shoppingcartitemclassid": "class_id"
            })

            guest_cart["user_id"] = guest_cart["user_id"].astype(str)
            guest_cart["class_id"] = guest_cart["class_id"].astype(str)

            guest_cart = guest_cart[guest_cart["class_id"].apply(is_available)]
            guest_cart["score"] = guest_cart["class_id"].apply(lambda c: 0.45 * cost_w(c))

            guest_interactions_df = guest_cart[["user_id", "class_id", "score"]]


        ###############################################
        # COMBINE ALL SIGNALS
        ###############################################
        df = pd.concat(
            [cart_df, enroll_df, repeat_enroll_df, transfer_df, sales_df, guest_interactions_df],
            ignore_index=True
        )

        df = df.groupby(["user_id", "class_id"])["score"].sum().reset_index()

        df["user_id"] = df["user_id"].astype(str)
        df["class_id"] = df["class_id"].astype(str)

        ################################################
        # POPULARITY FEATURES
        ################################################
        class_pop = df.groupby("class_id")["score"].sum().reset_index()
        class_pop.columns = ["class_id", "class_popularity"]

        max_pop = class_pop["class_popularity"].max()
        class_pop["class_popularity"] = class_pop["class_popularity"] / (max_pop + 1e-6)
        class_pop["class_popularity_rank"] = class_pop["class_popularity"].rank(pct=True)

        df = df.merge(class_pop, on="class_id", how="left")

        df["class_popularity"] = df["class_popularity"].fillna(0.0)
        df["class_popularity_rank"] = df["class_popularity_rank"].fillna(0.0)

        df["score"] = df["score"] * (1 + df["class_popularity"] * 0.10) # Previous value: 0.25

        # normalize per user
        max_scores = df.groupby("user_id")["score"].transform("max")
        df["score"] = (df["score"] / (max_scores + 1e-6)).clip(0, 1)

        ###############################################
        # USER INTERACTION FEATURES
        ###############################################
        df["user_interaction_count"] = df.groupby("user_id")["class_id"].transform("count")
        df["user_activity_weight"] = np.log1p(df["user_interaction_count"])

        df["score"] *= (1 + df["user_activity_weight"] * 0.02)  # Previous value: 0.05

        ###############################################
        # SAFE FEATURE GUARANTEES
        ###############################################
        for col in [
            "class_repeat_rate",
            "teacher_affinity_score",
            "location_affinity_score"
        ]:
            if col not in df.columns:
                df[col] = 0.0

        logger.info(
            "Final interaction matrix — %d rows | %d users | %d classes",
            len(df), df["user_id"].nunique(), df["class_id"].nunique()
        )

        return df

    #### label-encode categoricals in preparation for XGBoost model ##############
    def encode_categoricals(self, users_df, classes_df):
        """
        Fits LabelEncoders on user and class categorical columns and returns:
          - encoded users_df
          - encoded classes_df
          - user_encoders  dict[col -> LabelEncoder]
          - class_encoders dict[col -> LabelEncoder]

        'unknown' / unseen values are mapped to -1 so XGBoost treats them
        as missing (compatible with use_label_encoder=False).
        """
        user_encoders  = {}
        class_encoders = {}

        u_df = users_df.copy()
        for col in USER_CAT_COLS:
            if col not in u_df.columns:
                continue
            le = LabelEncoder()
            u_df[col] = u_df[col].astype(str).fillna("unknown")
            le.fit(u_df[col])
            u_df[col] = le.transform(u_df[col])
            user_encoders[col] = le

        c_df = classes_df.copy()
        for col in CLASS_CAT_COLS:
            if col not in c_df.columns:
                continue
            le = LabelEncoder()
            c_df[col] = c_df[col].astype(str).fillna("unknown")
            le.fit(c_df[col])
            c_df[col] = le.transform(c_df[col])
            class_encoders[col] = le

        logger.info(
            "Label encoding done — %d user cols | %d class cols",
            len(user_encoders), len(class_encoders)
        )
        return u_df, c_df, user_encoders, class_encoders

    def encode_new_user(self, user_row: pd.Series,
                        user_encoders: dict) -> pd.Series:
        """
        Applies fitted encoders to a single user row at inference time.
        Unseen categories are mapped to 0 (safe default for XGBoost).
        """
        row = user_row.copy()
        for col, le in user_encoders.items():
            if col not in row.index:
                continue
            val = str(row[col]) if pd.notna(row[col]) else "unknown"
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
        return row

    ###########################################################################
    # 2.  XGBoost training-row builder
    #     Joins interaction scores with user + class feature vectors.
    ###########################################################################
    def build_xgb_training_data(self, interactions_df, users_df, classes_df,
                                tfidf_matrix, class_index):

        logger.info("Building XGBoost training rows...")

        tfidf_matrix = csr_matrix(tfidf_matrix)

        # ----------------------------
        # ENSURE REQUIRED COLUMNS EXIST
        # ----------------------------
        required_interaction_cols = [
            "class_repeat_rate",
            "class_popularity",
            "class_popularity_rank",
            "user_interaction_count"
        ]

        for col in required_interaction_cols:
            if col not in interactions_df.columns:
                interactions_df[col] = 0.0

        # ----------------------------
        # SAFE CLASS FEATURE EXTRACTION
        # ----------------------------
        classes_df = classes_df.copy()
        engineered_class_features = interactions_df[
            ["class_id", "class_repeat_rate",
             "class_popularity", "class_popularity_rank"]
        ].drop_duplicates()

        engineered_class_features = engineered_class_features.copy()

        classes_df.columns = classes_df.columns.str.lower()
        engineered_class_features.columns = engineered_class_features.columns.str.lower()

        # enforce consistent key name
        if "class_id" in classes_df.columns and "classid" not in classes_df.columns:
            classes_df.rename(columns={"class_id": "classid"}, inplace=True)

        if "class_id" in engineered_class_features.columns and "classid" not in engineered_class_features.columns:
            engineered_class_features.rename(columns={"class_id": "classid"}, inplace=True)

        # safety check (IMPORTANT)
        assert "classid" in classes_df.columns, "classes_df missing classid"
        assert "classid" in engineered_class_features.columns, "engineered_class_features missing classid"

        classes_df = classes_df.merge(
            engineered_class_features,
            on="classid",
            how="left"
        )

        for col in ["class_repeat_rate", "class_popularity", "class_popularity_rank"]:
            if col not in classes_df.columns:
                classes_df[col] = 0.0

            classes_df[col] = classes_df[col].fillna(0.0)

        # ----------------------------
        # LOOKUPS
        # ----------------------------
        u_cols = [c for c in USER_NUMERIC_COLS + USER_CAT_COLS if c in users_df.columns]
        c_cols = [c for c in CLASS_NUMERIC_COLS + CLASS_CAT_COLS if c in classes_df.columns]

        rows = []
        users_df["userid"] = users_df["userid"].astype(str)
        classes_df["classid"] = classes_df["classid"].astype(str)

        u_lookup = users_df.set_index("userid")[u_cols]
        c_lookup = classes_df.set_index("classid")[c_cols]

        # TF-IDF MATRIX
        tfidf_class_ids = (
            classes_df["classid"]
            .astype(str)
            .tolist()
        )

        cid_to_tfidf_idx = {cid: i for i, cid in enumerate(class_index)}

        # ----------------------------
        # FAST FEATURE LOOKUPS
        # ----------------------------
        ic_lookup = dict(zip(
            interactions_df["user_id"].astype(str),
            interactions_df.get("user_interaction_count", 0)
        ))

        crr_lookup = dict(zip(
            interactions_df["class_id"].astype(str),
            interactions_df.get("class_repeat_rate", 0)
        ))

        rows = []

        user_history = interactions_df.groupby("user_id")["class_id"].apply(set).to_dict()
        all_classes = set(classes_df["classid"].astype(str))

        # ----------------------------
        # POSITIVE SAMPLES
        # ----------------------------
        import random
        for uid, seen_classes in user_history.items():

            user_rows = interactions_df[interactions_df["user_id"] == uid]

            for _, row in user_rows.iterrows():

                cid = str(row["class_id"])

                if cid not in c_lookup.index:
                    continue

                if uid in u_lookup.index:
                    u_feats = u_lookup.loc[uid].to_dict()
                else:
                    u_feats = {col: 0 for col in u_cols}
                c_feats = c_lookup.loc[cid].to_dict()

                if "user_interaction_count" not in u_feats:
                    u_feats["user_interaction_count"] = ic_lookup.get(uid, 0)

                if "class_repeat_rate" not in c_feats:
                    c_feats["class_repeat_rate"] = crr_lookup.get(cid, 0)

                rows.append({
                    **u_feats,
                    **c_feats,
                    "content_sim": 1.0, # Previous value: 0.0
                    "label": float(row["score"])
                })

            # ----------------------------
            # NEGATIVE SAMPLING (FIXED)
            # ----------------------------
            unseen = list(all_classes - seen_classes)

            if len(unseen) == 0:
                continue

            neg_sample_size = min(len(unseen), max(5, len(seen_classes) * 3))
            # neg_sample_size = min(len(unseen), max(3, len(seen_classes) * 2))
            # hard negatives = semantically similar unseen classes
            # negative_classes = random.sample(unseen, neg_sample_size)
            negative_candidates = []
            # negative_classes = []

            for seen_cid in seen_classes:

                if seen_cid not in cid_to_tfidf_idx:
                    continue

                seen_idx = cid_to_tfidf_idx[seen_cid]

                sims = cosine_similarity(
                    tfidf_matrix[seen_idx],
                    tfidf_matrix
                ).flatten()

                similar_indices = np.argsort(sims)[::-1][1:20]
                # similar_indices = np.argsort(sims)[::-1][1:15]

                for idx in similar_indices:

                    candidate_cid = tfidf_class_ids[idx]

                    if candidate_cid not in seen_classes and candidate_cid in unseen:
                        negative_candidates.append(candidate_cid)
            # deduplicate
            negative_candidates = list(set(negative_candidates))

            # fallback if too few hard negatives
            if len(negative_candidates) < neg_sample_size:
                remaining = list(set(unseen) - set(negative_candidates))
                extra_needed = neg_sample_size - len(negative_candidates)

                if remaining:
                    negative_candidates.extend(
                        random.sample(
                            remaining,
                            min(extra_needed, len(remaining))
                        )
                    )

            # final sample
            if len(negative_candidates) > neg_sample_size:
                negative_classes = random.sample(
                    negative_candidates,
                    neg_sample_size
                )
            else:
                negative_classes = negative_candidates

            for cid in negative_classes:
                if cid not in c_lookup.index:
                    continue

                if uid in u_lookup.index:
                    u_feats = u_lookup.loc[uid].to_dict()
                else:
                    u_feats = {col: 0 for col in u_cols}

                c_feats = c_lookup.loc[cid].to_dict()

                if "user_interaction_count" not in u_feats:
                    u_feats["user_interaction_count"] = ic_lookup.get(uid, 0)

                if "class_repeat_rate" not in c_feats:
                    c_feats["class_repeat_rate"] = crr_lookup.get(cid, 0)

                target_idx = cid_to_tfidf_idx.get(cid)
                content_sim = 0.0
                sim_indices = []

                # Guard: only proceed when this class exists in the TF-IDF
                # index. cid_to_tfidf_idx.get() returns None for classes
                # filtered out as unavailable before TF-IDF was fitted.
                if target_idx is not None:
                    if not isinstance(target_idx, (int, np.integer)):
                        target_idx = int(target_idx)

                    sim_indices = [
                        cid_to_tfidf_idx[c]
                        for c in seen_classes
                        if c in cid_to_tfidf_idx
                    ]

                if target_idx is not None and sim_indices:
                    target_vec = tfidf_matrix[int(target_idx), :]
                    peer_vecs  = tfidf_matrix[sim_indices, :]

                    sims = cosine_similarity(
                        target_vec,
                        peer_vecs
                    )
                    content_sim = float(sims.mean())

                rows.append({
                    **u_feats,
                    **c_feats,
                    "content_sim": content_sim, # Previous value: 0.0
                    "label": 0.0
                })

        xgb_df = pd.DataFrame(rows).fillna(0)

        logger.info("XGBoost matrix — %d rows × %d features",
                    len(xgb_df), xgb_df.shape[1] - 1)

        return xgb_df

    ###########################################################################
    # 3. Build a feature row for ONE (user, class) pair at inference time
    ###########################################################################
    def build_inference_row(self, user_features: dict, class_features: dict,
                            content_sim: float) -> pd.DataFrame:
        """
        Merges user + class feature dicts and content similarity into a single
        1-row DataFrame in the same column order as the training matrix.
        """
        record = {**user_features, **class_features, "content_sim": content_sim}
        return pd.DataFrame([record]).fillna(0)
