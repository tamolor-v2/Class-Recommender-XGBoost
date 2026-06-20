import ingest
import preprocess
import feature_engineering
import ml_model
import recommend
import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import pickle
import os
import pandas as pd
import logging
import sys
import evaluation_wiring
from evaluation_wiring import enrich_feature_df_with_dates, run_xgb_evaluation

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("recommendation_pipeline")

ingest_process = ingest.Ingestion()
# This makes dataframes of all the source data global variables, which is accessible everywhere
(users_df, classes_df, enrollment_df, sales_order_df, shopping_cart_df, discount_df, guest_users_df,
 transfer_class_df, order_weight_df, transfer_weight_df, enroll_weight_df, duration_df, channel_df,
 recency_df, time_of_day_df, cost_tier_df) = ingest_process.ingest_data()

########################################################################################
class RecommendationPipeline:

    def run_pipeline(self):
        logger.info("═══════════════════ Running XGBoost Pipeline ═══════════════════")
        pp  = preprocess.Preprocessing()
        fe  = feature_engineering.FeatureEngineering()
        mlm = ml_model.MLModeling()

        ###### 1. Preprocess ###########################################################
        logger.info("Step 1 — Preprocessing...")
        pre_users    = pp.preprocessing_users_data(users_df, channel_df, recency_df, time_of_day_df)
        pre_guests   = pp.preprocessing_guest_data(guest_users_df, time_of_day_df, recency_df)
        pre_classes  = pp.preprocessing_classes_data(classes_df, duration_df, cost_tier_df)
        pre_cart     = pp.preprocessing_cart_data(shopping_cart_df)
        pre_enroll   = pp.preprocessing_enrollment_data(enrollment_df)
        pre_sales    = pp.preprocessing_sales_data(sales_order_df)
        pre_transfer = pp.preprocessing_transfer_data(transfer_class_df)

        order_map, transfer_map, enroll_map = pp.preprocessing_weight_data(
            order_weight_df, transfer_weight_df, enroll_weight_df
        )

        ###### 2. Analytics outputs ####################################################
        logger.info("Step 2 — Writing analytics outputs...")
        os.makedirs("data/output", exist_ok=True)

        enrollment_df["checkin_date"] = pre_enroll["enrollmentcheckindate"]
        trend = (
            enrollment_df
            .groupby(enrollment_df["checkin_date"].dt.date)
            .size()
            .reset_index(name="count")
        )
        trend.to_csv("data/output/enrollment_trend.csv", index=False)

        funnel = pd.DataFrame({
            "stage": ["Cart", "Enrollment", "Completion"],
            "count": [
                shopping_cart_df["shoppingcartitemstudentid"].nunique(),
                enrollment_df["enrollmentstudentid"].nunique(),
                enrollment_df[
                    enrollment_df["enrollmentstatus"] == "Finished"
                ]["enrollmentstudentid"].nunique(),
            ]
        })
        funnel.to_csv("data/output/funnel.csv", index=False)

        #### 3. Interaction feature table ###################################################
        logger.info("Step 3 — Building interaction signals...")
        feature_df = fe.feature_interactions(
            cart=pre_cart, enroll=pre_enroll,
            order_map=order_map, transfer_map=transfer_map, enroll_map=enroll_map,
            sales_order=pre_sales, transfer_class=pre_transfer,
            guest_data=pre_guests, users_df=pre_users, classes_df=pre_classes,
        )

        popular = (
            feature_df.groupby("class_id")["score"].sum()
            .reset_index()
            .merge(pre_classes, left_on="class_id", right_on="classid")
        )
        popular.to_csv("data/output/popular_classes.csv", index=False)

        #### 4. Label-encode categoricals for XGBoost ######################################
        logger.info("Step 4 — Label-encoding categoricals...")
        enc_users, enc_classes, user_encoders, class_encoders = (
            fe.encode_categoricals(pre_users, pre_classes)
        )

        # Encode guest users using the same fitted encoders
        enc_guests = pre_guests.copy()
        for col, le in user_encoders.items():
            if col not in enc_guests.columns:
                continue
            enc_guests[col] = enc_guests[col].astype(str).fillna("unknown")
            enc_guests[col] = enc_guests[col].apply(
                lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
            )

        #### 5. Build XGBoost training matrix ###################################################
        logger.info("Step 5 — Building XGBoost training matrix...")
        # We need TF-IDF to compute content_sim; build a temporary vectoriser
        # on the raw class content strings.
        from sklearn.feature_extraction.text import TfidfVectorizer as _TFIDF
        avail_classes_raw = pre_classes[pre_classes["is_available"] == 1].reset_index(drop=True)
        _tfidf_tmp  = _TFIDF(stop_words="english", max_features=500)
        _tfidf_mat  = _tfidf_tmp.fit_transform(avail_classes_raw["content"])
        _class_idx  = avail_classes_raw["classid"].astype(str).tolist()

        xgb_train_df = fe.build_xgb_training_data(
            interactions_df = feature_df,
            users_df        = enc_users,
            classes_df      = enc_classes,
            tfidf_matrix    = _tfidf_mat,
            class_index     = _class_idx,
        )

        # Enrich feature_df with dates for temporal train/test split in evaluator
        # Done here so the dated df is ready immediately after training
        feature_df_dated = enrich_feature_df_with_dates(feature_df, pre_enroll, pre_sales)

        # ── 6. Train model ─────────────────────────────────────────────────
        logger.info("Step 6 — Training XGBoost hybrid model...")
        model_bundle = mlm.train_model(
            interactions    = feature_df,
            classes_df      = enc_classes,
            users_df        = enc_users,
            xgb_training_df = xgb_train_df,
        )

        #### 6a. Evaluate — train AND test metrics for overfitting detection ##################
        #
        # The evaluator runs two passes:
        #   TRAIN pass: empty_df passed → seen_set={} → full catalogue open
        #               ground truth = user's own training interactions
        #               high scores here = model memorised training data
        #
        #   TEST pass:  train_df passed → seen_set=training history
        #               ground truth = held-out recent interactions
        #               measures generalisation to future behaviour
        #
        # The GAP between train and test metrics is the overfitting signal.
        # The diagnosis key in xgb_results contains the automated verdict.
        logger.info("Step 6a — Evaluating model (train + test splits)...")
        rec_engine  = recommend.Recommendation()
        xgb_results = run_xgb_evaluation(
            model_bundle           = model_bundle,
            feature_df             = feature_df_dated,
            enc_users              = enc_users,
            enc_guests             = enc_guests,
            user_encoders          = user_encoders,
            preprocess_guest_df    = pre_guests,
            recommendation_process = rec_engine,
            output_path            = "data/output/evaluation_xgb.csv",
        )

        # Log diagnosis prominently so it is the first thing seen in output
        logger.info(
            "\n%s\n  MODEL DIAGNOSIS:\n  %s\n%s",
            "═" * 80,
            xgb_results.get("diagnosis", "No diagnosis available"),
            "═" * 80,
        )

        #### 7. Persist model bundle ###########################################################
        logger.info("Step 7 — Saving model...")
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(
                {
                    "version":        "xgb_hybrid_v1",
                    "model_bundle":   model_bundle,
                    "feature_df":     feature_df,
                    "users_df":       enc_users,
                    "guests_df":      enc_guests,
                    "raw_users_df":   pre_users,
                    "user_encoders":  user_encoders,
                    "class_encoders": class_encoders,
                    "evaluation":     xgb_results,   # ← train+test metrics persisted
                },
                f,
            )
        logger.info("Model saved → models/model.pkl")

        #### 8. Batch recommendations ###########################################################
        logger.info("Step 8 — Generating batch recommendations...")
        enriched_classes_df = model_bundle["classes_df"]
        all_recs = []

        # 8a. Registered users
        users_with_interactions    = set(feature_df["user_id"].astype(str))
        users_in_preprocess        = set(pre_users["userid"].astype(str))
        users_without_interactions = users_in_preprocess - users_with_interactions

        logger.info(
            "Registered — total: %d | with interactions: %d | cold-start: %d",
            len(users_in_preprocess),
            len(users_with_interactions & users_in_preprocess),
            len(users_without_interactions),
        )

        fallback_recs = rec_engine._popularity_fallback(
            model_bundle["interaction_popularity"],
            model_bundle["class_index"],
            n=5,
        )
        fallback_df = rec_engine.show_recommendations_with_names(
            fallback_recs, enriched_classes_df
        )

        for uid in users_in_preprocess:
            uid = str(uid)
            if uid in users_with_interactions:
                recs = rec_engine.recommend_hybrid(
                    user_id         = uid,
                    model_bundle    = model_bundle,
                    interactions_df = feature_df,
                    users_df        = enc_users,
                    user_encoders   = user_encoders,
                    n               = 5,
                )
                readable = rec_engine.show_recommendations_with_names(
                    recs, enriched_classes_df
                )
            else:
                readable = fallback_df.copy()

            readable["user_id"]   = uid
            readable["user_type"] = "registered"
            all_recs.append(readable)

        # 8b. Guest users
        all_guest_ids = pre_guests["guestuserid"].astype(str).unique()
        users_with_guest_interactions = set(feature_df["user_id"].astype(str))
        guests_hybrid   = 0
        guests_fallback = 0

        logger.info("Generating recommendations for %d guest users...", len(all_guest_ids))

        guest_recs = []
        for gid in all_guest_ids:
            gid = str(gid)
            if gid in users_with_guest_interactions:
                guests_hybrid += 1
                recs, rec_type = rec_engine.recommend_guest(
                    guest_user_id   = gid,
                    model_bundle    = model_bundle,
                    interactions_df = feature_df,
                    guest_df        = enc_guests,
                    user_encoders   = user_encoders,
                    n               = 5,
                )
            else:
                guests_fallback += 1
                recs = rec_engine._popularity_fallback(
                    model_bundle["interaction_popularity"],
                    model_bundle["class_index"],
                    n=5,
                )
                rec_type = "popularity"

            readable = rec_engine.show_recommendations_with_names(
                recs, enriched_classes_df
            )
            readable["user_id"]   = gid
            readable["user_type"] = "guest"
            readable["rec_type"]  = rec_type
            guest_recs.append(readable)

        logger.info(
            "Guest recommendations complete — %d hybrid | %d popularity fallback.",
            guests_hybrid, guests_fallback
        )

        # 8c. Save combined output
        df_all = pd.concat(all_recs + guest_recs, ignore_index=True)
        df_all.to_csv("data/output/recommendations.csv", index=False)

        logger.info(
            "Recommendations saved — %d unique users (%d registered + %d guests)",
            df_all["user_id"].nunique(),
            len(users_in_preprocess),
            len(all_guest_ids),
        )
        return df_all


##################################################################################
# FastAPI app
##################################################################################
app = FastAPI(title="Class Recommendation API — XGBoost Hybrid")

#### Load persisted model bundle at startup ######################################
_model_bundle  = None
_feature_df    = None
_enc_users     = None
_enc_guests    = None
_raw_users_df  = None
_user_encoders = None

try:
    with open("models/model.pkl", "rb") as _f:
        _data = pickle.load(_f)
        if isinstance(_data, dict) and "model_bundle" in _data:
            _model_bundle  = _data["model_bundle"]
            _feature_df    = _data.get("feature_df", pd.DataFrame())
            _enc_users     = _data.get("users_df")
            _enc_guests    = _data.get("guests_df")
            _raw_users_df  = _data.get("raw_users_df")
            _user_encoders = _data.get("user_encoders", {})
        else:
            logger.warning("Legacy or unknown model format in models/model.pkl.")
except (EOFError, FileNotFoundError) as _e:
    logger.warning("Model file missing or empty — run pipeline first. (%s)", _e)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


#### Registered-user endpoint #############################################################
@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the pipeline first.")

    rec_engine = recommend.Recommendation()
    uid        = str(user_id)

    # Check whether user has interactions
    has_interactions = (
        not _feature_df.empty and
        uid in set(_feature_df["user_id"].astype(str))
    )

    if has_interactions:
        recs = rec_engine.recommend_hybrid(
            user_id         = uid,
            model_bundle    = _model_bundle,
            interactions_df = _feature_df,
            users_df        = _enc_users,
            user_encoders   = _user_encoders,
            n               = 5,
        )
        rec_type = "hybrid"
    else:
        recs = rec_engine._popularity_fallback(
            _model_bundle["interaction_popularity"],
            _model_bundle["class_index"],
            n=5,
        )
        rec_type = "popularity"

    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user {user_id}."
        )

    enriched = _model_bundle["classes_df"]
    name_map = dict(zip(enriched["classid"].astype(str), enriched["classname"]))

    results = [
        {
            "class_id":   str(cid),
            "class_name": name_map.get(str(cid), "Unknown"),
            "score":      round(float(score), 6),
        }
        for cid, score in recs
    ]

    # User profile metadata for transparency
    user_meta = {}
    if _raw_users_df is not None:
        row = _raw_users_df[_raw_users_df["userid"].astype(str) == uid]
        if not row.empty:
            row = row.iloc[0]
            cf_trust = float(row.get("cf_trust", 0.0))
            user_meta = {
                "age_group":           str(row.get("age_group", "unknown")),
                "generation":          str(row.get("generation", "unknown")),
                "join_recency_bucket": str(row.get("join_recency_bucket", "unknown")),
                "cf_trust":            round(cf_trust, 4),
                "is_cold_start":       int(row.get("is_cold_start", 0)),
                "dynamic_alpha":       round(0.1 + cf_trust * 0.5, 4),
            }

    return {
        "user_id":             uid,
        "recommendation_type": rec_type,
        "user_profile":        user_meta,
        "recommendations":     results,
    }


#### Guest endpoint #########################################################################
@app.get("/recommend/guest/{guest_user_id}")
def get_guest_recommendations(guest_user_id: str):
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the pipeline first.")

    rec_engine = recommend.Recommendation()

    recs, rec_type = rec_engine.recommend_guest(
        guest_user_id   = str(guest_user_id),
        model_bundle    = _model_bundle,
        interactions_df = _feature_df if _feature_df is not None else pd.DataFrame(),
        guest_df        = _enc_guests,
        user_encoders   = _user_encoders,
        n               = 5,
    )

    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations for guest {guest_user_id}."
        )

    enriched = _model_bundle["classes_df"]
    name_map = dict(zip(enriched["classid"].astype(str), enriched["classname"]))

    results = [
        {
            "class_id":   str(cid),
            "class_name": name_map.get(str(cid), "Unknown"),
            "score":      round(float(score), 6),
        }
        for cid, score in recs
    ]

    return {
        "guest_user_id":       guest_user_id,
        "recommendation_type": rec_type,
        "recommendations":     results,
    }


####################################################################################
# Entry point
####################################################################################
if __name__ == "__main__":
    pipeline = RecommendationPipeline()
    pipeline.run_pipeline()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)