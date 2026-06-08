"""
evaluation_wiring_xgb.py
─────────────────────────
Drop into:  xgboost_project/src/evaluation_wiring_xgb.py

Contains:
  1. enrich_feature_df_with_dates() — adds 'date' column needed for
     temporal train/test splitting in evaluate.py
  2. run_xgb_evaluation() — wires XGBoost objects into RecommenderEvaluator

KEY CHANGE from previous version
─────────────────────────────────
The recommend callables now accept TWO arguments: (user_id, train_df).
evaluate.py passes the train-only interaction slice as the second argument.

For XGBoost this is CRITICAL:
  - recommend_hybrid(interactions_df=tr) builds seen_set from training
    interactions only, so test classes remain as candidates.
  - recommend_guest(interactions_df=tr) does the same for guest users.

Without this fix, test classes end up in seen_set and hits = 0 always.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger("evaluation_wiring_xgb")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — add 'date' column to feature_df
# ─────────────────────────────────────────────────────────────────────────────

def enrich_feature_df_with_dates(feature_df: pd.DataFrame,
                                 enroll_df: pd.DataFrame,
                                 sales_df: pd.DataFrame = None
                                 ) -> pd.DataFrame:
    """
    Joins the most recent interaction date back onto feature_df.

    feature_interactions() produces (user_id, class_id, score) with no
    date column. evaluate.py needs 'date' for temporal train/test splitting.

    Priority: enrollment checkin date -> sales order date -> today (fallback).
    Rows that receive today's date always land in train, never in test.
    """
    df = feature_df.copy()
    df["user_id"]  = df["user_id"].astype(str)
    df["class_id"] = df["class_id"].astype(str)

    # Enrollment dates (primary source)
    enroll_dates = enroll_df[
        ["enrollmentstudentid", "enrollmentclassid", "enrollmentcheckindate"]
    ].copy()
    enroll_dates.columns = ["user_id", "class_id", "date"]
    enroll_dates["user_id"]  = enroll_dates["user_id"].astype(str)
    enroll_dates["class_id"] = enroll_dates["class_id"].astype(str)
    enroll_dates = enroll_dates.dropna(subset=["date"])
    enroll_dates = (
        enroll_dates.sort_values("date")
        .groupby(["user_id", "class_id"])["date"]
        .last()
        .reset_index()
    )
    df = df.merge(enroll_dates, on=["user_id", "class_id"], how="left")

    # Sales order dates (fill remaining gaps)
    if sales_df is not None and "orderdate" in sales_df.columns:
        missing_mask = df["date"].isna()
        if missing_mask.any():
            sales_dates = (
                sales_df[["salesorderstudentid", "orderdate"]]
                .copy()
                .rename(columns={"salesorderstudentid": "user_id",
                                 "orderdate":           "sales_date"})
            )
            sales_dates["user_id"] = sales_dates["user_id"].astype(str)
            sales_dates = (
                sales_dates.dropna(subset=["sales_date"])
                .sort_values("sales_date")
                .groupby("user_id")["sales_date"]
                .last()
                .reset_index()
            )
            df = df.merge(sales_dates, on="user_id", how="left")
            df.loc[missing_mask, "date"] = df.loc[missing_mask, "sales_date"]
            df.drop(columns=["sales_date"], inplace=True, errors="ignore")

    # Safe fallback
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].fillna(pd.Timestamp.now())

    logger.info(
        "Date enrichment — %d / %d rows have a real date",
        (df["date"] < pd.Timestamp.now()).sum(), len(df)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_xgb_evaluation(
        model_bundle,
        feature_df,
        enc_users,
        enc_guests,
        user_encoders,
        preprocess_guest_df,
        recommendation_process,
        k: int = 5,
        test_ratio: float = 0.20,
        output_path: str = "data/output/evaluation_xgb.csv",
):
    """
    Wires XGBoost model objects into RecommenderEvaluator and runs evaluation.

    Call this in recommendation_pipeline_xgboost.py AFTER train_model()
    and BEFORE the model persistence / batch recommendations block.

    Parameters
    ----------
    model_bundle           : dict returned by MLModeling.train_model()
    feature_df             : interaction table WITH 'date' column
                             (run enrich_feature_df_with_dates() first)
    enc_users              : label-encoded registered users DataFrame
    enc_guests             : label-encoded guest users DataFrame
    user_encoders          : dict[col -> LabelEncoder] from encode_categoricals()
    preprocess_guest_df    : preprocessed guest users DataFrame (for guest_ids)
    recommendation_process : your Recommendation() instance
    k                      : number of recommendations per user
    test_ratio             : fraction of interactions held out as test
    output_path            : where to save evaluation_xgb.csv
    """
    from evaluate import RecommenderEvaluator

    evaluator           = RecommenderEvaluator(k=k)
    enriched_classes_df = model_bundle["classes_df"]
    guest_ids           = set(preprocess_guest_df["guestuserid"].astype(str).unique())

    # ── registered_fn ─────────────────────────────────────────────────────
    # Signature: (user_id, train_df)
    # train_df is passed as interactions_df so recommend_hybrid() builds
    # seen_set from training interactions only — NOT from test interactions.
    # Without this, test classes land in seen_set and are never surfaced.
    def registered_fn(uid, interactions_df):
        return recommendation_process.recommend_hybrid(
            user_id         = str(uid),
            model_bundle    = model_bundle,
            interactions_df = interactions_df,  # ← controlled by evaluator
            users_df        = enc_users,
            user_encoders   = user_encoders,
            n               = k,
        )

    # ── guest_fn ──────────────────────────────────────────────────────────
    # Signature: (guest_id, train_df)
    # train_df passed as interactions_df for the same reason — keeps
    # test classes out of seen_set for guest users too.
    # recommend_guest() returns a TUPLE (recs, rec_type) — unwrap with [0].
    # The evaluator also auto-unwraps tuples as a safety net.
    def guest_fn(gid, interactions_df):
        return recommendation_process.recommend_guest(
            guest_user_id   = str(gid),
            model_bundle    = model_bundle,
            interactions_df = interactions_df,  # ← controlled by evaluator
            guest_df        = enc_guests,
            user_encoders   = user_encoders,
            n               = k,
        )[0]   # unwrap (recs, rec_type) tuple — take recs list only

    results = evaluator.run_evaluation(
        model_name      = "XGBoost Hybrid",
        interactions_df = feature_df,
        registered_fn   = registered_fn,
        guest_fn        = guest_fn,
        classes_df      = enriched_classes_df,
        guest_ids       = guest_ids,
        test_ratio      = test_ratio,
    )

    evaluator.print_report(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluator.save_report(results, path=output_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Exact insertion point in recommendation_pipeline_xgboost.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Add at top of recommendation_pipeline_xgboost.py:

    from evaluation_wiring_xgb import enrich_feature_df_with_dates
    from evaluation_wiring_xgb import run_xgb_evaluation

Insert after Step 5 (training matrix) and Step 6 (train model):

    # Step 5 — build XGBoost training matrix
    xgb_train_df = fe.build_xgb_training_data(...)

    # EVALUATION PREP ── NEW
    feature_df_dated = enrich_feature_df_with_dates(
        feature_df, pre_enroll, pre_sales
    )

    # Step 6 — train XGBoost model
    model_bundle = mlm.train_model(
        interactions=feature_df, classes_df=enc_classes,
        users_df=enc_users, xgb_training_df=xgb_train_df,
    )

    # EVALUATION ── NEW
    rec_engine  = rec_module.Recommendation()
    xgb_results = run_xgb_evaluation(
        model_bundle=model_bundle,
        feature_df=feature_df_dated,
        enc_users=enc_users,
        enc_guests=enc_guests,
        user_encoders=user_encoders,
        preprocess_guest_df=pre_guests,
        recommendation_process=rec_engine,
        output_path="data/output/evaluation_xgb.csv",
    )

    # Step 7 — persist model (unchanged)
"""
