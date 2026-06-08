"""
evaluate.py
───────────
Offline evaluation module for ALS Hybrid and XGBoost Hybrid recommenders.

ROOT CAUSE OF ZERO PRECISION/RECALL/NDCG/MRR (and the fix)
────────────────────────────────────────────────────────────
recommend_hybrid() and recommend_guest() exclude classes the user has
already interacted with (the 'seen_set'). If the full interactions_df
is passed to those functions during evaluation, the test-set classes
end up in seen_set and are NEVER recommended — so hits are always 0.

Fix: this evaluator splits interactions into train/test, then passes
ONLY the train slice to the recommend callables. The test slice is
used exclusively as ground truth. This mirrors real deployment:
the model only knows about past interactions, not future ones.

Copy this file into BOTH project src/ folders:
    als_project/src/evaluate.py
    xgboost_project/src/evaluate.py
"""

import logging
import sys
from math import log2

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate")


class RecommenderEvaluator:
    """
    Model-agnostic offline evaluator for ranked recommendation lists.

    Handles two user populations independently then reports combined:
      Registered users  -> evaluated via registered_fn (recommend_hybrid)
      Guest users       -> evaluated via guest_fn      (recommend_guest)

    Metrics
    -------
    Precision@K   fraction of top-K recs that are relevant
    Recall@K      fraction of relevant items captured in top-K
    NDCG@K        ranking quality — rewards relevant items higher up
    MRR           position of the first relevant item
    Coverage %    fraction of all available classes ever recommended
    Novelty       how non-popular (surprising) the recommendations are
    """

    def __init__(self, k: int = 5):
        self.k = k

    # ─────────────────────────────────────────────────────────────────────────
    # Primary entry point
    # ─────────────────────────────────────────────────────────────────────────
    def run_evaluation(self,
                       model_name: str,
                       interactions_df: pd.DataFrame,
                       registered_fn,
                       guest_fn,
                       classes_df: pd.DataFrame,
                       guest_ids: set = None,
                       test_ratio: float = 0.20) -> dict:
        """
        Full evaluation pipeline covering both registered and guest users.

        Parameters
        ----------
        model_name      : human-readable label e.g. "ALS Hybrid"

        interactions_df : full (user_id, class_id, score, date) table with
                          BOTH registered and guest interactions.
                          Must have a 'date' column — call
                          enrich_feature_df_with_dates() beforehand.

        registered_fn   : callable(user_id: str, train_df: pd.DataFrame)
                          -> [(class_id, score), ...]

                          IMPORTANT — the signature now takes train_df as
                          the second argument. This is the train-only slice
                          of interactions so that recommend_hybrid()'s
                          seen_set is built from training history only,
                          not from test interactions.

                          ALS example (train_df ignored — ALS matrix already
                          encodes training interactions):
                            lambda uid, tr: als_rec.recommend_hybrid(
                                user_id=uid, model=model, matrix=matrix,
                                user_ids=user_ids, class_ids=class_ids,
                                similarity_matrix=similarity_matrix,
                                classes_df=enriched_classes_df,
                                users_df=preprocess_user_df, n=5
                            )

                          XGBoost example (train_df passed as interactions_df
                          so seen_set excludes test classes):
                            lambda uid, tr: rec_engine.recommend_hybrid(
                                user_id=uid, model_bundle=model_bundle,
                                interactions_df=tr,
                                users_df=enc_users,
                                user_encoders=user_encoders, n=5
                            )

        guest_fn        : callable(guest_id: str, train_df: pd.DataFrame)
                          -> [(class_id, score), ...]

                          ALS (returns plain list, train_df ignored):
                            lambda gid, tr: als_rec.recommend_guest(
                                guest_user_id=gid, model=model, matrix=matrix,
                                class_ids=class_ids,
                                similarity_matrix=similarity_matrix,
                                classes_df=enriched_classes_df, n=5
                            )

                          XGBoost (returns tuple — pass train slice,
                          unwrap with [0]):
                            lambda gid, tr: rec_engine.recommend_guest(
                                guest_user_id=gid, model_bundle=model_bundle,
                                interactions_df=tr,
                                guest_df=enc_guests,
                                user_encoders=user_encoders, n=5
                            )[0]

        classes_df      : enriched classes DataFrame (coverage + novelty).
                          Use model_bundle["classes_df"] for XGBoost,
                          or enriched_classes_df for ALS.

        guest_ids       : set of guest user ID strings.
                          Pass set(preprocess_guest_df["guestuserid"].astype(str))
                          If None, all users are treated as registered.

        test_ratio      : fraction of each user's most recent interactions
                          held out as ground truth (default 20%).
        """
        logger.info("═══ Evaluating: %s ═══", model_name)

        if guest_ids is None:
            guest_ids = set()
        guest_ids = {str(g) for g in guest_ids}

        # ── 1. Time-based split ───────────────────────────────────────────
        train_df, test_df = self.time_based_split(interactions_df, test_ratio)
        logger.info(
            "Split — train: %d rows (%d users) | test: %d rows (%d users)",
            len(train_df), train_df["user_id"].nunique(),
            len(test_df),  test_df["user_id"].nunique(),
        )

        # ── 2. Ground truth from TEST set only ────────────────────────────
        # Restrict to users who appear in BOTH train and test so the model
        # has at least one historical interaction to work with.
        train_user_ids = set(train_df["user_id"].astype(str).unique())
        popularity_map = self._build_popularity_map(train_df)
        # Empty DataFrame used for TRAIN evaluation so seen_set = {}
        empty_df = pd.DataFrame(columns=train_df.columns)

        # ── 2. TEST evaluation (generalisation) ───────────────────────────
        logger.info("--- Computing TEST metrics (generalisation) ---")

        test_gt = self.build_ground_truth(test_df, train_user_ids,
                                          min_train_interactions=2)
        test_reg_gt   = {uid: v for uid, v in test_gt.items() if uid not in guest_ids}
        test_guest_gt = {uid: v for uid, v in test_gt.items() if uid in guest_ids}

        logger.info(
            "TEST ground truth — registered: %d | guest: %d",
            len(test_reg_gt), len(test_guest_gt)
        )


        # Pass train_df so seen_set = training history (test classes stay open)
        test_reg_recs   = self._generate_recs(test_reg_gt,   registered_fn,
                                               train_df, "test/registered")
        test_guest_recs = self._generate_recs(test_guest_gt, guest_fn,
                                               train_df, "test/guest")

        test_reg_m   = self._compute_metrics(test_reg_recs,   test_reg_gt,
                                              popularity_map, classes_df)
        test_guest_m = self._compute_metrics(test_guest_recs, test_guest_gt,
                                              popularity_map, classes_df)
        test_overall = self._compute_metrics(
            {**test_reg_recs,   **test_guest_recs},
            {**test_reg_gt,     **test_guest_gt},
            popularity_map, classes_df
        )

        # ── 3. TRAIN evaluation (memorisation / overfitting check) ─────────
        # Ground truth = the user's OWN training interactions.
        # seen_set = {} (empty_df passed) so the full catalogue is available.
        # A model that perfectly memorises training data scores 1.0 here.
        # Comparing these numbers to test metrics reveals the gap.
        logger.info("--- Computing TRAIN metrics (memorisation check) ---")

        train_gt = self.build_ground_truth_from_train(train_df, guest_ids,
                                                       min_interactions=3)
        train_reg_gt   = {uid: v for uid, v in train_gt.items() if uid not in guest_ids}
        train_guest_gt = {uid: v for uid, v in train_gt.items() if uid in guest_ids}
        logger.info(
            "TRAIN ground truth — registered: %d | guest: %d",
            len(train_reg_gt), len(train_guest_gt)
        )

        # Pass empty_df so seen_set = {} — full catalogue open as candidates
        train_reg_recs   = self._generate_recs(train_reg_gt,   registered_fn,
                                                empty_df, "train/registered")
        train_guest_recs = self._generate_recs(train_guest_gt, guest_fn,
                                                empty_df, "train/guest")

        train_reg_m   = self._compute_metrics(train_reg_recs,   train_reg_gt,
                                               popularity_map, classes_df)
        train_guest_m = self._compute_metrics(train_guest_recs, train_guest_gt,
                                               popularity_map, classes_df)
        train_overall = self._compute_metrics(
            {**train_reg_recs,   **train_guest_recs},
            {**train_reg_gt,     **train_guest_gt},
            popularity_map, classes_df
        )

        # ── 4. Diagnosis ──────────────────────────────────────────────────
        diagnosis = self._diagnose(train_overall, test_overall)

        # ── 5. Flatten result dict ────────────────────────────────────────
        results = {
            "model":     model_name,
            "diagnosis": diagnosis,
            # Test (generalisation)
            **{f"test_overall_{k}":    v for k, v in test_overall.items()},
            **{f"test_registered_{k}": v for k, v in test_reg_m.items()},
            **{f"test_guest_{k}":      v for k, v in test_guest_m.items()},
            # Train (memorisation)
            **{f"train_overall_{k}":    v for k, v in train_overall.items()},
            **{f"train_registered_{k}": v for k, v in train_reg_m.items()},
            **{f"train_guest_{k}":      v for k, v in train_guest_m.items()},
        }

        self._log_full_summary(
            model_name,
            train_overall, test_overall,
            train_reg_m,   test_reg_m,
            train_guest_m, test_guest_m,
            diagnosis
        )
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnosis logic
    # ─────────────────────────────────────────────────────────────────────────
    def _diagnose(self, train_metrics: dict, test_metrics: dict) -> str:
        """
        Compares train vs test Precision@K and NDCG@K to produce a verdict.

        Thresholds
        ──────────────────────────────────────────────────────────────────────
        Precision gap > 0.10  OR  NDCG gap > 0.15   →  OVERFITTING
            Train metrics significantly higher than test — the model has
            memorised training patterns and cannot generalise.

        Precision gap > 0.05                         →  MILD OVERFITTING
            Some generalisation gap — review feature engineering,
            interaction weights, or consider regularisation.

        Both precision < 0.03                        →  UNDERFITTING
            Model has not learned enough signal. Consider richer features,
            stronger interaction weights, or a lower minimum interaction
            threshold for evaluation.
        Otherwise                                    →  ACCEPTABLE GENERALISATION
            Train and test metrics are similar — model generalises well.
            Low absolute values are expected given dataset sparsity.
        """
        k = self.k
        train_p    = train_metrics.get(f"precision@{k}", 0.0)
        test_p     = test_metrics.get(f"precision@{k}",  0.0)
        train_ndcg = train_metrics.get(f"ndcg@{k}",      0.0)
        test_ndcg  = test_metrics.get(f"ndcg@{k}",       0.0)
        gap_p      = train_p    - test_p
        gap_ndcg   = train_ndcg - test_ndcg
        if gap_p > 0.10 or gap_ndcg > 0.15:
            verdict = "OVERFITTING"
            detail  = (
                f"Train P@{k}={train_p:.4f} vs Test P@{k}={test_p:.4f} "
                f"(gap={gap_p:+.4f}). "
                f"Train NDCG@{k}={train_ndcg:.4f} vs Test NDCG@{k}={test_ndcg:.4f} "
                f"(gap={gap_ndcg:+.4f}). "
                "Model memorised training interactions and generalises poorly. "
                "Consider stronger regularisation (reg_alpha, reg_lambda) in XGBoost, "
                "reducing n_estimators, or increasing min_child_weight."
            )
        elif gap_p > 0.05:
            verdict = "MILD OVERFITTING"
            detail  = (
                f"Train P@{k}={train_p:.4f} vs Test P@{k}={test_p:.4f} "
                f"(gap={gap_p:+.4f}). "
                "Some generalisation gap detected. Review interaction weights "
                "and consider adding more interaction data to reduce sparsity."
            )
        elif train_p < 0.03 and test_p < 0.03:
            verdict = "UNDERFITTING"
            detail  = (
                f"Both Train P@{k}={train_p:.4f} and Test P@{k}={test_p:.4f} are low. "
                "Model has not learned enough signal. Consider: "
                "(1) richer features e.g. pref_time_of_day, "
                "(2) stronger enroll_map weights for completed enrollments, "
                "(3) sharper recency decay, "
                "(4) lowering min_interactions threshold for evaluation."
            )
        else:
            verdict = "ACCEPTABLE GENERALISATION"
            detail  = (
                f"Train P@{k}={train_p:.4f} vs Test P@{k}={test_p:.4f} "
                f"(gap={gap_p:+.4f}). "
                "Metrics are consistent across splits. "
                "Low absolute values are expected for sparse implicit feedback data."
            )

        return f"{verdict} | {detail}"

    # ─────────────────────────────────────────────────────────────────────────
    # Time-based split
    # ─────────────────────────────────────────────────────────────────────────
    def time_based_split(self, interactions_df: pd.DataFrame,
                         test_ratio: float = 0.20) -> tuple:
        """
        Holds out each user's most recent (test_ratio) interactions as test.
        Users with fewer than 3 interactions keep everything in train.
        Falls back to random split if 'date' column is absent.
        """
        df = interactions_df.copy()
        df["user_id"]  = df["user_id"].astype(str)
        df["class_id"] = df["class_id"].astype(str)

        if "date" not in df.columns:
            logger.warning(
                "'date' column missing — using random split. "
                "Call enrich_feature_df_with_dates() for temporal accuracy."
            )
            mask = np.random.rand(len(df)) < test_ratio
            return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
        df = df.sort_values(["user_id", "date"])
        train_rows, test_rows = [], []

        for _, group in df.groupby("user_id"):
            n = len(group)
            if n < 3:
                # Fewer than 3 interactions — keep all in train.
                # Not enough history to split meaningfully.
                train_rows.append(group)
                continue
            n_test = max(1, int(n * test_ratio))
            train_rows.append(group.iloc[:-n_test])
            test_rows.append(group.iloc[-n_test:])

        train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        test  = pd.concat(test_rows,  ignore_index=True) if test_rows  else pd.DataFrame()
        return train, test

    # ─────────────────────────────────────────────────────────────────────────
    # Ground truth builders
    # ─────────────────────────────────────────────────────────────────────────
    def build_ground_truth(self, test_df: pd.DataFrame,
                           train_user_ids: set = None,
                           min_train_interactions: int = 2) -> dict:
        """
        TEST ground truth: {user_id -> set of class_ids from held-out test}.
        Only includes users who also appear in train with enough history.

        Parameters
        ----------
        test_df                : held-out test interactions
        train_user_ids         : set of user IDs present in train
        min_train_interactions : minimum number of train interactions
                                 required for a user to be evaluated.
                                 Users below this threshold are excluded —
                                 the model has too little history for
                                 meaningful evaluation.
        """
        gt = {}
        for uid, group in test_df[test_df["score"] > 0].groupby("user_id"):
            uid_str = str(uid)
            if train_user_ids is not None and uid_str not in train_user_ids:
                continue
            gt[uid_str] = set(group["class_id"].astype(str))

        logger.info(
            "TEST ground truth — %d evaluable users | avg %.1f relevant items/user",
            len(gt), np.mean([len(v) for v in gt.values()]) if gt else 0
        )
        return gt

    def build_ground_truth_from_train(self, train_df: pd.DataFrame,
                                      guest_ids: set = None,
                                      min_interactions: int = 3) -> dict:
        """
        TRAIN ground truth: {user_id -> set of class_ids from training}.

        Used to measure memorisation: given an empty seen_set, does the
        model surface the classes the user actually interacted with?

        Only includes users with >= min_interactions training interactions
        to avoid trivially inflating the train metric with single-item users.

        Parameters
        ----------
        train_df         : training interaction table
        guest_ids        : set of guest user IDs (for logging split)
        min_interactions : minimum training interactions required
                           (default 3 — at least a real preference signal)
        """
        if guest_ids is None:
            guest_ids = set()

        counts = train_df[train_df["score"] > 0].groupby("user_id").size()
        eligible = set(counts[counts >= min_interactions].index.astype(str))

        gt = {}
        for uid, group in train_df[train_df["score"] > 0].groupby("user_id"):
            uid_str = str(uid)
            if uid_str not in eligible:
                continue
            gt[uid_str] = set(group["class_id"].astype(str))

        logger.info(
            "TRAIN ground truth — %d users (>= %d interactions) | "
            "avg %.1f relevant items/user",
            len(gt), min_interactions,
            np.mean([len(v) for v in gt.values()]) if gt else 0
        )
        return gt

    # ─────────────────────────────────────────────────────────────────────────
    # Recommendation generation
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_recs(self, ground_truth: dict,
                       recommend_fn,
                       interactions_df: pd.DataFrame,
                       label: str) -> dict:
        """
        Calls recommend_fn(user_id, interactions_df) for every user.

        The interactions_df argument controls seen_set behaviour:
          TEST evaluation  → pass train_df   (seen_set = training history)
          TRAIN evaluation → pass empty_df   (seen_set = {}, full catalogue open)

        Auto-unwraps tuple returns (safety net for XGBoost recommend_guest).
        """
        recs    = {}
        skipped = 0

        for uid in ground_truth:
            try:
                result = recommend_fn(str(uid), interactions_df)
                if isinstance(result, tuple):
                    result = result[0]
                if result:
                    recs[str(uid)] = result
            except Exception as exc:
                skipped += 1
                logger.debug("Skipping %s user %s — %s", label, uid, exc)

        logger.info(
            "%s — recs for %d / %d users | %d skipped",
            label, len(recs), len(ground_truth), skipped
        )
        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # Core metric computation
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_metrics(self, recommendations: dict,
                         ground_truth: dict,
                         popularity_map: dict,
                         classes_df: pd.DataFrame) -> dict:

        precisions, recalls, ndcgs, rrs, novelties = [], [], [], [], []
        all_rec_ids = set()

        for uid, recs in recommendations.items():
            if uid not in ground_truth or not ground_truth[uid]:
                continue

            relevant = ground_truth[uid]
            rec_ids  = [str(cid) for cid, _ in recs[:self.k]]
            all_rec_ids.update(rec_ids)

            hits = sum(1 for cid in rec_ids if cid in relevant)
            precisions.append(hits / self.k)
            recalls.append(hits / len(relevant))

            dcg  = sum(1 / log2(i + 2) for i, cid in enumerate(rec_ids)
                       if cid in relevant)
            idcg = sum(1 / log2(i + 2) for i in range(min(len(relevant), self.k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            rr = next(
                (1 / (i + 1) for i, cid in enumerate(rec_ids) if cid in relevant),
                0.0
            )
            rrs.append(rr)

            novelties.append(float(np.mean([
                -log2(popularity_map.get(cid, 1e-6) + 1e-9)
                for cid in rec_ids
            ])))

        total_classes = len(classes_df["classid"].astype(str).unique())
        coverage_pct  = (
            round(len(all_rec_ids) / total_classes * 100, 2)
            if total_classes > 0 else 0.0
        )

        def _avg(lst):
            return round(float(np.mean(lst)), 4) if lst else 0.0

        return {
            f"precision@{self.k}": _avg(precisions),
            f"recall@{self.k}":    _avg(recalls),
            f"ndcg@{self.k}":      _avg(ndcgs),
            "mrr":                 _avg(rrs),
            "coverage_pct":        coverage_pct,
            "coverage_count":      len(all_rec_ids),
            "novelty":             _avg(novelties),
            "users_evaluated":     len(precisions),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _build_popularity_map(self, train_df: pd.DataFrame) -> dict:
        pop     = train_df.groupby("class_id")["score"].sum()
        max_pop = pop.max() or 1.0
        return (pop / max_pop).to_dict()

    def _log_full_summary(self, model_name,
                          train_overall, test_overall,
                          train_reg,     test_reg,
                          train_guest,   test_guest,
                          diagnosis):
        k   = self.k
        sep = "═" * 90
        logger.info(
            "\n%s\n  %s — OVERFITTING / UNDERFITTING REPORT\n%s\n"
            "\n  OVERALL\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  "
            "Cov=%.1f%%  Nov=%.4f  n=%d\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  "
            "Cov=%.1f%%  Nov=%.4f  n=%d\n"
            "  %-20s  P@%d=%+.4f R@%d=%+.4f NDCG@%d=%+.4f MRR=%+.4f\n"
            "\n  REGISTERED\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  n=%d\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  n=%d\n"
            "\n  GUEST\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  n=%d\n"
            "  %-20s  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f  MRR=%.4f  n=%d\n"
            "\n%s\n  DIAGNOSIS: %s\n%s",
            sep, model_name, sep,
            # Overall train
            "TRAIN (memorisation)",
            k, train_overall[f"precision@{k}"], k, train_overall[f"recall@{k}"],
            k, train_overall[f"ndcg@{k}"], train_overall["mrr"],
            train_overall["coverage_pct"], train_overall["novelty"],
            train_overall["users_evaluated"],
            # Overall test
            "TEST (generalisation)",
            k, test_overall[f"precision@{k}"], k, test_overall[f"recall@{k}"],
            k, test_overall[f"ndcg@{k}"], test_overall["mrr"],
            test_overall["coverage_pct"], test_overall["novelty"],
            test_overall["users_evaluated"],
            # Gap row
            "GAP (train - test)",
            k, train_overall[f"precision@{k}"] - test_overall[f"precision@{k}"],
            k, train_overall[f"recall@{k}"]    - test_overall[f"recall@{k}"],
            k, train_overall[f"ndcg@{k}"]      - test_overall[f"ndcg@{k}"],
            train_overall["mrr"] - test_overall["mrr"],
            # Registered
            "TRAIN registered",
            k, train_reg[f"precision@{k}"], k, train_reg[f"recall@{k}"],
            k, train_reg[f"ndcg@{k}"], train_reg["mrr"], train_reg["users_evaluated"],
            "TEST registered",
            k, test_reg[f"precision@{k}"],  k, test_reg[f"recall@{k}"],
            k, test_reg[f"ndcg@{k}"],  test_reg["mrr"],  test_reg["users_evaluated"],
            # Guest
            "TRAIN guest",
            k, train_guest[f"precision@{k}"], k, train_guest[f"recall@{k}"],
            k, train_guest[f"ndcg@{k}"], train_guest["mrr"],
            train_guest["users_evaluated"],
            "TEST guest",
            k, test_guest[f"precision@{k}"],  k, test_guest[f"recall@{k}"],
            k, test_guest[f"ndcg@{k}"],  test_guest["mrr"],
            test_guest["users_evaluated"],
            sep, diagnosis, sep,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────
    def print_report(self, *results: dict):
        """
        Pretty-prints a comparison table.
        Groups columns by train_*, test_*, and diagnosis.
        """
        for result in results:
            model = result.get("model", "Unknown")
            diag  = result.get("diagnosis", "")
            print("\n" + "═" * 90)
            print(f"  {model}")
            print("═" * 90)

            # Print train vs test side by side for key metrics
            k = self.k
            groups = ["overall", "registered", "guest"]
            metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
                       "mrr", "coverage_pct", "novelty", "users_evaluated"]

            header = f"  {'Group':<12} {'Metric':<20} {'TRAIN':>10} {'TEST':>10} {'GAP':>10}"
            print(header)
            print("  " + "─" * 56)

            for group in groups:
                for metric in metrics:
                    train_val = result.get(f"train_{group}_{metric}", 0.0)
                    test_val  = result.get(f"test_{group}_{metric}",  0.0)
                    try:
                        gap = round(float(train_val) - float(test_val), 4)
                        print(
                            f"  {group.upper():<12} {metric:<20} "
                            f"{float(train_val):>10.4f} {float(test_val):>10.4f} "
                            f"{gap:>+10.4f}"
                        )
                    except (TypeError, ValueError):
                        print(f"  {group.upper():<12} {metric:<20} "
                              f"{str(train_val):>10} {str(test_val):>10}")

            print("\n  DIAGNOSIS:")
            # Word-wrap the diagnosis for readability
            words   = diag.split()
            line    = "  "
            for word in words:
                if len(line) + len(word) > 88:
                    print(line)
                    line = "  " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)
            print("═" * 90 + "\n")

    def save_report(self, *results: dict,
                    path: str = "data/output/evaluation.csv"):
        """Saves result dicts to CSV in long format."""
        rows = []
        for result in results:
            model = result.get("model", "Unknown")
            diag  = result.get("diagnosis", "")
            for key, val in result.items():
                if key in ("model", "diagnosis"):
                    continue
                # key format: split_group_metric e.g. test_overall_precision@5
                parts = key.split("_", 2)
                if len(parts) == 3:
                    split, group, metric = parts
                else:
                    split, group, metric = key, "", key
                rows.append({
                    "model":     model,
                    "split":     split,
                    "group":     group,
                    "metric":    metric,
                    "value":     val,
                    "diagnosis": diag,
                })

        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Evaluation report saved → %s", path)
