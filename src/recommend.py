import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("recommendation")

# Feature column lists (must match ml_model.py / feature_engineering.py)
USER_NUMERIC_COLS = [
    "age", "account_age_days", "has_referral_code", "is_social_referral",
    "join_day_of_week", "join_hour", "join_month", "age_missing",
]
USER_CAT_COLS = [
    "usergender", "channel_type",
]
CLASS_NUMERIC_COLS = [
    "classcost_numeric", "class_duration_mins", "start_hour",
]
CLASS_CAT_COLS = [
    "teacher_token", "location_token", "location_clean",
]

class Recommendation:
    """
    Hybrid recommendation engine built on XGBoost + TF-IDF content model.

    Routing logic
    ─────────────
    ┌─────────────────────────────────────────────────────────────────┐
    │  registered user WITH interactions  →  XGBoost hybrid scoring   │
    │  registered user WITHOUT interactions (cold-start)              │
    │    → content-based (TF-IDF) + popularity fallback               │
    │  guest user WITH cart/order interactions  →  XGBoost hybrid     │
    │  guest user WITHOUT any interactions  →  popularity fallback    │
    └─────────────────────────────────────────────────────────────────┘

    The XGBoost score is blended with the TF-IDF content score using
    dynamic_alpha derived from cf_trust (same mapping as the ALS version):

        final_score = dynamic_alpha * xgb_score
                    + (1 - dynamic_alpha) * content_score
                    + 0.10 * pop_score
                    + 0.05 * demo_boost
                    + 0.05 * attr_boost
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — registered users
    # ─────────────────────────────────────────────────────────────────────────
    def recommend_hybrid(self,
                         user_id: str,
                         model_bundle: dict,
                         interactions_df: pd.DataFrame,
                         users_df: pd.DataFrame = None,
                         user_encoders: dict = None,
                         n: int = 5,
                         alpha: float = 0.3) -> list:
        """
        Returns top-n (class_id, score) tuples for a registered user.

        Parameters
        ----------
        user_id         : str — registered user ID
        model_bundle    : dict returned by MLModeling.train_model()
        interactions_df : (user_id, class_id, score) interaction table
        users_df        : preprocessed + label-encoded users DataFrame
        user_encoders   : dict[col -> LabelEncoder] from encode_categoricals
        n               : number of recommendations
        alpha           : default CF weight (overridden by cf_trust when available)
        """
        xgb_model    = model_bundle["xgb_model"]
        tfidf_matrix = model_bundle["tfidf_matrix"]
        class_index  = model_bundle["class_index"]
        classes_df   = model_bundle["classes_df"]
        feature_cols = model_bundle["feature_cols"]
        pop_map      = model_bundle["interaction_popularity"]

        uid = str(user_id)

        # ── User profile features ──
        user_feats, dynamic_alpha, user_age_group, user_channel = \
            self._get_user_features(uid, users_df, user_encoders, alpha)

        # ── Classes this user has already interacted with ──
        user_history = interactions_df[
            interactions_df["user_id"].astype(str) == uid
            ]["class_id"].astype(str).unique().tolist()

        seen_set = set(user_history)

        # ── Candidate classes (available, unseen) ──
        candidates = [
            cid for cid in class_index if cid not in seen_set
        ]
        if not candidates:
            logger.info("User %s: no unseen candidates — falling back to popularity.", uid)
            return self._popularity_fallback(pop_map, class_index, seen_set, n)

        # ── Class feature lookup ──
        c_cols = [c for c in CLASS_NUMERIC_COLS + CLASS_CAT_COLS
                  if c in classes_df.columns]
        c_lookup = classes_df.set_index("classid")[c_cols]

        # ── Content similarity: target class vs user's history ──
        cid_to_idx = {cid: i for i, cid in enumerate(class_index)}
        history_indices = [cid_to_idx[c] for c in user_history if c in cid_to_idx]

        # ── Score each candidate ──
        xgb_scores     = {}
        content_scores = {}
        attr_boost     = self._class_attribute_boost(seen_set, classes_df)

        # Pre-fetch user-history tfidf vectors for batch cosine sim
        if history_indices:
            history_vecs = tfidf_matrix[history_indices]
        else:
            history_vecs = None

        # Batch XGBoost scoring
        if xgb_model is not None and user_feats and feature_cols:
            batch_rows = []
            batch_cids = []
            for cid in candidates:
                if cid not in c_lookup.index:
                    continue
                c_feats = c_lookup.loc[cid].to_dict()
                # Content sim: mean similarity of this candidate to history
                target_idx = cid_to_idx.get(cid)
                csim = 0.0
                if target_idx is not None and history_vecs is not None:
                    sims = cosine_similarity(
                        tfidf_matrix[target_idx], history_vecs
                    )
                    csim = float(sims.mean())

                row = {**user_feats, **c_feats, "content_sim": csim}
                batch_rows.append(row)
                batch_cids.append(cid)

            if batch_rows:
                X_infer = pd.DataFrame(batch_rows).reindex(
                    columns=feature_cols, fill_value=0
                ).astype(float)
                preds = xgb_model.predict(X_infer)
                xgb_scores = dict(zip(batch_cids, preds.tolist()))

        # Content-based scores for all candidates
        for cid in candidates:
            target_idx = cid_to_idx.get(cid)
            if target_idx is None or history_vecs is None:
                content_scores[cid] = 0.0
                continue
            sims = cosine_similarity(tfidf_matrix[target_idx], history_vecs)
            content_scores[cid] = float(sims.mean())

        # ── Normalise ──
        xgb_scores     = self._normalize(xgb_scores)
        content_scores = self._normalize(content_scores)

        # Demographic peer boost
        demo_boost = self._demographic_boost(
            uid, user_age_group, user_channel,
            users_df, interactions_df, class_index
        )

        # ── Hybrid combination ──
        all_items     = set(candidates)
        hybrid_scores = {}
        for item in all_items:
            xgb  = xgb_scores.get(item, 0.0)
            cb   = content_scores.get(item, 0.0)
            pop  = pop_map.get(item, 0.0)
            demo = demo_boost.get(item, 0.0)
            attr = attr_boost.get(item, 0.0)

            hybrid_scores[item] = (
                    dynamic_alpha       * xgb  +
                    (1 - dynamic_alpha) * cb   +
                    0.10                * pop  +
                    0.05                * demo +
                    0.05                * attr
            )

        hybrid_scores = self._normalize(hybrid_scores)
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return ranked

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — guest users
    # ─────────────────────────────────────────────────────────────────────────
    def recommend_guest(self,
                        guest_user_id: str,
                        model_bundle: dict,
                        interactions_df: pd.DataFrame,
                        guest_df: pd.DataFrame = None,
                        user_encoders: dict = None,
                        n: int = 5) -> tuple:
        """
        Recommendation for a guest user.

        • If the guest has cart/order interactions → XGBoost hybrid
          (alpha forced low because cf_trust = 0, so content dominates)
        • Otherwise → pure content-based popularity fallback

        Returns (list of (class_id, score), rec_type_str)
        """
        xgb_model    = model_bundle["xgb_model"]
        tfidf_matrix = model_bundle["tfidf_matrix"]
        class_index  = model_bundle["class_index"]
        classes_df   = model_bundle["classes_df"]
        feature_cols = model_bundle["feature_cols"]
        pop_map      = model_bundle["interaction_popularity"]

        gid = str(guest_user_id)

        # Does this guest have any interactions?
        guest_history = interactions_df[
            interactions_df["user_id"].astype(str) == gid
            ]["class_id"].astype(str).unique().tolist()

        seen_set = set(guest_history)

        if not guest_history:
            # Pure popularity fallback
            recs = self._popularity_fallback(pop_map, class_index, seen_set, n)
            return recs, "popularity"

        # Has interactions → use hybrid with guest profile
        # Build guest feature dict (all unknowns / zeros except interaction signal)
        guest_feats = self._build_guest_features(gid, guest_df, user_encoders)

        c_cols   = [c for c in CLASS_NUMERIC_COLS + CLASS_CAT_COLS
                    if c in classes_df.columns]
        c_lookup = classes_df.set_index("classid")[c_cols]
        cid_to_idx = {cid: i for i, cid in enumerate(class_index)}

        history_indices = [cid_to_idx[c] for c in guest_history if c in cid_to_idx]
        history_vecs    = tfidf_matrix[history_indices] if history_indices else None

        candidates = [cid for cid in class_index if cid not in seen_set]

        xgb_scores     = {}
        content_scores = {}

        if xgb_model is not None and guest_feats and feature_cols:
            batch_rows = []
            batch_cids = []
            for cid in candidates:
                if cid not in c_lookup.index:
                    continue
                c_feats    = c_lookup.loc[cid].to_dict()
                target_idx = cid_to_idx.get(cid)
                csim = 0.0
                if target_idx is not None and history_vecs is not None:
                    sims = cosine_similarity(
                        tfidf_matrix[target_idx], history_vecs
                    )
                    csim = float(sims.mean())
                row = {**guest_feats, **c_feats, "content_sim": csim}
                batch_rows.append(row)
                batch_cids.append(cid)

            if batch_rows:
                X_infer = pd.DataFrame(batch_rows).reindex(
                    columns=feature_cols, fill_value=0
                ).astype(float)
                preds      = xgb_model.predict(X_infer)
                xgb_scores = dict(zip(batch_cids, preds.tolist()))

        for cid in candidates:
            target_idx = cid_to_idx.get(cid)
            if target_idx is None or history_vecs is None:
                content_scores[cid] = 0.0
                continue
            sims = cosine_similarity(tfidf_matrix[target_idx], history_vecs)
            content_scores[cid] = float(sims.mean())

        xgb_scores     = self._normalize(xgb_scores)
        content_scores = self._normalize(content_scores)

        # Guests have cf_trust=0 → alpha = 0.1 (content-heavy)
        alpha = 0.1
        hybrid_scores = {}
        for item in candidates:
            hybrid_scores[item] = (
                    alpha       * xgb_scores.get(item, 0.0) +
                    (1 - alpha) * content_scores.get(item, 0.0) +
                    0.10        * pop_map.get(item, 0.0)
            )

        hybrid_scores = self._normalize(hybrid_scores)
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return ranked, "hybrid"

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _get_user_features(self, uid: str, users_df, user_encoders, default_alpha):
        """
        Returns (feature_dict, dynamic_alpha, age_group, channel) for a user.
        Gracefully returns empty dict if user not found.
        """
        if users_df is None:
            return {}, default_alpha, None, None

        u_cols = [c for c in USER_NUMERIC_COLS + USER_CAT_COLS
                  if c in users_df.columns]
        row = users_df[users_df["userid"].astype(str) == uid]

        if row.empty:
            return {}, default_alpha, None, None

        row = row.iloc[0]

        cf_trust      = float(row.get("cf_trust", 0.0))
        dynamic_alpha = round(0.1 + cf_trust * 0.5, 4)   # [0.1, 0.6]
        age_group     = row.get("age_group")
        channel       = row.get("ads")

        logger.info(
            "User %s: cf_trust=%.2f  alpha=%.2f  age_group=%s  channel=%s",
            uid, cf_trust, dynamic_alpha, age_group, channel
        )

        feat_dict = {c: row[c] for c in u_cols if c in row.index}
        return feat_dict, dynamic_alpha, age_group, channel

    def _build_guest_features(self, gid: str, guest_df, user_encoders) -> dict:
        """
        Returns a feature dict for a guest user.
        All profile features default to 0 / encoded 'unknown'.
        """
        base = {c: 0 for c in USER_NUMERIC_COLS + USER_CAT_COLS}
        base["cf_trust"]      = 0.0
        base["is_cold_start"] = 1

        if guest_df is not None:
            row = guest_df[guest_df["guestuserid"].astype(str) == gid]
            if not row.empty:
                row = row.iloc[0]
                for c in USER_NUMERIC_COLS:
                    if c in row.index:
                        base[c] = float(row[c]) if pd.notna(row[c]) else 0
                # For categoricals, look up encoded value if encoders available
                if user_encoders:
                    for c, le in user_encoders.items():
                        if c in row.index:
                            val = str(row[c]) if pd.notna(row[c]) else "unknown"
                            base[c] = int(le.transform([val])[0]) \
                                if val in le.classes_ else 0
        return base

    def _demographic_boost(self, uid, age_group, channel, users_df,
                           interactions_df, class_index) -> dict:
        """
        Computes per-class popularity among demographic peers
        (same age_group × channel) of the given user.
        """
        if users_df is None or age_group is None or channel is None:
            return {}

        peer_mask = (
                (users_df["age_group"].astype(str) == str(age_group)) &
                (users_df["ads"].astype(str)       == str(channel))
        )
        peer_ids = set(users_df[peer_mask]["userid"].astype(str))
        peer_ids.discard(uid)

        if not peer_ids:
            return {}

        peer_inter = interactions_df[
            interactions_df["user_id"].astype(str).isin(peer_ids)
        ]
        if peer_inter.empty:
            return {}

        peer_pop  = peer_inter.groupby("class_id")["score"].sum().to_dict()
        max_pop   = max(peer_pop.values()) if peer_pop else 1.0
        return {cid: v / max_pop for cid, v in peer_pop.items()}

    def _class_attribute_boost(self, seen_set: set,
                               classes_df: pd.DataFrame) -> dict:
        """
        Boosts unseen classes that share teacher / time / cost / location
        tokens with classes the user has already seen.
        """
        if "content" not in classes_df.columns or not seen_set:
            return {}

        c_lookup = {str(row["classid"]): str(row.get("content", ""))
                    for _, row in classes_df.iterrows()}

        pref_tokens = set()
        for cid in seen_set:
            for token in c_lookup.get(cid, "").split():
                if any(token.startswith(p) for p in
                       ("teacher_", "time_", "cost_", "location_")):
                    pref_tokens.add(token)

        if not pref_tokens:
            return {}

        boost = {}
        for cid, content in c_lookup.items():
            if cid in seen_set:
                continue
            matches = len(pref_tokens & set(content.split()))
            if matches > 0:
                boost[cid] = matches

        if not boost:
            return {}
        max_m = max(boost.values())
        return {k: v / max_m for k, v in boost.items()}

    def _popularity_fallback(self, pop_map: dict, class_index: list,
                             seen_set: set = None, n: int = 5) -> list:
        """
        Returns top-n (class_id, score) sorted by global popularity.
        """
        if seen_set is None:
            seen_set = set()
        filtered = {cid: pop_map.get(cid, 0.0)
                    for cid in class_index if cid not in seen_set}
        return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:n]

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Min-max normalise a score dict to [0, 1]."""
        if not scores:
            return scores
        mn = min(scores.values())
        mx = max(scores.values())
        rng = mx - mn
        if rng == 0:
            return {k: 1.0 for k in scores}
        return {k: (v - mn) / rng for k, v in scores.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Display helpers
    # ─────────────────────────────────────────────────────────────────────────
    def show_recommendations_with_names(self, recommendations: list,
                                        classes_df: pd.DataFrame
                                        ) -> pd.DataFrame:
        """Joins class IDs with human-readable class names."""
        if not recommendations:
            return pd.DataFrame(columns=["classid", "score", "classname"])
        rec_df   = pd.DataFrame(recommendations, columns=["classid", "score"])
        final_df = rec_df.merge(
            classes_df[["classid", "classname"]], on="classid", how="left"
        )
        return final_df
