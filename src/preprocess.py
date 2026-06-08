import logging
import sys
import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime, timedelta, time
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("preprocessing")

SOCIAL_CHANNELS = {"friends", "instagram", "youtube", "tiktok"}


class Preprocessing:

    # ── Registered users ──
    def preprocessing_users_data(self, df, channel_map, recency_map, time_of_day_map):
        logger.info("Preprocessing registered users...")
        today = pd.Timestamp(date.today())
        df = df.copy()

        df["userid"] = df["userid"].astype(str)
        df["userdob"] = pd.to_datetime(df["userdob"], errors="coerce")
        df["age"] = ((today - df["userdob"]).dt.days / 365.25).round(0).astype("Int64")
        df["age_missing"] = df["userdob"].isna().astype(int)
        df["age"] = df["age"].fillna(df["age"].median()).round(0).astype("Int64")

        df["age_group"] = pd.cut(
            df["age"].astype(float),
            bins=[0, 17, 24, 34, 49, 120],
            labels=["teen", "young_adult", "adult", "mid_career", "senior"],
            right=True
        ).astype(str).replace("nan", "unknown")

        birth_year = df["userdob"].dt.year
        df["generation"] = np.select(
            [birth_year >= 1997,
             (birth_year >= 1981) & (birth_year < 1997),
             (birth_year >= 1965) & (birth_year < 1981)],
            ["gen_z", "millennial", "gen_x"],
            default="unknown"
        )

        df["is_minor"] = (df["age"].astype(float) < 18).astype(int)

        df["usergender"] = (
            df["usergender"].astype(str).str.strip().str.lower()
            .replace({"none": "unknown", "nan": "unknown",
                      "null": "unknown", "": "unknown"})
        )
        df["gender_known"] = (df["usergender"] != "unknown").astype(int)
        df["gender_age_group"] = df["usergender"] + "_" + df["age_group"]

        df["ads"] = (
            df["ads"].astype(str).str.strip().str.lower()
            .replace({"nan": "unknown", "none": "unknown",
                      "null": "unknown", "": "unknown"})
        )
        df["is_social_referral"] = df["ads"].isin(SOCIAL_CHANNELS).astype(int)

        # df["channel_type"]   = channel_map["channel"].fillna("unknown").tolist()
        channel_map = channel_map.copy()
        channel_map = dict(zip(channel_map["channel"], channel_map["category"]))
        df["channel_type"]   = df["ads"].map(channel_map).fillna("unknown")

        referral_clean = df["referralcode"].astype(str).str.strip().str.upper()
        df["has_referral_code"] = (
            ~referral_clean.isin(["NULL", "NAN", "NONE", ""])
        ).astype(int)

        df["usertype"] = (
            df["usertype"].astype(str).str.strip().str.lower().fillna("unknown")
        )
        usertype_unique = df["usertype"].nunique()
        usertype_is_constant = (usertype_unique <= 1)

        if usertype_is_constant:
            logger.warning(
                "usertype column has only %d unique value(s) — "
                "is_student and channel_usertype may carry zero-variance bias.",
                usertype_unique
            )
            df["usertype_is_constant"] = 1
        else:
            df["usertype_is_constant"] = 0

        df["is_student"] = (df["usertype"] == "student").astype(int)

        if usertype_is_constant:
            df["channel_usertype"] = df["channel_type"]
        else:
            df["channel_usertype"] = df["channel_type"] + "_" + df["usertype"]

        df["userjoindate"] = pd.to_datetime(df["userjoindate"], errors="coerce")
        df["userjointime"] = df["userjoindate"].dt.strftime("%H:%M:%S")
        df["account_age_days"] = (
            (today - df["userjoindate"]).dt.days.fillna(0).astype(int)
        )

        # Convert 'inf' strings to actual float infinity and create bin edges
        upper  = recency_map["upper_limit"].replace("inf", float("inf")).astype(float).tolist()
        labels = recency_map["label"].tolist()
        bins = [-1] + upper

        df["join_recency_bucket"] = pd.cut(
            df["account_age_days"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")

        upper   = time_of_day_map["upper_limit"].astype(float).tolist()
        labels = time_of_day_map["label"].tolist()
        bins = [-1] + upper

        df["join_day_of_week"] = df["userjoindate"].dt.dayofweek
        df["join_is_weekend"] = (df["join_day_of_week"] >= 5).astype(int)
        df["join_hour"] = df["userjoindate"].dt.hour
        df["join_time_of_day"] = pd.cut(
            df["join_hour"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")
        df["join_month"] = df["userjoindate"].dt.month
        df["join_season"] = pd.cut(
            df["join_month"],
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"]
        ).astype(str).replace("nan", "unknown")

        df["userjoindate"] = df["userjoindate"].dt.date

        df["young_social"] = (
                df["age_group"].isin(["teen", "young_adult"]) &
                (df["is_social_referral"] == 1)
        ).astype(int)

        df["is_cold_start"] = (
                (df["has_referral_code"] == 0) &
                (df["account_age_days"] <= 30)
        ).astype(int)

        # cf_trust: composite score [0, 1] — how much CF weight to apply
        df["cf_trust"] = (
                (df["account_age_days"] > 30).astype(float) * 0.40 +
                df["gender_known"].astype(float)             * 0.30 +
                df["has_referral_code"].astype(float)        * 0.30
        )

        logger.info(
            "Registered user preprocessing done — %d rows, %d cols", *df.shape
        )
        return df

    # ── Guest users ───────────────────────────────────────────────────────────
    def preprocessing_guest_data(self, guest_df, time_of_day_map, recency_map):
        """
        Guest users always get cf_trust=0.0 and is_cold_start=1.
        Profile fields are filled with 'unknown' / 0 sentinels so XGBoost
        can process them through the same encoder as registered users.
        """
        logger.info("Preprocessing guest users...")
        today = pd.Timestamp(date.today())
        df = guest_df.copy()

        df["guestuserid"] = df["guestuserid"].astype(str)

        # ── Derived date components ─────────
        df["guestjointime"] = pd.to_datetime(df["guestjointime"], errors="coerce")
        df["userjoindate"]     = df["guestjointime"].dt.date
        df["join_hour"]        = df["guestjointime"].dt.hour
        df["join_day_of_week"] = df["guestjointime"].dt.dayofweek
        df["join_is_weekend"]  = (df["join_day_of_week"] >= 5).astype(int)
        df["join_month"]       = df["guestjointime"].dt.month

        # ── Time-of-day bucket (mirrors registered-user logic) ─────
        upper   = time_of_day_map["upper_limit"].astype(float).tolist()
        labels = time_of_day_map["label"].tolist()
        bins = [-1] + upper

        df["join_time_of_day"] = pd.cut(
            df["join_hour"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")

        # ── Season (mirrors registered-user logic) ─────
        df["join_season"] = pd.cut(
            df["join_month"],
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"]
        ).astype(str).replace("nan", "unknown")

        # ── Account age / recency bucket ────
        # ── Time-of-day bucket (mirrors registered-user logic) ─────
        upper   = recency_map["upper_limit"].astype(float).tolist()
        labels = recency_map["label"].tolist()
        bins = [-1] + upper

        df["account_age_days"] = (
            (today - df["guestjointime"]).dt.days.fillna(0).astype(int)
        )
        df["join_recency_bucket"] = pd.cut(
            df["account_age_days"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")

        # ── Preserve original guest-specific temporal columns for audit ───────
        df["guest_join_hour"]        = df["join_hour"]
        df["guest_join_time_of_day"] = df["join_time_of_day"]
        df["guest_join_day_of_week"] = df["join_day_of_week"]
        df["guest_join_is_weekend"]  = df["join_is_weekend"]

        # Placeholder values — consistent with USER_CAT_COLS encoding
        df["age_group"]          = "unknown"
        df["generation"]         = "unknown"
        df["usergender"]         = "unknown"
        df["gender_known"]       = 0
        df["ads"]                = "unknown"
        df["channel_type"]       = "unknown"
        df["is_social_referral"] = 0
        df["has_referral_code"]  = 0
        df["young_social"]       = 0
        df["gender_age_group"]   = "unknown_unknown"
        df["channel_usertype"]   = "unknown_guest"
        df["is_minor"]           = 0
        df["is_student"]         = 0
        df["usertype"]           = "guest"
        df["age_missing"]        = 1

        # XGBoost numeric features for cold-start / content-based path
        df["cf_trust"]       = 0.0
        df["is_cold_start"]  = 1

        logger.info(
            "Guest user preprocessing done — %d rows, %d cols", *df.shape
        )
        return df

    # ── Classes ──
    def preprocessing_classes_data(self, df, duration_map, cost_tier_map):
        logger.info("Preprocessing classes data...")
        df = df.copy()
        df["classid"] = df["classid"].astype(str)

        df["is_available"] = (
                df["classavailable"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)

        def parse_hour(t):
            try:
                return int(str(t).split(":")[0])
            except Exception:
                return -1

        df["start_hour"] = df["classstarttime"].apply(parse_hour)

        def time_of_day_token(h):
            if 6  <= h < 12: return "time_morning"
            if 12 <= h < 17: return "time_afternoon"
            if h  >= 17:     return "time_evening"
            return "time_unknown"

        df["time_of_day_token"] = df["start_hour"].apply(time_of_day_token)

        def get_minutes(t):
            try:
                parts = str(t).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return None

        start_mins = df["classstarttime"].apply(get_minutes)
        end_mins   = df["classendtime"].apply(get_minutes)
        df["class_duration_mins"] = (end_mins - start_mins).fillna(0)

        upper = duration_map["upper_limit"].replace("inf", float("inf")).astype(float).tolist()
        labels = duration_map["label"].tolist()
        bins = [-1] + upper

        df["duration_token"] = pd.cut(
            df["class_duration_mins"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")

        df["classteacher"] = df["classteacher"].astype(str).str.strip().str.upper()
        df["teacher_token"] = "teacher_" + df["classteacher"].str.lower().fillna("unknown")

        def normalise_location(loc):
            loc = str(loc).strip().lower()
            if ("1" in loc and "2" in loc) or ("&" in loc):
                return "studio_1_and_2"
            elif "studio 1" in loc or loc in ("studio1", "studio_1"):
                return "studio_1"
            elif "studio 2" in loc or loc in ("studio2", "studio_2"):
                return "studio_2"
            elif loc == "a":
                return "studio_a"
            else:
                return loc.replace(" ", "_").replace("&", "and")

        df["location_clean"] = df["classlocation"].apply(normalise_location)
        df["location_token"] = "location_" + df["location_clean"]

        df["classcost_numeric"] = pd.to_numeric(df["classcost"], errors="coerce").fillna(0)

        upper = cost_tier_map["upper_limit"].replace("inf", float("inf")).astype(float).tolist()
        labels = cost_tier_map["label"].tolist()
        bins = [-1] + upper

        df["cost_tier_token"] = pd.cut(
            df["classcost_numeric"],
            bins=bins,
            labels=labels
        ).astype(str).replace("nan", "unknown")

        # def cost_tier_token(c):
        #     if c <= 5:  return "cost_budget"
        #     if c <= 12: return "cost_standard"
        #     return "cost_premium"

        # df["cost_tier_token"] = df["classcost_numeric"].apply(cost_tier_token)

        max_cost = df["classcost_numeric"].max()
        df["cost_weight"] = (
            1.0 + (df["classcost_numeric"] / max_cost) if max_cost > 0 else 1.0
        )

        # TF-IDF content string (used by content-based branch)
        df["content"] = (
                df["classname"].fillna("") + " " +
                df["classtag"].fillna("")  + " " +
                df["teacher_token"]        + " " +
                df["location_token"]       + " " +
                df["time_of_day_token"]    + " " +
                df["duration_token"]       + " " +
                df["cost_tier_token"]
        )
        df["content"] = df["content"].str.replace(r"\s+", " ", regex=True).str.strip()

        logger.info(
            "Classes preprocessing done — %d rows | %d available | %d unavailable",
            len(df), df["is_available"].sum(), (df["is_available"] == 0).sum()
        )
        return df

    # ── Transactional tables ──────
    def preprocessing_cart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing shopping cart data...")
        df = df.copy()
        df["shoppingcartitemstudentid"] = df["shoppingcartitemstudentid"].astype(str)
        df["shoppingcartitemclassid"]   = df["shoppingcartitemclassid"].astype(str)
        df["shoppingcartid"]            = df["shoppingcartid"].astype(str)
        if "shoppingcartaddtime" in df.columns:
            df["shoppingcartaddtime"] = pd.to_datetime(df["shoppingcartaddtime"], errors="coerce")
        logger.info("Cart preprocessing done — %d rows", len(df))
        return df

    def preprocessing_enrollment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing enrollment data...")
        df = df.copy()
        df["enrollmentstudentid"] = df["enrollmentstudentid"].astype(str)
        df["enrollmentclassid"]   = df["enrollmentclassid"].astype(str)
        df["enrollmentstatus"]    = (
            df["enrollmentstatus"].astype(str).str.strip().str.lower()
        )
        df["enrollmentcheckindate"] = pd.to_datetime(
            df["enrollmentcheckindate"], errors="coerce"
        )
        invalid_mask = df["enrollmentcheckindate"].dt.year < 1900
        df.loc[invalid_mask, "enrollmentcheckindate"] = pd.NaT
        logger.info("Enrollment preprocessing done — %d rows", len(df))
        return df

    def preprocessing_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing sales order data...")
        df = df.copy()
        df["salesorderstudentid"] = df["salesorderstudentid"].astype(str)
        df["shoppingcartid"]      = df["shoppingcartid"].astype(str)
        df["creditcardapprovalstatus"] = (
            df["creditcardapprovalstatus"].astype(str).str.strip().str.lower()
        )
        df["orderdate"] = pd.to_datetime(df["orderdate"], errors="coerce")
        logger.info("Sales order preprocessing done — %d rows", len(df))
        return df

    def preprocessing_transfer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing transfer class data...")
        df = df.copy()
        df["transferstatus"] = (
            df["transferstatus"].astype(str).str.strip().str.lower()
            .replace({"not used": "not_used"})
        )
        df["transferdaterequest"] = pd.to_datetime(
            df["transferdaterequest"], errors="coerce"
        )
        logger.info("Transfer class preprocessing done — %d rows", len(df))
        return df

    def preprocessing_weight_data(self, order_weight_df, transfer_weight_df,
                                  enroll_weight_df) -> tuple:
        def _to_map(df: pd.DataFrame) -> dict:
            df = df.copy()
            df["status"] = df["status"].astype(str).str.strip().str.lower()
            df["value"]  = pd.to_numeric(df["value"], errors="coerce").fillna(0)
            return dict(zip(df["status"], df["value"]))

        order_map    = _to_map(order_weight_df)
        transfer_map = _to_map(transfer_weight_df)
        enroll_map   = _to_map(enroll_weight_df)

        logger.info(
            "Weight maps loaded — order: %d | transfer: %d | enroll: %d",
            len(order_map), len(transfer_map), len(enroll_map)
        )
        return order_map, transfer_map, enroll_map

