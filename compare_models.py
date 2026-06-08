"""
compare_models.py
─────────────────
Standalone comparison script — Option 1 (Shared Results File).

No dependency on either the ALS or XGBoost project.
Reads the long-format evaluation CSVs that each pipeline saves and
produces a side-by-side comparison including:
  • Train metrics (memorisation)
  • Test metrics  (generalisation)
  • GAP           (train − test, the overfitting signal)
  • Diagnosis     (OVERFITTING / UNDERFITTING / ACCEPTABLE GENERALISATION)

CSV SCHEMA (long format from evaluate.py save_report())
────────────────────────────────────────────────────────
  model | split | group | metric | value | diagnosis

  split  : "train" or "test"
  group  : "overall", "registered", "guest"
  metric : "precision@5", "recall@5", "ndcg@5", "mrr", etc.


Output
──────
  1. Console — TRAIN / TEST / GAP table per group per model
  2. Console — Diagnosis per model
  3. Console — Win-count summary (test metrics only)
  4. model_comparison.csv — long format with split, group, metric, model_name, value
"""

import sys
import os
import pandas as pd

# ── UPDATE THESE TWO PATHS ────────────────────────────────────────────────────
ALS_CSV = r"C:/Users/tamol/Desktop/Working/Other_Docs/US_Docs/TAMUC/521_Business_Analytics_Capstone/Group_Project/class_recommender/data/output/evaluation_als.csv"
XGB_CSV = r"C:/Users/tamol/Desktop/Working/Other_Docs/US_Docs/TAMUC/521_Business_Analytics_Capstone/Group_Project/class_recommender_XGBoost/data/output/evaluation_xgb.csv"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_comparison.csv"
)

METRIC_ORDER = [
    "precision@5", "recall@5", "ndcg@5", "mrr",
    "coverage_pct", "coverage_count", "novelty", "users_evaluated",
]
GROUPS = ["overall", "registered", "guest"]
SPLITS = ["train", "test"]


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_long_csv(path: str, label: str) -> pd.DataFrame:
    """
    Loads a long-format evaluation CSV produced by evaluate.py save_report().
    Schema: model | split | group | metric | value | diagnosis
    """
    if not os.path.exists(path):
        print(f"[ERROR] Cannot find {label} evaluation file:\n  {path}")
        print(f"  Run the {label} pipeline first to generate this file.")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        # Normalise column values
        df["split"]  = df["split"].astype(str).str.strip().str.lower()
        df["group"]  = df["group"].astype(str).str.strip().str.lower()
        df["metric"] = df["metric"].astype(str).str.strip()
        return df
    except Exception as exc:
        print(f"[ERROR] Failed to read {path}: {exc}")
        sys.exit(1)


def extract_diagnosis(df: pd.DataFrame) -> str:
    """Extracts the diagnosis string from any row that has one."""
    diag_col = df[df["diagnosis"].astype(str).str.strip() != ""]["diagnosis"]
    return diag_col.iloc[0] if not diag_col.empty else "No diagnosis available"


def extract_model_name(df: pd.DataFrame) -> str:
    return df["model"].iloc[0] if "model" in df.columns and not df.empty else "Unknown"


def pivot_to_lookup(df: pd.DataFrame) -> dict:
    """
    Converts long-format df to:
      {(split, group, metric): value}
    for fast O(1) lookup.
    """
    lookup = {}
    for _, row in df.iterrows():
        key = (str(row["split"]).lower(),
               str(row["group"]).lower(),
               str(row["metric"]))
        try:
            lookup[key] = float(row["value"])
        except (TypeError, ValueError):
            lookup[key] = row["value"]
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Comparison builders
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_df(als_lookup: dict, xgb_lookup: dict,
                        als_model: str, xgb_model: str) -> pd.DataFrame:
    """
    Builds a long-format comparison DataFrame with columns:
      split | group | metric | model_name | value | delta(XGB-ALS) | better

    Covers BOTH train and test splits so the full overfitting picture
    is visible in the output CSV.
    """
    rows = []
    for split in SPLITS:
        for group in GROUPS:
            for metric in METRIC_ORDER:
                key     = (split, group, metric)
                als_val = als_lookup.get(key)
                xgb_val = xgb_lookup.get(key)

                if als_val is None and xgb_val is None:
                    continue

                try:
                    als_f     = round(float(als_val), 4)
                    xgb_f     = round(float(xgb_val), 4)
                    delta     = round(xgb_f - als_f, 4)
                    winner    = "XGBoost" if delta > 0 else ("ALS" if delta < 0 else "Tie")
                    delta_str = f"{delta:+.4f}"
                except (TypeError, ValueError):
                    als_f, xgb_f, delta_str, winner = als_val, xgb_val, "n/a", "n/a"

                rows.append({
                    "split":          split.upper(),
                    "group":          group.upper(),
                    "metric":         metric,
                    "model_name":     als_model,
                    "value":          als_f,
                    "delta(XGB-ALS)": delta_str,
                    "better":         winner,
                })
                rows.append({
                    "split":          split.upper(),
                    "group":          group.upper(),
                    "metric":         metric,
                    "model_name":     xgb_model,
                    "value":          xgb_f,
                    "delta(XGB-ALS)": delta_str,
                    "better":         winner,
                })

    return pd.DataFrame(rows)


def build_gap_df(lookup: dict, model_name: str) -> pd.DataFrame:
    """
    Builds a GAP table (train − test) for a single model.
    Used to show the overfitting/underfitting signal per model.
    """
    rows = []
    for group in GROUPS:
        for metric in METRIC_ORDER:
            train_val = lookup.get(("train", group, metric))
            test_val  = lookup.get(("test",  group, metric))
            if train_val is None or test_val is None:
                continue
            try:
                gap = round(float(train_val) - float(test_val), 4)
                rows.append({
                    "model":      model_name,
                    "group":      group.upper(),
                    "metric":     metric,
                    "train":      round(float(train_val), 4),
                    "test":       round(float(test_val),  4),
                    "gap(tr-te)": f"{gap:+.4f}",
                })
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Console print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_model_gap_table(gap_df: pd.DataFrame, model_name: str, diagnosis: str):
    """
    Prints TRAIN / TEST / GAP table for one model.
    This is the primary overfitting/underfitting readout.
    """
    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  {model_name} — TRAIN vs TEST (Overfitting / Underfitting Check)")
    print(sep)

    for group in GROUPS:
        g_rows = gap_df[gap_df["group"] == group.upper()]
        if g_rows.empty:
            continue
        print(f"\n  ── {group.upper()} ──")
        print("  " + "─" * 70)
        print(f"  {'Metric':<22} {'TRAIN':>10} {'TEST':>10} {'GAP(tr-te)':>12}")
        print("  " + "─" * 70)
        for metric in METRIC_ORDER:
            row = g_rows[g_rows["metric"] == metric]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"  {metric:<22} {str(r['train']):>10} "
                f"{str(r['test']):>10} {str(r['gap(tr-te)']):>12}"
            )

    print(f"\n  DIAGNOSIS:")
    # Word-wrap diagnosis
    words, line = diagnosis.split(), "  "
    for word in words:
        if len(line) + len(word) > 80:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(sep + "\n")


def print_side_by_side(comp_df: pd.DataFrame, als_model: str, xgb_model: str,
                       split: str):
    """
    Prints a wide ALS vs XGBoost table for a given split (TRAIN or TEST).
    """
    sep  = "═" * 85
    rows = comp_df[comp_df["split"] == split.upper()]
    if rows.empty:
        return

    print(f"\n{sep}")
    print(f"  {split.upper()} METRICS — {als_model}  vs  {xgb_model}")
    print(sep)

    for group in GROUPS:
        g_rows = rows[rows["group"] == group.upper()]
        if g_rows.empty:
            continue
        print(f"\n  ── {group.upper()} ──")
        print("  " + "─" * 78)
        print(
            f"  {'Metric':<22} {als_model[:14]:>14} {xgb_model[:16]:>16} "
            f"{'Delta':>12} {'Better':>10}"
        )
        print("  " + "─" * 78)
        for metric in METRIC_ORDER:
            sub     = g_rows[g_rows["metric"] == metric]
            als_row = sub[sub["model_name"] == als_model]
            xgb_row = sub[sub["model_name"] == xgb_model]
            als_val = als_row["value"].values[0] if not als_row.empty else "n/a"
            xgb_val = xgb_row["value"].values[0] if not xgb_row.empty else "n/a"
            delta   = sub["delta(XGB-ALS)"].values[0] if not sub.empty else "n/a"
            better  = sub["better"].values[0]          if not sub.empty else "n/a"
            print(
                f"  {metric:<22} {str(als_val):>14} {str(xgb_val):>16} "
                f"{str(delta):>12} {str(better):>10}"
            )

    print(f"\n{'═' * 85}\n")


def print_win_summary(comp_df: pd.DataFrame, split: str = "TEST"):
    """Win counts on test metrics (generalisation is what matters for deployment)."""
    rows = comp_df[
        (comp_df["split"] == split.upper()) &
        (comp_df["better"].isin(["ALS", "XGBoost", "Tie"]))
    ].drop_duplicates(subset=["split", "group", "metric"])

    print(f"  WIN SUMMARY ({split.upper()} metrics)")
    print("  " + "─" * 58)
    for group in GROUPS:
        g = rows[rows["group"] == group.upper()]
        if g.empty:
            continue
        c = g["better"].value_counts().to_dict()
        print(
            f"  {group.upper():<14}  ALS: {c.get('ALS', 0)}  |  "
            f"XGBoost: {c.get('XGBoost', 0)}  |  Tie: {c.get('Tie', 0)}"
        )
    print("  " + "─" * 58)
    print(
        f"  {'TOTAL':<14}  ALS: {(rows['better'] == 'ALS').sum()}  |  "
        f"XGBoost: {(rows['better'] == 'XGBoost').sum()}  |  "
        f"Tie: {(rows['better'] == 'Tie').sum()}"
    )
    print()


def print_diagnosis_summary(als_diagnosis: str, xgb_diagnosis: str,
                             als_model: str, xgb_model: str):
    """Prints both model diagnoses side by side for easy comparison."""
    sep = "═" * 82
    print(f"\n{sep}")
    print("  OVERFITTING / UNDERFITTING DIAGNOSIS SUMMARY")
    print(sep)

    def _extract_verdict(diag: str) -> str:
        """Extracts just the verdict word before the first |"""
        return diag.split("|")[0].strip() if "|" in diag else diag[:40]

    als_verdict = _extract_verdict(als_diagnosis)
    xgb_verdict = _extract_verdict(xgb_diagnosis)

    print(f"\n  {als_model:<30}  →  {als_verdict}")
    print(f"  {xgb_model:<30}  →  {xgb_verdict}")

    print(f"\n  {als_model} detail:")
    words, line = als_diagnosis.split(), "    "
    for word in words:
        if len(line) + len(word) > 80:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n  {xgb_model} detail:")
    words, line = xgb_diagnosis.split(), "    "
    for word in words:
        if len(line) + len(word) > 80:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load long-format CSVs
    als_df = load_long_csv(ALS_CSV, "ALS")
    xgb_df = load_long_csv(XGB_CSV, "XGBoost")

    als_model    = extract_model_name(als_df)
    xgb_model    = extract_model_name(xgb_df)
    als_diagnosis = extract_diagnosis(als_df)
    xgb_diagnosis = extract_diagnosis(xgb_df)

    als_lookup = pivot_to_lookup(als_df)
    xgb_lookup = pivot_to_lookup(xgb_df)

    # ── 1. GAP tables per model (overfitting readout) ─────────────────────
    als_gap_df = build_gap_df(als_lookup, als_model)
    xgb_gap_df = build_gap_df(xgb_lookup, xgb_model)

    print_model_gap_table(als_gap_df, als_model, als_diagnosis)
    print_model_gap_table(xgb_gap_df, xgb_model, xgb_diagnosis)

    # ── 2. Combined diagnosis summary ─────────────────────────────────────
    print_diagnosis_summary(als_diagnosis, xgb_diagnosis, als_model, xgb_model)

    # ── 3. Side-by-side TEST metrics (generalisation comparison) ──────────
    comp_df = build_comparison_df(als_lookup, xgb_lookup, als_model, xgb_model)
    print_side_by_side(comp_df, als_model, xgb_model, split="TEST")

    # ── 4. Side-by-side TRAIN metrics (memorisation comparison) ───────────
    print_side_by_side(comp_df, als_model, xgb_model, split="TRAIN")

    # ── 5. Win summary on test metrics ────────────────────────────────────
    print_win_summary(comp_df, split="TEST")

    # ── 6. Save output CSV ────────────────────────────────────────────────
    # Add diagnosis column to comparison df for downstream use
    comp_df["als_diagnosis"] = als_diagnosis
    comp_df["xgb_diagnosis"] = xgb_diagnosis

    # Also append gap rows for each model
    als_gap_df["model_name"] = als_model
    xgb_gap_df["model_name"] = xgb_model
    gap_combined = pd.concat([als_gap_df, xgb_gap_df], ignore_index=True)

    # Save two sheets — comparison and gap — as separate CSVs
    comp_df.to_csv(OUTPUT_CSV, index=False)
    gap_path = OUTPUT_CSV.replace(".csv", "_gap.csv")
    gap_combined.to_csv(gap_path, index=False)

    print(f"  Saved comparison (long format) → {OUTPUT_CSV}")
    print(f"  Saved gap table                → {gap_path}")
    print(f"  Schema (comparison): split | group | metric | model_name | value | "
          f"delta(XGB-ALS) | better | als_diagnosis | xgb_diagnosis")
    print(f"  Schema (gap):        model | group | metric | train | test | gap(tr-te)\n")


if __name__ == "__main__":
    main()
