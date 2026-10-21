from pathlib import Path
import pandas as pd
import importlib.util


OUT = Path(
    r"C:\Users\Mike\Desktop\Predicting-Variation-in-Emotional-Valence-and-Arousal-over-Time-main\outputs"
)

EVAL_PATH = Path(
    r"eval.py"
)

# Load eval.py
spec = importlib.util.spec_from_file_location("official_eval", EVAL_PATH)
official_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_eval)


def create_val_m11_if_missing():
    val_m11_path = OUT / "val_m11.csv"

    if val_m11_path.exists():
        return val_m11_path

    val_m5 = pd.read_csv(OUT / "val_m5.csv")
    val_m10 = pd.read_csv(OUT / "val_m10.csv")

    merged = val_m5.merge(
        val_m10,
        on=["user_id", "text_id"],
        suffixes=("_m5", "_m10"),
    )

    w_m5 = 0.4
    w_m10 = 0.6

    val_m11 = pd.DataFrame({
        "user_id": merged["user_id"],
        "text_id": merged["text_id"],
        "true_valence": merged["true_valence_m5"],
        "true_arousal": merged["true_arousal_m5"],
        "pred_valence": (
            w_m5 * merged["pred_valence_m5"]
            + w_m10 * merged["pred_valence_m10"]
        ),
        "pred_arousal": (
            w_m5 * merged["pred_arousal_m5"]
            + w_m10 * merged["pred_arousal_m10"]
        ),
    })

    val_m11.to_csv(val_m11_path, index=False)
    return val_m11_path


def evaluate_one_file(path):
    df = pd.read_csv(path)

    valence = official_eval.task1_correlation(
        user_ids=df["user_id"],
        text_ids=df["text_id"],
        predictions=df["pred_valence"],
        labels=df["true_valence"],
    )

    arousal = official_eval.task1_correlation(
        user_ids=df["user_id"],
        text_ids=df["text_id"],
        predictions=df["pred_arousal"],
        labels=df["true_arousal"],
    )

    return {
        "Model": path.stem.replace("val_", "").upper(),

        "Valence r_btw": valence["r_between"],
        "Valence r_win": valence["r_within"],
        "Valence r_comp": valence["r_composite"],
        "Valence MAE_comp": valence["mae_composite"],

        "Arousal r_btw": arousal["r_between"],
        "Arousal r_win": arousal["r_within"],
        "Arousal r_comp": arousal["r_composite"],
        "Arousal MAE_comp": arousal["mae_composite"],
    }


def main():
    create_val_m11_if_missing()

    rows = []

    for i in range(1, 12):
        path = OUT / f"val_m{i}.csv"
        if path.exists():
            rows.append(evaluate_one_file(path))
        else:
            print(f"Missing {path.name}")

    table = pd.DataFrame(rows)

    metric_cols = [c for c in table.columns if c != "Model"]
    table[metric_cols] = table[metric_cols].round(4)

    print(table.to_string(index=False))

    out_path = OUT / "official_metric_table_for_report.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()