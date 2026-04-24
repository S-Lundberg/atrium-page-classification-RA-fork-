import pandas as pd
import matplotlib.pyplot as plt


def compare_two_prediction_csv(gt_csv, pred_csv1, pred_csv2):
    df_gt = pd.read_csv(gt_csv)
    df1 = pd.read_csv(pred_csv1)
    df2 = pd.read_csv(pred_csv2)

    required_gt = {"local_filename", "true_cat"}
    required_pr = {"FILE", "PAGE", "CLASS-1", "CLASS-2"}

    if missing := (required_gt - set(df_gt.columns)):
        raise ValueError(f"GT CSV saknar kolumner: {missing}")
    if missing := (required_pr - set(df1.columns)):
        raise ValueError(f"Pred CSV 1 saknar kolumner: {missing}")
    if missing := (required_pr - set(df2.columns)):
        raise ValueError(f"Pred CSV 2 saknar kolumner: {missing}")

    # ============================
    # BUILD KEYS
    # ============================
    df_gt["key"] = df_gt["local_filename"].str.replace(".jpg", "", regex=False)
    df1["PAGE"] = df1["PAGE"].astype(str).str.zfill(5)
    df1["key"] = df1["FILE"].astype(str) + "_" + df1["PAGE"]
    df2["PAGE"] = df2["PAGE"].astype(str).str.zfill(5)
    df2["key"] = df2["FILE"].astype(str) + "_" + df2["PAGE"]

    # Merge alla tre
    df = df_gt.merge(df1, on="key", suffixes=("", "_p1"))
    df = df.merge(df2, on="key", suffixes=("_p1", "_p2"))

    # Döp om för tydlighet
    df = df.rename(columns={
        "CLASS-1_p1": "CLASS-1_1",
        "CLASS-2_p1": "CLASS-2_1",
        "CLASS-1_p2": "CLASS-1_2",
        "CLASS-2_p2": "CLASS-2_2",
    })

    df["err_1"] = df["CLASS-1_1"] != df["true_cat"]
    df["err_2"] = df["CLASS-1_2"] != df["true_cat"]

    # båda fel, samma felklass
    same_error = df[
        (df["err_1"]) &
        (df["err_2"]) &
        (df["CLASS-1_1"] == df["CLASS-1_2"])
    ][["local_filename", "true_cat", "CLASS-1_1", "CLASS-1_2"]]

    # båda fel, olika felklass
    diff_error = df[
        (df["err_1"]) &
        (df["err_2"]) &
        (df["CLASS-1_1"] != df["CLASS-1_2"])
    ][["local_filename", "true_cat", "CLASS-1_1", "CLASS-1_2"]]

    # bara modell 1 fel
    only_1_wrong = df[
        (df["err_1"]) &
        (~df["err_2"])
    ][["local_filename", "true_cat", "CLASS-1_1", "CLASS-1_2"]]

    # bara modell 2 fel
    only_2_wrong = df[
        (~df["err_1"]) &
        (df["err_2"])
    ][["local_filename", "true_cat", "CLASS-1_1", "CLASS-1_2"]]

    print("\n=== SAMMA FEL (båda fel, samma felklass) ===")
    print("Antal:", len(same_error))
    if not same_error.empty:
        print(same_error.to_string(index=False))

    print("\n=== OLIKA FEL (båda fel, olika felklass) ===")
    print("Antal:", len(diff_error))
    if not diff_error.empty:
        print(diff_error.to_string(index=False))

    print("\n=== ENDAST MODELL 1 FEL ===")
    print("Antal:", len(only_1_wrong))
    if not only_1_wrong.empty:
        print(only_1_wrong.to_string(index=False))

    print("\n=== ENDAST MODELL 2 FEL ===")
    print("Antal:", len(only_2_wrong))
    if not only_2_wrong.empty:
        print(only_2_wrong.to_string(index=False))

    # rätt TOP-1 i båda, men olika TOP-2
    correct_but_diff_top2 = df[
        (df["CLASS-1_1"] == df["true_cat"]) &
        (df["CLASS-1_2"] == df["true_cat"]) &
        (df["CLASS-2_1"] != df["CLASS-2_2"])
    ][["local_filename", "true_cat", "CLASS-2_1", "CLASS-2_2"]]

    return same_error, diff_error, only_1_wrong, only_2_wrong, correct_but_diff_top2


def save_table_plot(df, out_png, title="Correct but different TOP-2"):
    if df.empty:
        print(f"Inget att plotta för {out_png}")
        return

    fig, ax = plt.subplots(figsize=(10, min(0.4 * len(df), 20)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Tabell-plot sparad till {out_png}")


if __name__ == "__main__":
    #GT = r".\anno\anno_images.csv"
    GT = r"C:\RA_ufal_atrium\images_page\anno_page.csv" 
    PRED1 = r"C:\RA_ufal_atrium\Testing_results\atrium-20260421-0957_154_result_ViT-B16_main_TOP-3.csv"
    PRED2 = r"C:\RA_ufal_atrium\Testing_results\few-shot-20260423-1010_154_result_ViT-B16_main_TOP-3.csv"

    same_error, diff_error, only_1_wrong, only_2_wrong, correct_top2 = compare_two_prediction_csv(
        GT, PRED1, PRED2
    )

    correct_top2.to_csv("correct_but_diff_top2.csv", index=False)
    print(f"\nSparade correct_but_diff_top2.csv ({len(correct_top2)} rader)")

    save_table_plot(
        correct_top2,
        out_png="correct_but_diff_top2.png",
        title="Correct TOP-1, different TOP-2"
    )


