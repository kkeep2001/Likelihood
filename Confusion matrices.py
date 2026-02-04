import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef

def compute_binary_metrics(
    label_excel_path,
    pred_excel_path,
    label_col,
    pred_col
):
    df_label = pd.read_excel(label_excel_path,sheet_name="label")
    df_pred = pd.read_excel(pred_excel_path)

    y_true = df_label[label_col].astype(int)
    y_pred = df_pred[pred_col].astype(int)

    assert len(y_true) == len(y_pred)
    assert set(y_true.unique()).issubset({0, 1})
    assert set(y_pred.unique()).issubset({0, 1})

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    ACC = accuracy_score(y_true, y_pred)
    F1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    BA = balanced_accuracy_score(y_true, y_pred)

    C = confusion_matrix(y_true, y_pred)
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        MCC = np.nan
    else:
        MCC = matthews_corrcoef(y_true, y_pred)

    results = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "F1": F1_macro,
        "MCC": MCC,
        "BA": BA,
        "ACC": ACC
    }


    return results


if __name__ == "__main__":
    metrics = compute_binary_metrics(
        label_excel_path="path/label.xlsx",
        pred_excel_path="path/predict.xlsx",
        label_col="label_col",
        pred_col="pred_col"
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")

