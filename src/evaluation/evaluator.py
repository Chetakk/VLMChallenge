import json
from pathlib import Path


def compute_oca(gt_dict, pred_dict):
    correct = 0
    total = len(gt_dict)

    for clip_id in gt_dict:
        if clip_id in pred_dict:
            if gt_dict[clip_id]["dominant_operation"] == pred_dict[clip_id]["dominant_operation"]:
                correct += 1

    return correct / total if total > 0 else 0


def compute_tiou_at_05(gt_dict, pred_dict):
    correct = 0
    total = len(gt_dict)

    for clip_id in gt_dict:
        if clip_id not in pred_dict:
            continue

        gt = gt_dict[clip_id]["temporal_segment"]
        pred = pred_dict[clip_id]["temporal_segment"]

        gt_start, gt_end = gt["start_frame"], gt["end_frame"]
        pred_start, pred_end = pred["start_frame"], pred["end_frame"]

        intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
        union = max(gt_end, pred_end) - min(gt_start, pred_start)

        if union > 0:
            iou = intersection / union
            if iou >= 0.5:
                correct += 1

    return correct / total if total > 0 else 0


def compute_aa_at_1(gt_dict, pred_dict):
    correct = 0
    total = len(gt_dict)

    for clip_id in gt_dict:
        if clip_id in pred_dict:
            if gt_dict[clip_id]["anticipated_next_operation"] == pred_dict[clip_id]["anticipated_next_operation"]:
                correct += 1

    return correct / total if total > 0 else 0


# -----------------------------
# Evaluate Single Model
# -----------------------------

def evaluate_model(gt_data, pred_data):
    gt_dict = {item["clip_id"]: item for item in gt_data}
    pred_dict = {item["clip_id"]: item for item in pred_data}

    return {
        "OCA": compute_oca(gt_dict, pred_dict),
        "tIoU@0.5": compute_tiou_at_05(gt_dict, pred_dict),
        "AA@1": compute_aa_at_1(gt_dict, pred_dict),
    }


# -----------------------------
# Main Evaluation Entry
# -----------------------------

def evaluate(
    gt_path,
    base_pred_path,
    finetuned_pred_path,
    output_path="results.json"
):

    gt_data = json.loads(Path(gt_path).read_text())
    base_pred_data = json.loads(Path(base_pred_path).read_text())
    finetuned_pred_data = json.loads(Path(finetuned_pred_path).read_text())

    results = {
        "base_model": evaluate_model(gt_data, base_pred_data),
        "finetuned_model": evaluate_model(gt_data, finetuned_pred_data)
    }

    Path(output_path).write_text(json.dumps(results, indent=2))

    print("Evaluation Results:")
    print(json.dumps(results, indent=2))


# -----------------------------
# CLI Execution
# -----------------------------

if __name__ == "__main__":
    evaluate(
        gt_path="data/eval/ground_truth.json",
        base_pred_path="data/eval/base_predictions.json",
        finetuned_pred_path="data/eval/finetuned_predictions.json",
        output_path="results.json"
    )