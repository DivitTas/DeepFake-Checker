import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =====================================================
# IMAGE-LEVEL EVALUATION
# =====================================================

def load_image_predictions(csv_path):
    """
    CSV format:
    image_path,label,score
    """
    y_true = []
    y_score = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["label"]))
            y_score.append(float(row["score"]))

    return y_true, y_score


def compute_image_metrics(y_true, y_score, threshold=0.5):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    return precision, recall, roc_auc, fpr


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Image Deepfake Detection")
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# VIDEO-LEVEL EVALUATION
# =====================================================

def load_frame_scores(csv_path):
    """
    CSV format:
    time,score
    """
    times = []
    scores = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            scores.append(float(row["score"]))

    return times, scores


def temporal_iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2

    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)

    return intersection / union if union > 0 else 0


def build_predicted_segments(frame_times, frame_scores, threshold=0.5):
    segments = []
    start = None
    confidences = []

    for t, s in zip(frame_times, frame_scores):
        if s >= threshold:
            if start is None:
                start = t
                confidences = []
            confidences.append(s)
        else:
            if start is not None:
                segments.append((start, t, np.mean(confidences)))
                start = None

    if start is not None:
        segments.append((start, frame_times[-1], np.mean(confidences)))

    return segments


def segment_level_metrics(gt_segments, pred_segments, tiou_threshold=0.5):
    matched_gt = set()
    tp = 0

    for ps, pe, _ in pred_segments:
        for i, (gs, ge) in enumerate(gt_segments):
            if i in matched_gt:
                continue
            if temporal_iou((ps, pe), (gs, ge)) >= tiou_threshold:
                tp += 1
                matched_gt.add(i)
                break

    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def detection_delay(gt_segments, pred_segments):
    delays = []

    for gs, _ in gt_segments:
        detected = [ps for ps, pe, _ in pred_segments if pe >= gs]
        if detected:
            delays.append(min(detected) - gs)

    return np.mean(delays) if delays else None


def plot_confidence_vs_time(frame_times, frame_scores, threshold=0.5):
    plt.figure(figsize=(10, 4))
    plt.plot(frame_times, frame_scores, label="Fake Confidence")
    plt.axhline(threshold, linestyle="--", color="red", label="Threshold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Confidence")
    plt.title("Deepfake Confidence vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # ---------------- IMAGE METRICS ----------------
    y_true, y_score = load_image_predictions("image_preds.csv")

    precision, recall, roc_auc, fpr = compute_image_metrics(y_true, y_score)

    print("\nIMAGE METRICS")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    plot_roc_curve(y_true, y_score)

    # ---------------- VIDEO METRICS ----------------
    frame_times, frame_scores = load_frame_scores("video_frame_scores.csv")

    # Ground truth segments (seconds)
    gt_segments = [
        (42, 50),
        (75, 82)
    ]

    pred_segments = build_predicted_segments(frame_times, frame_scores)

    seg_precision, seg_recall = segment_level_metrics(gt_segments, pred_segments)
    delay = detection_delay(gt_segments, pred_segments)

    print("\nVIDEO METRICS")
    print(f"Segment-level Precision: {seg_precision:.4f}")
    print(f"Segment-level Recall: {seg_recall:.4f}")
    print(f"Average Detection Delay (seconds): {delay}")

    plot_confidence_vs_time(frame_times, frame_scores)
