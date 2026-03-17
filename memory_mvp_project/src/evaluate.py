from typing import Dict, List


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / max(1, len(y_true))


def f1_macro(y_true: List[int], y_pred: List[int]) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    f1_values = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return sum(f1_values) / max(1, len(f1_values))


def binary_auc(y_true: List[int], y_score: List[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum = 0.0
    pos_count = 0
    neg_count = 0

    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
            pos_count += 1
        else:
            neg_count += 1

    if pos_count == 0 or neg_count == 0:
        return 0.5

    auc = (rank_sum - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return auc


def basic_metrics(y_true: List[int], y_pred: List[int], y_prob: List[List[float]]) -> Dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred),
    }

    class_count = len(y_prob[0]) if y_prob else 0
    if class_count == 2:
        pos_scores = [p[1] for p in y_prob]
        result["auc"] = binary_auc(y_true, pos_scores)
    return result
