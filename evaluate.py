def evaluate_frame(prediction, label):
    tp = 0
    fp = 0

    for p, l in zip(prediction, label):
        # get lower half for semantics
        p = p & 0xFFFF
        l = l & 0xFFFF

        if p == 0:
            continue

        if p == l:
            tp += 1
            continue

        fp += 1

    ap = tp / (tp + fp)

    return tp, fp, ap


def evaluate_sequence(predition, label):
    frames = len(predition)
    ap_sum = 0

    for p, l in zip(predition, label):
        tp, fp, ap = evaluate_frame(p, l)
        ap_sum += ap

    return ap_sum / frames
