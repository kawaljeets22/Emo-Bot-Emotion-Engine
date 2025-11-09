# utils.py
def format_prediction(pred_dict):
    if not pred_dict:
        return "{}"
    # sort by score descending
    items = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k} ({v:.2f})" for k, v in items])
