from __future__ import annotations

from collections import defaultdict
from typing import Any


def _prf(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = sorted(set(y_true))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    macro_p = macro_r = macro_f = 0.0
    for label in labels:
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_p += prec
        macro_r += rec
        macro_f += f1
    n = len(labels)
    return {
        "precision": macro_p / n if n else 0.0,
        "recall": macro_r / n if n else 0.0,
        "f1": macro_f / n if n else 0.0,
    }


def _hits_at_k(gold: str, option_scores: dict[str, float], k: int = 2) -> bool:
    if not option_scores:
        return False
    ranked = sorted(option_scores, key=lambda x: option_scores[x], reverse=True)
    return gold in ranked[:k]


def compute_metrics(results: list[dict[str, Any]], mode: str) -> dict[str, dict[str, Any]]:
    """Each entry in results needs qa_type, gold, preds[mode], option_scores[mode]."""
    by_op: dict[str, list] = defaultdict(list)
    for r in results:
        by_op[r["qa_type"]].append(r)

    report: dict[str, dict[str, Any]] = {}
    for op, items in list(by_op.items()) + [("Overall", results)]:
        if not items:
            continue
        y_true = [r["gold"] for r in items]
        y_pred = [r["preds"].get(mode) for r in items]
        prf = _prf(y_true, y_pred)
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(items)
        h2 = sum(
            _hits_at_k(r["gold"], r["option_scores"].get(mode, {}), k=2) for r in items
        ) / len(items)
        report[op] = {
            "n": len(items), "accuracy": acc, "precision": prf["precision"],
            "recall": prf["recall"], "macro_f1": prf["f1"], "hits_at_2": h2,
        }
    return report


def print_metrics(report: dict[str, dict[str, Any]], mode: str) -> None:
    print(f"\n{'-'*64}")
    print(f"  Metrics ({mode})")
    print(f"{'-'*64}")
    print(f"  {'Operator':<12} {'N':>4}  {'Acc':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'H@2':>6}")
    print(f"{'-'*64}")
    order = [k for k in report if k != "Overall"] + ["Overall"]
    for op in order:
        m = report[op]
        print(f"  {op:<12} {m['n']:>4}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['macro_f1']:>6.3f}  {m['hits_at_2']:>6.3f}")
    print(f"{'-'*64}")
