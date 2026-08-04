"""
Threshold Ablation Study for Dyslexia Detector YOLO Model
============================================================
Purpose:
    Empirically justify the confidence threshold used to decide whether a
    detected letter is "reversal" / "corrected" vs "normal" — instead of
    citing borrowed literature values that don't actually apply to your
    model, your dataset, or your classes.

What this script does:
    1. Loads model/best.pt
    2. Runs YOLO validation on the synthetic dyslexia dataset at a sweep of
       confidence thresholds (default 0.15 -> 0.75, step 0.05)
    3. Records overall Precision / Recall / F1 / mAP50 at each threshold
    4. Records PER-CLASS Precision / Recall / F1 for normal, reversal, corrected
       (reversal is the clinically significant class, so it's reported and
       reasoned about separately from the overall numbers)
    5. Saves three files into OUTPUT_DIR:
        - threshold_ablation_results.csv   (raw numbers, every threshold)
        - threshold_ablation_f1_curve.png  (F1 vs threshold, overall + per class)
        - threshold_ablation_summary.txt   (plain-English findings, ready to
                                             drop into your report's evidence
                                             section)

Usage:
    python threshold_ablation.py

Requirements (in addition to your existing requirements.txt):
    pip install matplotlib --break-system-packages   (if not already installed)

Notes:
    - IMGSZ/IOU below are set to match what app.py actually uses at inference
      time (classify_letters_with_yolo calls model(..., conf=0.15, iou=0.5,
      imgsz=64)). If you change those in app.py, update them here too so the
      ablation stays representative of production behaviour.
    - This script validates against ONE dataset (synthetic_dyslexia_dataset).
      If you want to justify two different thresholds for upload vs. canvas
      input specifically, you need two threshold-labelled validation subsets
      (one per source) and to run this script against each separately — a
      single combined ablation run cannot justify two different production
      thresholds. See the note printed at the end of summary.txt.
"""

import os
import sys
import csv
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # no display needed, just save the PNG
    import matplotlib.pyplot as plt
except ImportError:
    print("❌ matplotlib is not installed. Run:")
    print("   pip install matplotlib --break-system-packages")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics is not installed. Run:")
    print("   pip install ultralytics --break-system-packages")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths if yours differ
# ─────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\model\best.pt"
DATA_YAML  = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\synthetic_dyslexia_dataset\data.yaml"
OUTPUT_DIR = r"C:\Users\visha\FYP\smartDyslexiaDetector_v2\validation_reports"

# IMGSZ = None lets YOLO validate at the model's native training resolution
# (this matches check_model.py, which is known to reproduce your ~99.9% report).
# NOTE: app.py's classify_letters_with_yolo() calls model(..., imgsz=64) at
# inference time on individual cropped letters, but forcing imgsz=64 during
# *validation* against the full dataset destroys spatial detail and collapses
# all detections to zero — confirmed by testing. Leave this as None unless you
# specifically rebuild the ablation around 64x64-cropped single-letter images
# to mirror production exactly.
IMGSZ = None
IOU   = 0.5

# Sweep range
CONF_START = 0.15
CONF_END   = 0.75
CONF_STEP  = 0.05

CLASS_NAMES = {0: 'normal', 1: 'reversal', 2: 'corrected'}


def frange(start, end, step):
    vals = []
    v = start
    while v <= end + 1e-9:
        vals.append(round(v, 2))
        v += step
    return vals


def get_per_class_metrics(metrics, class_names):
    """
    Extract per-class precision/recall/F1/AP50 from an ultralytics val() result.
    Returns None fields for any class not present in this validation run
    (e.g. zero instances at this threshold).
    """
    per_class = {name: {'precision': None, 'recall': None, 'f1': None, 'ap50': None}
                 for name in class_names.values()}
    try:
        ap_class_index = list(metrics.box.ap_class_index)
        for i, cls_id in enumerate(ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(i)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            name = class_names.get(int(cls_id), f'class_{cls_id}')
            per_class[name] = {
                'precision': round(float(p), 4),
                'recall': round(float(r), 4),
                'f1': round(float(f1), 4),
                'ap50': round(float(ap50), 4),
            }
    except Exception as e:
        print(f"   ⚠️ Could not extract per-class metrics: {e}")
    return per_class


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset yaml not found at {DATA_YAML}")
        return

    print("=" * 70)
    print("THRESHOLD ABLATION STUDY")
    print("=" * 70)
    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Sweep:   {CONF_START} -> {CONF_END} (step {CONF_STEP})")
    print(f"imgsz={'native (model default)' if not IMGSZ else IMGSZ}, iou={IOU}")
    print("=" * 70)

    model = YOLO(MODEL_PATH)
    thresholds = frange(CONF_START, CONF_END, CONF_STEP)

    rows = []
    for conf in thresholds:
        print(f"\n🔍 Validating at conf={conf} ...")
        try:
            if IMGSZ:
                metrics = model.val(data=DATA_YAML, conf=conf, iou=IOU, imgsz=IMGSZ, verbose=False)
            else:
                metrics = model.val(data=DATA_YAML, conf=conf, iou=IOU, verbose=False)
        except Exception as e:
            print(f"   ❌ Validation failed at conf={conf}: {e}")
            continue

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)

        per_class = get_per_class_metrics(metrics, CLASS_NAMES)
        reversal = per_class.get('reversal', {})

        row = {
            'threshold': conf,
            'precision_overall': round(precision, 4),
            'recall_overall': round(recall, 4),
            'f1_overall': round(f1, 4),
            'map50_overall': round(map50, 4),
            'map50_95_overall': round(map50_95, 4),
            'reversal_precision': reversal.get('precision'),
            'reversal_recall': reversal.get('recall'),
            'reversal_f1': reversal.get('f1'),
            'reversal_ap50': reversal.get('ap50'),
            'normal_f1': per_class.get('normal', {}).get('f1'),
            'corrected_f1': per_class.get('corrected', {}).get('f1'),
        }
        rows.append(row)

        print(f"   Overall  P={precision:.4f} R={recall:.4f} F1={f1:.4f} mAP50={map50:.4f}")
        if reversal.get('f1') is not None:
            print(f"   Reversal P={reversal['precision']:.4f} R={reversal['recall']:.4f} F1={reversal['f1']:.4f}")

    if not rows:
        print("❌ No successful validation runs — nothing to report.")
        return

    # ── Save CSV ──
    csv_path = os.path.join(OUTPUT_DIR, 'threshold_ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ CSV saved: {csv_path}")

    # ── Plot F1 vs threshold ──
    plt.figure(figsize=(9, 6))
    thr = [r['threshold'] for r in rows]
    plt.plot(thr, [r['f1_overall'] for r in rows], marker='o', label='Overall F1')
    if any(r['reversal_f1'] is not None for r in rows):
        plt.plot(thr, [r['reversal_f1'] for r in rows], marker='s', label='Reversal F1 (clinical class)')
    if any(r['normal_f1'] is not None for r in rows):
        plt.plot(thr, [r['normal_f1'] for r in rows], marker='^', linestyle='--', alpha=0.6, label='Normal F1')
    if any(r['corrected_f1'] is not None for r in rows):
        plt.plot(thr, [r['corrected_f1'] for r in rows], marker='d', linestyle='--', alpha=0.6, label='Corrected F1')

    best_overall = max(rows, key=lambda r: r['f1_overall'])
    plt.axvline(best_overall['threshold'], color='gray', linestyle=':', alpha=0.7,
                label=f"Peak overall F1 @ {best_overall['threshold']}")

    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence Threshold — Dyslexia Detector YOLO Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'threshold_ablation_f1_curve.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Plot saved: {plot_path}")

    # ── Write summary.txt ──
    best_reversal = None
    reversal_rows = [r for r in rows if r['reversal_f1'] is not None]
    if reversal_rows:
        best_reversal = max(reversal_rows, key=lambda r: r['reversal_f1'])

    # plateau range: thresholds within 1% (absolute F1) of the peak
    peak_f1 = best_overall['f1_overall']
    plateau = [r['threshold'] for r in rows if r['f1_overall'] >= peak_f1 - 0.01]
    plateau_range = (min(plateau), max(plateau)) if plateau else (best_overall['threshold'], best_overall['threshold'])

    summary_path = os.path.join(OUTPUT_DIR, 'threshold_ablation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("THRESHOLD ABLATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Dataset: {DATA_YAML}\n")
        f.write(f"Sweep: {CONF_START}-{CONF_END} step {CONF_STEP}, imgsz={IMGSZ}, iou={IOU}\n\n")

        f.write("OVERALL RESULT\n")
        f.write("-" * 60 + "\n")
        f.write(f"Peak overall F1 = {best_overall['f1_overall']:.4f} at threshold = {best_overall['threshold']}\n")
        f.write(f"  Precision at peak: {best_overall['precision_overall']:.4f}\n")
        f.write(f"  Recall at peak:    {best_overall['recall_overall']:.4f}\n")
        f.write(f"  mAP@0.5 at peak:   {best_overall['map50_overall']:.4f}\n")
        f.write(f"F1 stays within 1% of peak across thresholds {plateau_range[0]}-{plateau_range[1]}\n\n")

        if best_reversal:
            f.write("REVERSAL CLASS (clinically significant)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Peak reversal F1 = {best_reversal['reversal_f1']:.4f} at threshold = {best_reversal['threshold']}\n")
            f.write(f"  Precision: {best_reversal['reversal_precision']:.4f}\n")
            f.write(f"  Recall:    {best_reversal['reversal_recall']:.4f}\n")
            if best_reversal['threshold'] != best_overall['threshold']:
                f.write(f"\nNOTE: The reversal-optimal threshold ({best_reversal['threshold']}) differs from the "
                        f"overall-optimal threshold ({best_overall['threshold']}).\n"
                        f"Since reversal is the clinically significant class, you should justify your final "
                        f"production threshold using the REVERSAL row above rather than the overall row — "
                        f"missing a reversal (false negative) matters more here than a false positive on "
                        f"'normal'.\n")
            else:
                f.write("\nThe reversal-optimal threshold matches the overall-optimal threshold — "
                        "no conflict between overall accuracy and clinical sensitivity.\n")
        else:
            f.write("REVERSAL CLASS: per-class metrics unavailable (check ultralytics version / API).\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("HOW TO USE THIS IN YOUR REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(
            "1. Describe the sweep methodology (F1-vs-threshold curve across "
            f"{CONF_START}-{CONF_END}) and cite general literature establishing that this is "
            "standard practice for threshold selection in YOLO-based detectors.\n"
            "2. Report the numbers above as YOUR empirical result, from YOUR dataset and model —\n"
            "   this is the actual evidence, not a borrowed value.\n"
            "3. If you want separate thresholds for upload vs. canvas input, you must re-run this\n"
            "   script against threshold-labelled subsets for each source — a single combined\n"
            "   ablation run (like this one) cannot justify two different production thresholds.\n"
            "4. Your code currently hardcodes 0.5 for both the upload and canvas branches in\n"
            "   analyze_single_word(). Either (a) update the code to use the threshold(s) this\n"
            "   ablation justifies, or (b) update your report/documentation to describe the single\n"
            "   uniform threshold the code actually uses. Don't leave the two out of sync.\n"
        )

    print(f"✅ Summary saved: {summary_path}")
    print("\n" + "=" * 70)
    print("DONE. Use threshold_ablation_summary.txt as your report's evidence section,")
    print("threshold_ablation_f1_curve.png as the figure, and the CSV as your appendix data.")
    print("=" * 70)


if __name__ == '__main__':
    main()