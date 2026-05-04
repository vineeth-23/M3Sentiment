import argparse
import csv
import glob
import math
import os
from collections import defaultdict


COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#4f46e5", "#be123c", "#65a30d", "#0f766e",
]

MODEL_DISPLAY_NAMES = {
    "late_fusion": "Late Fusion",
    "cross_modal": "Cross-Modal Fusion",
    "ortho_fusion": "Ortho Fusion",
    "aux_fusion": "Aux Fusion",
}

LEGACY_MODEL_NAMES = {
    "baseline1": "late_fusion",
    "baseline2": "cross_modal",
    "improved_ortho": "ortho_fusion",
    "orthogonality": "ortho_fusion",
    "improved_aux": "aux_fusion",
    "auxiliary": "aux_fusion",
}


def canonical_model_name(model_name):
    return LEGACY_MODEL_NAMES.get(model_name, model_name)


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(canonical_model_name(model_name), model_name.replace("_", " ").title())


def first_existing_csv(csv_dir, filenames):
    for filename in filenames:
        path = os.path.join(csv_dir, filename)
        if os.path.exists(path):
            return path
    return os.path.join(csv_dir, filenames[0])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def svg_text(x, y, text, size=12, weight="400", anchor="start", fill="#111827"):
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{text}</text>'


def scale(values, start, end):
    lo = min(values) if values else 0.0
    hi = max(values) if values else 1.0
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5

    def mapper(value):
        return start + (value - lo) * (end - start) / (hi - lo)

    return mapper, lo, hi


def write_svg(path, width, height, body):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Arial, sans-serif">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f"{body}\n"
            "</svg>\n"
        )


def line_chart(path, title, series, y_label, x_domain=None, y_domain=None):
    width, height = 980, 560
    left, right, top, bottom = 80, 220, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    plotted_series = {}
    points = []
    for name, rows in series.items():
        visible_rows = []
        for x, y in rows:
            if x_domain is not None and not (x_domain[0] <= x <= x_domain[1]):
                continue
            visible_rows.append((x, y))
            points.append((x, y))
        if visible_rows:
            plotted_series[name] = visible_rows

    if not points:
        return False

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_map, x_min, x_max = scale(xs, left, left + plot_w)
    if y_domain is None:
        y_map_raw, y_min, y_max = scale(ys, top + plot_h, top)
    else:
        y_min, y_max = y_domain
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5

        def y_map_raw(value):
            return top + plot_h - (value - y_min) * plot_h / (y_max - y_min)

    body = [
        svg_text(left, 35, title, size=20, weight="700"),
        svg_text(20, top + plot_h / 2, y_label, size=13, anchor="middle"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]

    for i in range(5):
        y_value = y_min + (y_max - y_min) * i / 4
        y = y_map_raw(y_value)
        body.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 8, y + 4, f"{y_value:.3f}", size=11, anchor="end", fill="#4b5563"))

    for i in range(5):
        x_value = x_min + (x_max - x_min) * i / 4
        x = x_map(x_value)
        body.append(svg_text(x, top + plot_h + 24, f"{x_value:.0f}", size=11, anchor="middle", fill="#4b5563"))

    for idx, (name, rows) in enumerate(plotted_series.items()):
        color = COLORS[idx % len(COLORS)]
        rows = sorted(rows)
        poly_points = " ".join(f"{x_map(x):.2f},{y_map_raw(y):.2f}" for x, y in rows)
        body.append(f'<polyline points="{poly_points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in rows:
            body.append(f'<circle cx="{x_map(x):.2f}" cy="{y_map_raw(y):.2f}" r="3" fill="{color}"/>')
        legend_y = top + 5 + idx * 22
        body.append(f'<rect x="{left + plot_w + 35}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        body.append(svg_text(left + plot_w + 55, legend_y, name, size=12))

    write_svg(path, width, height, "\n".join(body))
    return True


def grouped_bar_chart(path, title, groups, y_label, y_max=1.0):
    width, height = 1100, 560
    left, right, top, bottom = 80, 260, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not groups:
        return False

    labels = sorted({label for values in groups.values() for label in values})
    group_names = list(groups.keys())
    y_max = max(y_max, max((v for values in groups.values() for v in values.values()), default=1.0))
    y_max = max(0.01, y_max)

    def y_map(value):
        return top + plot_h - (value / y_max) * plot_h

    body = [
        svg_text(left, 35, title, size=20, weight="700"),
        svg_text(20, top + plot_h / 2, y_label, size=13, anchor="middle"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]

    for i in range(5):
        value = y_max * i / 4
        y = y_map(value)
        body.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 8, y + 4, f"{value:.2f}", size=11, anchor="end", fill="#4b5563"))

    group_w = plot_w / max(1, len(group_names))
    bar_w = min(28, group_w / max(1, len(labels)) * 0.72)

    for g_idx, group in enumerate(group_names):
        cx = left + group_w * g_idx + group_w / 2
        start_x = cx - (len(labels) * bar_w) / 2
        for l_idx, label in enumerate(labels):
            value = groups[group].get(label, 0.0)
            color = COLORS[l_idx % len(COLORS)]
            x = start_x + l_idx * bar_w
            y = y_map(value)
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.8:.2f}" height="{top + plot_h - y:.2f}" fill="{color}"/>')
        body.append(svg_text(cx, top + plot_h + 24, group, size=11, anchor="middle", fill="#374151"))

    for idx, label in enumerate(labels):
        legend_y = top + 5 + idx * 22
        color = COLORS[idx % len(COLORS)]
        body.append(f'<rect x="{left + plot_w + 35}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        body.append(svg_text(left + plot_w + 55, legend_y, label, size=12))

    write_svg(path, width, height, "\n".join(body))
    return True


def heatmap(path, title, rows, cols, values):
    width, height = 980, 560
    left, top = 180, 80
    cell_w, cell_h = 90, 45
    if not rows or not cols:
        return False

    max_value = max(values.values(), default=1.0)
    min_value = min(values.values(), default=0.0)
    span = max(max_value - min_value, 1e-9)

    body = [svg_text(left, 35, title, size=20, weight="700")]
    for c_idx, col in enumerate(cols):
        body.append(svg_text(left + c_idx * cell_w + cell_w / 2, top - 18, col, size=11, anchor="middle"))

    for r_idx, row in enumerate(rows):
        y = top + r_idx * cell_h
        body.append(svg_text(left - 12, y + cell_h / 2 + 4, row, size=11, anchor="end"))
        for c_idx, col in enumerate(cols):
            value = values.get((row, col), 0.0)
            intensity = (value - min_value) / span
            blue = int(245 - intensity * 145)
            fill = f"rgb({blue},{blue + 5},255)"
            x = left + c_idx * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff"/>')
            body.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, f"{value:.3f}", size=11, anchor="middle"))

    write_svg(path, width, height, "\n".join(body))
    return True


def plot_training_metrics(csv_dir, out_dir):
    metric_files = {
        "late_fusion": ["late_fusion_metrics.csv", "baseline1_metrics.csv"],
        "cross_modal": ["cross_modal_metrics.csv", "baseline2_metrics.csv"],
        "ortho_fusion": ["ortho_fusion_metrics.csv", "ortho_metrics.csv"],
        "aux_fusion": ["aux_fusion_metrics.csv", "aux_metrics.csv"],
    }
    train_acc_series = {}
    validation_acc_series = {}
    acc_series = {}
    loss_series = {}

    for name, filenames in metric_files.items():
        path = first_existing_csv(csv_dir, filenames)
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        if rows and "train_acc" in rows[0]:
            train_acc_series[display_model_name(name)] = [(to_float(r["epoch"]), to_float(r["train_acc"])) for r in rows]
        if rows and "val_acc" in rows[0]:
            validation_acc_series[display_model_name(name)] = [(to_float(r["epoch"]), to_float(r["val_acc"])) for r in rows]
        if rows and "test_acc" in rows[0]:
            acc_series[f"{display_model_name(name)} test"] = [(to_float(r["epoch"]), to_float(r["test_acc"])) for r in rows]
        if rows and "test_loss" in rows[0]:
            loss_series[f"{display_model_name(name)} test"] = [(to_float(r["epoch"]), to_float(r["test_loss"])) for r in rows]

        component_series = {}
        for col in [
            "classification_loss", "ortho_loss_raw", "ortho_loss_weighted",
            "main_loss", "aux_text_loss", "aux_audio_loss", "aux_video_loss", "aux_loss_weighted",
            "text_only_loss", "audio_only_loss", "vision_only_loss",
        ]:
            if rows and col in rows[0]:
                component_series[col] = [(to_float(r["epoch"]), to_float(r[col])) for r in rows]
        if component_series:
            line_chart(
                os.path.join(out_dir, f"{name}_loss_components.svg"),
                f"{display_model_name(name)} Loss Components",
                component_series,
                "loss",
            )

    line_chart(
        os.path.join(out_dir, "train_accuracy_by_model.svg"),
        "Training Accuracy by Model (Zoomed: Epochs 5+)",
        train_acc_series,
        "training accuracy",
        x_domain=(5, 40),
        y_domain=(0.65, 0.735),
    )
    line_chart(
        os.path.join(out_dir, "validation_accuracy_by_model.svg"),
        "Validation Accuracy by Model",
        validation_acc_series,
        "validation accuracy",
    )
    line_chart(os.path.join(out_dir, "test_accuracy_by_model.svg"), "Test Accuracy by Model", acc_series, "accuracy")
    line_chart(os.path.join(out_dir, "test_loss_by_model.svg"), "Test Loss by Model", loss_series, "loss")


def smooth_series(points, window=20):
    if window <= 1 or len(points) <= 2:
        return points
    smoothed = []
    running = []
    for x, y in sorted(points):
        running.append(y)
        if len(running) > window:
            running.pop(0)
        smoothed.append((x, sum(running) / len(running)))
    return smoothed


def plot_batch_metrics(csv_dir, out_dir):
    batch_files = {
        "late_fusion": ["late_fusion_batch_metrics.csv", "baseline1_batch_metrics.csv"],
        "cross_modal": ["cross_modal_batch_metrics.csv", "baseline2_batch_metrics.csv"],
        "ortho_fusion": ["ortho_fusion_batch_metrics.csv", "ortho_batch_metrics.csv"],
        "aux_fusion": ["aux_fusion_batch_metrics.csv", "aux_batch_metrics.csv"],
    }

    total_loss_series = {}
    for model_name, filenames in batch_files.items():
        path = first_existing_csv(csv_dir, filenames)
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        if not rows:
            continue

        def x_value(row, idx):
            step = to_float(row.get("global_step"), default=0.0)
            return step if step > 0 else idx + 1

        if "total_loss" in rows[0]:
            points = [(x_value(row, idx), to_float(row["total_loss"])) for idx, row in enumerate(rows)]
            total_loss_series[display_model_name(model_name)] = smooth_series(points)

        component_cols = [
            "classification_loss", "ortho_loss_raw", "ortho_loss_weighted",
            "main_loss", "aux_text_loss", "aux_audio_loss", "aux_video_loss", "aux_loss_weighted",
            "text_only_loss", "audio_only_loss", "vision_only_loss",
        ]
        component_series = {}
        for col in component_cols:
            if col in rows[0]:
                points = [(x_value(row, idx), to_float(row[col])) for idx, row in enumerate(rows)]
                component_series[col] = smooth_series(points)
        if component_series:
            line_chart(
                os.path.join(out_dir, f"{model_name}_batch_loss_components.svg"),
                f"{display_model_name(model_name)} Batch Loss Components",
                component_series,
                "loss",
            )

            epoch_series = {}
            epochs = sorted({int(row["epoch"]) for row in rows if row.get("epoch")})
            for col in component_cols:
                if col not in rows[0]:
                    continue
                points = []
                for epoch in epochs:
                    values = [
                        to_float(row[col])
                        for row in rows
                        if row.get("epoch") and int(row["epoch"]) == epoch and row.get(col) not in ("", None)
                    ]
                    if values:
                        points.append((epoch, sum(values) / len(values)))
                if points:
                    epoch_series[col] = points
            if epoch_series:
                line_chart(
                    os.path.join(out_dir, f"{model_name}_loss_components.svg"),
                    f"{display_model_name(model_name)} Epoch Loss Components",
                    epoch_series,
                    "loss",
                )

        if "batch_acc" in rows[0]:
            acc_points = [(x_value(row, idx), to_float(row["batch_acc"])) for idx, row in enumerate(rows)]
            line_chart(
                os.path.join(out_dir, f"{model_name}_batch_accuracy.svg"),
                f"{display_model_name(model_name)} Batch Accuracy",
                {display_model_name(model_name): smooth_series(acc_points)},
                "accuracy",
            )

    if total_loss_series:
        line_chart(
            os.path.join(out_dir, "batch_total_loss_by_model.svg"),
            "Batch Total Loss by Model",
            total_loss_series,
            "loss",
        )


def parse_diag_filename(path):
    base = os.path.basename(path).replace(".csv", "")
    for split in ("train", "val", "test"):
        suffix = f"_{split}"
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if "_batch_" in base:
        model, step = base.split("_batch_", 1)
        return canonical_model_name(model), f"batch_{step}"
    if "_epoch_" in base:
        model, epoch = base.split("_epoch_", 1)
        return canonical_model_name(model), f"epoch_{epoch}"
    if base.endswith("_final"):
        return canonical_model_name(base[:-6]), "final"
    return canonical_model_name(base), "unknown"


def stage_sort_key(stage):
    if stage.startswith("batch_"):
        return int(stage.split("_", 1)[1])
    if stage.startswith("epoch_"):
        return int(stage.split("_", 1)[1])
    if stage == "final":
        return 10**9
    return 10**9 + 1


def diagnostic_x_value(row):
    step = to_float(row.get("global_step"), default=0.0)
    if step > 0:
        return step
    return stage_sort_key(row["stage"])


def plot_diagnostic_metrics(csv_dir, out_dir):
    paths = glob.glob(os.path.join(csv_dir, "diagnostics", "*.csv"))
    if not paths:
        return

    diag_rows = []
    for path in paths:
        model, fallback_stage = parse_diag_filename(path)
        for row in read_csv(path):
            row["model"] = model
            row["stage"] = row.get("stage") or fallback_stage
            diag_rows.append(row)

    for model in sorted({r["model"] for r in diag_rows}):
        rows = [r for r in diag_rows if r["model"] == model]

        ortho_series = defaultdict(list)
        for row in rows:
            if row["category"] == "orthogonality" and row["metric"] in {
                "overall_cosine_abs_mean",
                "overall_cosine_squared_mean",
                "text_audio_cosine_abs_mean",
                "text_vision_cosine_abs_mean",
                "audio_vision_cosine_abs_mean",
            }:
                x = diagnostic_x_value(row)
                ortho_series[row["metric"]].append((x, to_float(row["value"])))
        if ortho_series:
            line_chart(
                os.path.join(out_dir, f"{model}_orthogonality_over_training.svg"),
                f"{display_model_name(model)} Orthogonality Over Training",
                dict(ortho_series),
                "similarity",
            )

        final_rows = [r for r in rows if r["stage"] == "final"]
        attention_over_time = defaultdict(list)
        for row in rows:
            if row["category"] == "cross_attention_overall":
                label = f'{row["query"].replace("_query", "")}->{row["attended_modality"]}'
                attention_over_time[label].append((diagnostic_x_value(row), to_float(row["value"])))
        if attention_over_time:
            line_chart(
                os.path.join(out_dir, f"{model}_cross_attention_over_training.svg"),
                f"{display_model_name(model)} Cross-Modal Attention Over Training",
                dict(attention_over_time),
                "attention weight",
            )

        fusion_over_time = defaultdict(list)
        for row in rows:
            if row["category"] == "fusion_attention" and row["metric"] == "mean_attention" and row.get("head") == "all":
                label = f'layer{row.get("layer", "")}->{row["attended_modality"]}'
                fusion_over_time[label].append((diagnostic_x_value(row), to_float(row["value"])))
        if fusion_over_time:
            line_chart(
                os.path.join(out_dir, f"{model}_fusion_attention_over_training.svg"),
                f"{display_model_name(model)} Fusion Attention Over Training",
                dict(fusion_over_time),
                "attention weight",
            )

        self_attention_over_time = {
            "mean_entropy": defaultdict(list),
            "mean_self_position_attention": defaultdict(list),
            "mean_other_position_attention": defaultdict(list),
        }
        for row in rows:
            if (
                row["category"] == "self_attention"
                and row.get("head") == "all"
                and row["metric"] in self_attention_over_time
            ):
                label = f'{row["query"]} layer{row.get("layer", "")}'
                self_attention_over_time[row["metric"]][label].append(
                    (diagnostic_x_value(row), to_float(row["value"]))
                )

        self_attention_plot_names = {
            "mean_entropy": ("entropy", "attention entropy"),
            "mean_self_position_attention": ("self_position_attention", "attention weight"),
            "mean_other_position_attention": ("other_position_attention", "attention weight"),
        }
        for metric, series in self_attention_over_time.items():
            if series:
                suffix, y_label = self_attention_plot_names[metric]
                line_chart(
                    os.path.join(out_dir, f"{model}_self_attention_{suffix}_over_training.svg"),
                    f"{model} Self-Attention {suffix.replace('_', ' ').title()} Over Training",
                    dict(series),
                    y_label,
                )

        attention_groups = defaultdict(dict)
        for row in final_rows:
            if row["category"] == "cross_attention_overall":
                group = row["query"].replace("_query", "")
                label = row["attended_modality"]
                attention_groups[group][label] = to_float(row["value"])
        if attention_groups:
            grouped_bar_chart(
                os.path.join(out_dir, f"{model}_final_cross_attention.svg"),
                f"{display_model_name(model)} Final Cross-Modal Attention",
                dict(attention_groups),
                "attention weight",
                y_max=1.0,
            )

        fusion_groups = defaultdict(dict)
        for row in final_rows:
            if row["category"] == "fusion_attention" and row["metric"] == "mean_attention" and row.get("head") == "all":
                fusion_groups[f'layer {row.get("layer", "")}'][row["attended_modality"]] = to_float(row["value"])
        if fusion_groups:
            grouped_bar_chart(
                os.path.join(out_dir, f"{model}_final_fusion_attention.svg"),
                f"{display_model_name(model)} Final Fusion Attention",
                dict(fusion_groups),
                "attention weight",
                y_max=1.0,
            )

        self_attention_groups = defaultdict(dict)
        for row in final_rows:
            if row["category"] == "self_attention" and row.get("head") == "all" and row["metric"] in {
                "mean_entropy",
                "mean_self_position_attention",
                "mean_other_position_attention",
            }:
                group = f'{row["query"]} layer {row.get("layer", "")}'
                self_attention_groups[group][row["metric"]] = to_float(row["value"])
        if self_attention_groups:
            grouped_bar_chart(
                os.path.join(out_dir, f"{model}_final_self_attention_summary.svg"),
                f"{display_model_name(model)} Final Within-Modality Attention",
                dict(self_attention_groups),
                "value",
            )

        head_rows = [r for r in final_rows if r["category"] == "cross_attention_head"]
        if head_rows:
            heat_values = {}
            row_labels = []
            col_labels = []
            for row in head_rows:
                row_label = f'{row["query"].replace("_query", "")} h{row["head"]}'
                col_label = row["attended_modality"]
                row_labels.append(row_label)
                col_labels.append(col_label)
                heat_values[(row_label, col_label)] = to_float(row["value"])
            heatmap(
                os.path.join(out_dir, f"{model}_final_head_attention_heatmap.svg"),
                f"{display_model_name(model)} Final Head-Wise Attention",
                sorted(set(row_labels)),
                sorted(set(col_labels)),
                heat_values,
            )

        fusion_head_rows = [
            r for r in final_rows
            if r["category"] == "fusion_attention" and r["metric"] == "mean_attention" and r.get("head") != "all"
        ]
        if fusion_head_rows:
            heat_values = {}
            row_labels = []
            col_labels = []
            for row in fusion_head_rows:
                row_label = f'layer{row.get("layer", "")} h{row["head"]}'
                col_label = row["attended_modality"]
                row_labels.append(row_label)
                col_labels.append(col_label)
                heat_values[(row_label, col_label)] = to_float(row["value"])
            heatmap(
                os.path.join(out_dir, f"{model}_final_fusion_head_attention_heatmap.svg"),
                f"{display_model_name(model)} Final Fusion Head Attention",
                sorted(set(row_labels)),
                sorted(set(col_labels)),
                heat_values,
            )


def main():
    parser = argparse.ArgumentParser(description="Create SVG plots from training metrics and diagnostics CSV files.")
    parser.add_argument("--csv-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs/plots/diagnostics")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    plot_training_metrics(args.csv_dir, args.out_dir)
    plot_batch_metrics(args.csv_dir, args.out_dir)
    plot_diagnostic_metrics(args.csv_dir, args.out_dir)
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
