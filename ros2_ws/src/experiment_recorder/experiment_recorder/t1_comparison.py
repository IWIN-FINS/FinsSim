"""Reproducible aggregate analysis for frozen T1 PID/PPO result batches.

This module deliberately aggregates *trial-level* late-hold metrics instead
of treating ROS samples as independent experimental repetitions.  It creates a
paper-facing ledger, summary tables, paired target deltas, and compact plots
from immutable per-trial derived products.  Raw bags are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import yaml

from .manifest import file_sha256, now_iso, read_json, write_json


METRICS: tuple[tuple[str, str, float], ...] = (
    ("x_rmse_m", "x RMSE [m]", 1.0),
    ("y_rmse_m", "depth RMSE [m]", 1.0),
    ("z_rmse_m", "z RMSE [m]", 1.0),
    ("yaw_rmse_deg", "yaw RMSE [deg]", 1.0),
    ("horizontal_xz_rmse_m", "horizontal xz RMSE [m]", 1.0),
    ("position_3d_rmse_m", "3D position RMSE [m]", 1.0),
)
BASE_COLUMNS: tuple[str, ...] = (
    "domain",
    "method",
    "method_label",
    "action_interface",
    "run_directory",
    "run_status",
    "git_revision",
    "session_id",
    "target_id",
    "controller_config",
    "checkpoint",
    "checkpoint_sha256",
    "timing_provenance",
    "time_alignment_valid",
    "metrics_valid",
    "valid_fraction",
    "samples",
    "expected_samples",
    "x_rmse_m",
    "y_rmse_m",
    "z_rmse_m",
    "yaw_rmse_deg",
    "horizontal_xz_rmse_m",
    "position_3d_rmse_m",
    "x_p95_abs_m",
    "y_p95_abs_m",
    "z_p95_abs_m",
    "yaw_p95_abs_deg",
)


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a YAML mapping")
    return value


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_cli_path(value: Path, repo_root: Path) -> Path:
    """Accept a comparison config relative to the caller or repository root."""

    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    from_cwd = (Path.cwd() / path).resolve()
    return from_cwd if from_cwd.exists() else _resolve_path(str(path), repo_root)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"comparison YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid comparison YAML {path}: {exc}") from exc
    return _mapping(loaded, "comparison YAML")


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_timing_by_target(session_dir: Path) -> dict[str, str]:
    report_path = session_dir / "derived" / "analysis_report.json"
    if not report_path.is_file():
        return {}
    report = read_json(report_path)
    values = report.get("target_point_metrics")
    if not isinstance(values, list):
        return {}
    result: dict[str, str] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_id", "")).strip()
        if target:
            result[target] = str(item.get("timing_provenance", "unknown"))
    return result


def _simulation_time_alignment_valid(session_dir: Path, metadata: dict[str, Any]) -> bool:
    """Return whether a simulation T1 trial passed the lockstep audit.

    The historical accelerated free-running bags may contain apparently dense
    pose data while their command stream predates the received goal.  They are
    useful diagnostics but are not admissible controller-comparison evidence.
    Hardware sessions are intentionally outside this rule.
    """

    if not bool(metadata.get("simulation", False)):
        return True
    clock = metadata.get("simulation_clock") if isinstance(metadata.get("simulation_clock"), dict) else {}
    if str(clock.get("mode", "")) != "ros2_control_lockstep":
        return False
    report_path = session_dir / "derived" / "analysis_report.json"
    if not report_path.is_file():
        return False
    report = read_json(report_path)
    audit = report.get("t1_lockstep_alignment") if isinstance(report.get("t1_lockstep_alignment"), dict) else {}
    return bool(audit.get("passes", False))


def _load_run(
    *,
    domain: str,
    method: str,
    method_config: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_value = str(method_config.get("run_directory", "")).strip()
    if not run_value:
        raise SystemExit(f"domains.{domain}.methods.{method}.run_directory is required")
    run_dir = _resolve_path(run_value, repo_root)
    if not run_dir.is_dir():
        raise SystemExit(f"selected T1 run directory does not exist: {run_dir}")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.is_file() else {}
    result: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    for metrics_path in sorted(run_dir.glob("trials/**/derived/tracking_metrics.csv")):
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            metric = next(csv.DictReader(stream), None)
        if metric is None:
            continue
        session_dir = metrics_path.parents[1]
        session_manifest_path = session_dir / "manifest.json"
        session_manifest = read_json(session_manifest_path) if session_manifest_path.is_file() else {}
        metadata = session_manifest.get("metadata") if isinstance(session_manifest.get("metadata"), dict) else {}
        target_id = str(metric.get("target_id") or metadata.get("setpoint_id") or session_dir.name)
        timing = _report_timing_by_target(session_dir).get(target_id, str(metric.get("timing_provenance", "unknown")))
        time_alignment_valid = _simulation_time_alignment_valid(session_dir, metadata)
        x = _float(metric.get("x_rmse_m"))
        y = _float(metric.get("y_rmse_m"))
        z = _float(metric.get("z_rmse_m"))
        yaw_rad = _float(metric.get("yaw_rmse_rad"))
        row: dict[str, Any] = {
            "domain": domain,
            "method": method,
            "method_label": str(method_config.get("label", method)),
            "action_interface": str(method_config.get("action_interface", "unspecified")),
            "run_directory": str(run_dir.relative_to(repo_root)),
            "run_status": str(run_manifest.get("status", "unknown")),
            "git_revision": run_manifest.get("git_revision"),
            "session_id": session_manifest.get("session_id", session_dir.name),
            "target_id": target_id,
            "controller_config": metadata.get("controller_config"),
            "checkpoint": metadata.get("checkpoint"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "timing_provenance": timing,
            "time_alignment_valid": time_alignment_valid,
            "metrics_valid": _bool(metric.get("metrics_valid")) and time_alignment_valid,
            "valid_fraction": _float(metric.get("valid_fraction")),
            "samples": _float(metric.get("samples")),
            "expected_samples": _float(metric.get("expected_samples")),
            "x_rmse_m": x,
            "y_rmse_m": y,
            "z_rmse_m": z,
            "yaw_rmse_deg": math.degrees(yaw_rad) if yaw_rad is not None else None,
            "horizontal_xz_rmse_m": math.hypot(x, z) if x is not None and z is not None else None,
            "position_3d_rmse_m": math.sqrt(x * x + y * y + z * z) if x is not None and y is not None and z is not None else None,
            "x_p95_abs_m": _float(metric.get("x_p95_abs_m")),
            "y_p95_abs_m": _float(metric.get("y_p95_abs_m")),
            "z_p95_abs_m": _float(metric.get("z_p95_abs_m")),
            "yaw_p95_abs_deg": (
                math.degrees(value)
                if (value := _float(metric.get("yaw_p95_abs_rad"))) is not None
                else None
            ),
        }
        result.append(row)
    expected_targets = [str(item) for item in method_config.get("expected_target_ids", [])]
    found_targets = {str(row["target_id"]) for row in result}
    missing_targets = sorted(set(expected_targets) - found_targets)
    return result, {
        "run_directory": str(run_dir.relative_to(repo_root)),
        "run_status": run_manifest.get("status", "unknown"),
        "git_revision": run_manifest.get("git_revision"),
        "completed_trials": len(run_manifest.get("completed_trials", [])) if isinstance(run_manifest.get("completed_trials"), list) else None,
        "failed_trials": len(run_manifest.get("failed_trials", [])) if isinstance(run_manifest.get("failed_trials"), list) else None,
        "tracking_metric_files": len(result),
        "missing_expected_targets": missing_targets,
    }


def _summary(rows: list[dict[str, Any]], domain: str, method: str) -> dict[str, Any]:
    valid = [row for row in rows if row["metrics_valid"]]
    summary: dict[str, Any] = {
        "domain": domain,
        "method": method,
        "method_label": valid[0]["method_label"] if valid else method,
        "action_interface": valid[0]["action_interface"] if valid else "unspecified",
        "trial_count": len(rows),
        "valid_trial_count": len(valid),
        "valid_target_ids": ";".join(sorted(str(row["target_id"]) for row in valid)),
    }
    for key, _, _ in METRICS:
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        summary[f"{key}_mean"] = statistics.fmean(values) if values else None
        summary[f"{key}_median"] = statistics.median(values) if values else None
    return summary


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["metrics_valid"]]
    baseline = {
        (str(row["domain"]), str(row["target_id"])): row
        for row in valid
        if str(row["method"]) == "PID"
    }
    result: list[dict[str, Any]] = []
    for row in valid:
        if str(row["method"]) == "PID":
            continue
        reference = baseline.get((str(row["domain"]), str(row["target_id"])))
        if reference is None:
            continue
        item = {
            "domain": row["domain"],
            "target_id": row["target_id"],
            "method": row["method"],
            "baseline_method": "PID",
        }
        for key, _, _ in METRICS:
            candidate = row.get(key)
            base = reference.get(key)
            item[f"delta_{key}"] = (
                float(candidate) - float(base)
                if candidate is not None and base is not None
                else None
            )
        result.append(item)
    return result


def _plots(output_dir: Path, rows: list[dict[str, Any]], method_order: dict[str, list[str]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []
    generated: list[str] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"PID": "#4C78A8", "PPO_WRENCH6": "#F58518", "PPO_THRUSTER8": "#54A24B"}
    for domain, methods in method_order.items():
        valid = [row for row in rows if row["domain"] == domain and row["metrics_valid"]]
        if not valid:
            continue
        labels = {
            str(row["method"]): str(row["method_label"])
            for row in valid
        }
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
        for axis, (key, title, _) in zip(axes.flat, METRICS[:4]):
            for index, method in enumerate(methods):
                values = [float(row[key]) for row in valid if row["method"] == method and row.get(key) is not None]
                if not values:
                    continue
                offsets = [0.0] if len(values) == 1 else [0.16 * (item / (len(values) - 1) - 0.5) for item in range(len(values))]
                axis.scatter(
                    [index + offset for offset in offsets],
                    values,
                    color=colors.get(method, "#777777"),
                    alpha=0.85,
                    s=28,
                    zorder=3,
                )
                axis.hlines(statistics.fmean(values), index - 0.23, index + 0.23, color="black", linewidth=1.5)
            axis.set_title(title)
            axis.set_xticks(range(len(methods)), [labels.get(method, method) for method in methods], rotation=18, ha="right")
            axis.grid(True, axis="y", alpha=0.25)
        path = plot_dir / f"{domain}_per_target_rmse.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        generated.append(str(path.relative_to(output_dir)))

        targets = sorted({str(row["target_id"]) for row in valid})
        fig, axis = plt.subplots(figsize=(8.2, 3.9), constrained_layout=True)
        for method in methods:
            by_target = {str(row["target_id"]): row for row in valid if row["method"] == method}
            x_values = [index for index, target in enumerate(targets) if target in by_target]
            y_values = [float(by_target[target]["position_3d_rmse_m"]) for target in targets if target in by_target]
            if y_values:
                axis.plot(x_values, y_values, marker="o", linewidth=1.2, markersize=4.5, label=labels.get(method, method), color=colors.get(method, "#777777"))
        axis.set_xticks(range(len(targets)), targets, rotation=25, ha="right")
        axis.set_ylabel("3D position RMSE [m]")
        axis.set_title(f"T1 {domain}: target-level late-hold position error")
        axis.grid(True, alpha=0.25)
        axis.legend()
        path = plot_dir / f"{domain}_paired_3d_rmse.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        generated.append(str(path.relative_to(output_dir)))
    return generated


def _write_readme(output_dir: Path, report: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# T1 PID/PPO 对比分析结果",
        "",
        "本目录由 `compare_t1_experiments` 从指定的原始 trial 目录重建。统计单位为 trial/目标点，而不是 ROS 帧。",
        "",
        "- `trial_ledger.csv`：逐 trial 的来源、时基、原始四自由度误差与派生位置误差。",
        "- `method_summary.csv`：每个 domain/method 的目标点级均值与中位数。",
        "- `paired_deltas_vs_pid.csv`：相同目标点上 PPO 相对 PID 的差值；负值代表该 PPO 的误差更小。",
        "- `plots/`：逐目标散点/均值及三维位置 RMSE 配对图。",
        "",
        "## 解释边界",
        "",
        "实机 T1 误差相对于 `/finsrov/controller/pose` 的融合估计状态计算，不能替代 AprilTag 的独立绝对定位精度实验。仿真状态为 Unity/controller truth；因此不得直接把实机与仿真误差的差值解释为纯动力学 sim-to-real gap。",
        "",
        "若 `timing_provenance=position_goal_inferred_missing_hold_start`，该 trial 以记录到的、与 frozen target 完全匹配的 controller-world position goal 作为 hold 起点；若为 `phase_end_calibrated_missing_hold_start`，则使用同一 run 中有 marker trial 标定的 phase-end 延迟反推起点。两者均使用与 event start 相同的后 40--60 s 评价窗，且恢复来源保留在 ledger 中。",
        "",
        "## 当前汇总",
        "",
        "| Domain | Method | Valid trials | Mean 3D RMSE [m] | Mean yaw RMSE [deg] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        mean_3d = row.get("position_3d_rmse_m_mean")
        mean_yaw = row.get("yaw_rmse_deg_mean")
        mean_3d_text = f"{float(mean_3d):.4f}" if mean_3d is not None else "n/a"
        mean_yaw_text = f"{float(mean_yaw):.2f}" if mean_yaw is not None else "n/a"
        lines.append(
            f"| {row['domain']} | {row['method_label']} | {row['valid_trial_count']} | "
            f"{mean_3d_text} | {mean_yaw_text} |"
        )
    lines.extend([
        "",
        "完整选择条件、源 run 和版本信息见 `analysis_report.json`。",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(config_path: Path, *, force: bool = False) -> Path:
    """Run a frozen T1 comparison and return its output directory."""

    repo_root = _repo_root()
    config_path = _resolve_cli_path(config_path, repo_root)
    config = _read_yaml(config_path)
    analysis_id = str(config.get("analysis_id", "")).strip()
    if not analysis_id:
        raise SystemExit("comparison YAML requires analysis_id")
    output_value = str(config.get("output_directory", "")).strip()
    if not output_value:
        raise SystemExit("comparison YAML requires output_directory")
    output_dir = _resolve_path(output_value, repo_root)
    report_path = output_dir / "analysis_report.json"
    if report_path.exists() and not force:
        raise SystemExit(f"comparison output already exists: {output_dir}; use --force to regenerate it")
    domains = _mapping(config.get("domains"), "domains")
    all_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    method_order: dict[str, list[str]] = {}
    for domain, domain_value in domains.items():
        domain_config = _mapping(domain_value, f"domains.{domain}")
        methods = _mapping(domain_config.get("methods"), f"domains.{domain}.methods")
        method_order[str(domain)] = [str(method) for method in methods]
        for method, method_value in methods.items():
            rows, run_coverage = _load_run(
                domain=str(domain),
                method=str(method),
                method_config=_mapping(method_value, f"domains.{domain}.methods.{method}"),
                repo_root=repo_root,
            )
            all_rows.extend(rows)
            coverage[f"{domain}/{method}"] = run_coverage
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _summary(
            [row for row in all_rows if row["domain"] == domain and row["method"] == method],
            domain,
            method,
        )
        for domain, methods in method_order.items()
        for method in methods
    ]
    paired = _paired_deltas(all_rows)
    plots = _plots(output_dir, all_rows, method_order)
    _write_csv(output_dir / "trial_ledger.csv", all_rows, BASE_COLUMNS)
    summary_fields = [
        "domain", "method", "method_label", "action_interface", "trial_count", "valid_trial_count", "valid_target_ids",
        *[f"{key}_{statistic}" for key, _, _ in METRICS for statistic in ("mean", "median")],
    ]
    _write_csv(output_dir / "method_summary.csv", summaries, summary_fields)
    paired_fields = ["domain", "target_id", "method", "baseline_method", *[f"delta_{key}" for key, _, _ in METRICS]]
    _write_csv(output_dir / "paired_deltas_vs_pid.csv", paired, paired_fields)
    report = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "generated_at": now_iso(),
        "comparison_config": str(config_path.relative_to(repo_root)),
        "comparison_config_sha256": file_sha256(config_path),
        "statistical_unit": "trial_target_point",
        "selection_coverage": coverage,
        "method_order": method_order,
        "valid_trial_count": sum(1 for row in all_rows if row["metrics_valid"]),
        "all_trial_count": len(all_rows),
        "plots": plots,
        "summary": summaries,
        "comparability_notes": config.get("comparability_notes", []),
    }
    write_json(report_path, report)
    _write_readme(output_dir, report, summaries)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate a frozen T1 PID/PPO comparison batch.")
    parser.add_argument("--config", required=True, type=Path, help="Comparison YAML under experiment_recorder/config.")
    parser.add_argument("--force", action="store_true", help="Regenerate known comparison outputs in the selected output directory.")
    args = parser.parse_args(argv)
    output = run_comparison(args.config, force=args.force)
    print(output)


if __name__ == "__main__":
    main()
