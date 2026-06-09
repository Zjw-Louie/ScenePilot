#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from collections import Counter
from typing import Any, Dict, List


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["items", "records", "data", "scene_metrics", "metrics", "scenes"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported JSON format in: {path}")


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def calc_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n   # population std
    std = math.sqrt(var)

    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def merge_records(records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 不做去重，直接拼接
    merged = list(records_a) + list(records_b)
    return merged


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    room_type_counts = Counter()
    input_layout_counts = Counter()

    total_oob_loss_vals = []
    total_mbl_loss_vals = []
    total_pbl_loss_vals = []
    txt_pms_score_vals = []
    txt_pms_sampled_score_vals = []
    valid_scene_flags = []

    for item in records:
        room_type = item.get("room_type")
        if room_type is not None:
            room_type_counts[str(room_type)] += 1

        input_layout = item.get("input_layout")
        if input_layout is not None:
            input_layout_counts[str(input_layout)] += 1

        if is_number(item.get("total_oob_loss")):
            total_oob_loss_vals.append(float(item["total_oob_loss"]))

        if is_number(item.get("total_mbl_loss")):
            total_mbl_loss_vals.append(float(item["total_mbl_loss"]))

        if is_number(item.get("total_pbl_loss")):
            total_pbl_loss_vals.append(float(item["total_pbl_loss"]))

        if is_number(item.get("txt_pms_score")):
            txt_pms_score_vals.append(float(item["txt_pms_score"]))

        if is_number(item.get("txt_pms_sampled_score")):
            txt_pms_sampled_score_vals.append(float(item["txt_pms_sampled_score"]))

        if "is_valid_scene_pbl" in item:
            valid_scene_flags.append(bool(item["is_valid_scene_pbl"]))

    valid_scene_ratio_pbl = (
        sum(valid_scene_flags) / len(valid_scene_flags) if valid_scene_flags else None
    )

    summary = {
        "n_scenes": len(records),
        "room_type_counts": dict(room_type_counts),
        "input_layout_counts": dict(input_layout_counts),
        "physical_metrics": {
            "total_oob_loss": calc_stats(total_oob_loss_vals),
            "total_mbl_loss": calc_stats(total_mbl_loss_vals),
            "total_pbl_loss": calc_stats(total_pbl_loss_vals),
            "valid_scene_ratio_pbl": valid_scene_ratio_pbl,
        },
        "semantic_metrics": {
            "txt_pms_score": calc_stats(txt_pms_score_vals),
            "txt_pms_sampled_score": calc_stats(txt_pms_sampled_score_vals),
        },
    }
    return summary


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Merge two scene_metrics.json files and compute summary.")
    parser.add_argument(
        "--file1",
        type=str,
        default="/home2/zhangjiawei/respace/results/batch_outputs_baseline_123/eval_metrics/scene_metrics.json",
    )
    parser.add_argument(
        "--file2",
        type=str,
        default="/home2/zhangjiawei/respace/results/batch_outputs_baseline_456/eval_metrics/scene_metrics.json",
    )
    parser.add_argument(
        "--merged_out",
        type=str,
        default="/home2/zhangjiawei/respace/results/batch_outputs_baseline_merge/eval_metrics/scene_metrics_merged.json",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default="/home2/zhangjiawei/respace/results/batch_outputs_baseline_merge/eval_metrics/summary.json",
    )

    args = parser.parse_args()

    records1 = load_json_records(args.file1)
    records2 = load_json_records(args.file2)

    merged_records = merge_records(records1, records2)
    summary = summarize(merged_records)

    save_json(merged_records, args.merged_out)
    save_json(summary, args.summary_out)

    print("=" * 80)
    print(f"file1: {args.file1} ({len(records1)} records)")
    print(f"file2: {args.file2} ({len(records2)} records)")
    print(f"merged records: {len(merged_records)}")
    print(f"merged_out: {args.merged_out}")
    print(f"summary_out: {args.summary_out}")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()