#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_benchmark_by_ratio.py

按给定房间配比，从以下两个来源基于 scene_id 随机抽取测试样本：
1) /home2/zhangjiawei/respace/benchmark/empty_scenes_room_id
2) /home2/zhangjiawei/respace/benchmark/room_prompts_qwen3_vl/all_prompt_room_id.jsonl

默认采样配比：
- livingroom: 30
- bedroom: 30
- diningroom: 20
- library: 10
- laundry: 10

采样规则：
1) livingroom / bedroom / diningroom / library
   - 房间类型只以 empty_scenes_room_id 下的子文件夹名字为准
   - 不使用 split 强约束
   - 从对应子文件夹拿 scene_id
   - 再去 all_prompt_room_id.jsonl 里按 scene_id 找记录
   - 只要 scene_id 对上，就算候选

2) laundry
   - 不看 split
   - 直接在 all_prompt_room_id.jsonl 里找 prompt 含 "washing machine" 的记录
   - 再去 empty_scenes_room_id 整个目录树中按 scene_id 找对应空场景 json

输出：
1) jsonl 文件：每行一个采样样本
2) summary.json：记录采样统计信息

用法示例：
python /home2/zhangjiawei/respace/scripts/sample_benchmark_by_ratio.py

python /home2/zhangjiawei/respace/scripts/sample_benchmark_by_ratio.py --seed 123

python /home2/zhangjiawei/respace/scripts/sample_benchmark_by_ratio.py \
  --output /home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_seed123.jsonl \
  --summary_output /home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_seed123_summary.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_EMPTY_ROOT = "/home2/zhangjiawei/respace/benchmark/empty_scenes_room_id"
DEFAULT_PROMPT_JSONL = "/home2/zhangjiawei/respace/benchmark/room_prompts_qwen3_vl/all_prompt_room_id.jsonl"
DEFAULT_OUTPUT = "/home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio.jsonl"
DEFAULT_SUMMARY = "/home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_summary.json"

DEFAULT_COUNTS = {
    "livingroom": 30,
    "bedroom": 30,
    "diningroom": 20,
    "library": 10,
    "laundry": 10,
}


def normalize_room_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_counts(count_str: str) -> Dict[str, int]:
    """
    解析格式：
    livingroom=30,bedroom=30,diningroom=20,library=10,laundry=10
    """
    if not count_str.strip():
        return DEFAULT_COUNTS.copy()

    result = {}
    for item in count_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"无效的 counts 格式: {item}")
        k, v = item.split("=", 1)
        result[normalize_room_name(k.strip())] = int(v.strip())
    return result


def load_prompt_records(prompt_jsonl: Path) -> Dict[str, List[dict]]:
    """
    建立：
    scene_id -> [records...]

    不使用 split 强约束，因此这里只保留 prompt 的小写版本供 laundry 检索。
    """
    prompt_index = defaultdict(list)

    with prompt_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"解析 JSONL 失败，第 {line_no} 行: {e}")

            scene_id = obj.get("scene_id")
            if not scene_id:
                continue

            prompt = obj.get("prompt", "")
            obj["_normalized_prompt"] = str(prompt).lower()

            prompt_index[scene_id].append(obj)

    return prompt_index


def discover_top_level_room_folders(empty_root: Path) -> Dict[str, Path]:
    """
    扫描 empty_root 下一级子目录：
    normalized_folder_name -> folder_path
    """
    folder_map = {}
    for p in empty_root.iterdir():
        if p.is_dir():
            folder_map[normalize_room_name(p.name)] = p
    return folder_map


def collect_scene_files_in_folder(folder: Path) -> Dict[str, Path]:
    """
    收集某个房间文件夹下：
    scene_id -> json_path
    """
    result = {}
    for p in folder.glob("*.json"):
        result[p.stem] = p
    return result


def collect_scene_files_recursive(empty_root: Path) -> Dict[str, Path]:
    """
    递归收集整个 empty_root 下所有 json：
    scene_id -> json_path

    若 scene_id 重复出现，保留第一次遇到的。
    """
    result = {}
    for p in empty_root.rglob("*.json"):
        if p.stem not in result:
            result[p.stem] = p
    return result


def choose_prompt_record_any(records: List[dict]) -> Optional[dict]:
    """
    不看 split，只要 scene_id 对上，就取第一条记录。
    """
    return records[0] if records else None


def choose_prompt_record_for_laundry(records: List[dict]) -> Optional[dict]:
    """
    laundry 规则：
    prompt 中包含 'washing machine'
    """
    for r in records:
        prompt = r.get("_normalized_prompt", "")
        if "washing machine" in prompt:
            return r
    return None


def sample_standard_room(
    target_room: str,
    sample_count: int,
    room_folder: Path,
    prompt_index: Dict[str, List[dict]],
    rng: random.Random,
    used_scene_ids: set,
) -> List[dict]:
    """
    标准房间：
    - 房间类型只以 empty_scenes_room_id 下的子文件夹为准
    - 不使用 split 强约束
    - 只按 scene_id 交集匹配
    """
    scene_files = collect_scene_files_in_folder(room_folder)

    candidates = []
    matched_scene_only = 0

    for scene_id, empty_scene_path in scene_files.items():
        if scene_id in used_scene_ids:
            continue
        if scene_id not in prompt_index:
            continue

        matched_scene_only += 1

        prompt_record = choose_prompt_record_any(prompt_index[scene_id])
        if prompt_record is None:
            continue

        merged = dict(prompt_record)
        merged["target_room"] = target_room
        merged["empty_scene_room_id_path"] = str(empty_scene_path)
        merged["empty_scene_room_id_folder"] = room_folder.name
        candidates.append(merged)

    if len(candidates) < sample_count:
        raise ValueError(
            f"房间 '{target_room}' 可用候选不足：需要 {sample_count}，实际只有 {len(candidates)}。\n"
            f"对应文件夹: {room_folder}\n"
            f"scene_id 交集数: {matched_scene_only}"
        )

    rng.shuffle(candidates)
    selected = candidates[:sample_count]

    for x in selected:
        used_scene_ids.add(x["scene_id"])

    return selected


def sample_laundry(
    sample_count: int,
    all_scene_files: Dict[str, Path],
    prompt_index: Dict[str, List[dict]],
    rng: random.Random,
    used_scene_ids: set,
) -> List[dict]:
    """
    laundry 特殊规则：
    - 不看 split
    - 在 all_prompt_room_id.jsonl 中找 prompt 含 'washing machine' 的记录
    - 再去 empty_scenes_room_id 全目录树中按 scene_id 找空场景
    - 最终输出时强制把 split 改成 'laundry'
    """
    candidates = []

    for scene_id, records in prompt_index.items():
        if scene_id in used_scene_ids:
            continue
        if scene_id not in all_scene_files:
            continue

        prompt_record = choose_prompt_record_for_laundry(records)
        if prompt_record is None:
            continue

        empty_scene_path = all_scene_files[scene_id]

        merged = dict(prompt_record)
        merged["target_room"] = "laundry"
        merged["split"] = "laundry"
        merged["empty_scene_room_id_path"] = str(empty_scene_path)
        merged["empty_scene_room_id_folder"] = empty_scene_path.parent.name
        candidates.append(merged)

    if len(candidates) < sample_count:
        raise ValueError(
            f"房间 'laundry' 可用候选不足：需要 {sample_count}，实际只有 {len(candidates)}。\n"
            f"筛选条件：prompt 中包含 'washing machine'"
        )

    rng.shuffle(candidates)
    selected = candidates[:sample_count]

    for x in selected:
        used_scene_ids.add(x["scene_id"])

    return selected


def write_jsonl(records: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary(summary: dict, summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty_root", type=str, default=DEFAULT_EMPTY_ROOT)
    parser.add_argument("--prompt_jsonl", type=str, default=DEFAULT_PROMPT_JSONL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary_output", type=str, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--counts",
        type=str,
        default="livingroom=30,bedroom=30,diningroom=20,library=10,laundry=10",
        help="例如: livingroom=30,bedroom=30,diningroom=20,library=10,laundry=10",
    )
    args = parser.parse_args()

    empty_root = Path(args.empty_root)
    prompt_jsonl = Path(args.prompt_jsonl)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    if not empty_root.exists():
        raise FileNotFoundError(f"empty_root 不存在: {empty_root}")
    if not prompt_jsonl.exists():
        raise FileNotFoundError(f"prompt_jsonl 不存在: {prompt_jsonl}")

    counts = parse_counts(args.counts)
    rng = random.Random(args.seed)

    prompt_index = load_prompt_records(prompt_jsonl)
    folder_map = discover_top_level_room_folders(empty_root)
    all_scene_files = collect_scene_files_recursive(empty_root)

    used_scene_ids = set()
    all_selected = []
    room_resolution_info = {}

    standard_rooms = ["livingroom", "bedroom", "diningroom", "library"]

    for room in standard_rooms:
        if room not in counts:
            continue
        if room not in folder_map:
            raise FileNotFoundError(
                f"empty_scenes_room_id 下未找到房间文件夹: {room}\n"
                f"当前可用文件夹: {sorted(folder_map.keys())}"
            )

        selected = sample_standard_room(
            target_room=room,
            sample_count=counts[room],
            room_folder=folder_map[room],
            prompt_index=prompt_index,
            rng=rng,
            used_scene_ids=used_scene_ids,
        )
        all_selected.extend(selected)

        room_resolution_info[room] = {
            "mode": "folder_scene_id_match_only",
            "resolved_folder": folder_map[room].name,
            "selected_count": len(selected),
        }

    if "laundry" in counts:
        selected = sample_laundry(
            sample_count=counts["laundry"],
            all_scene_files=all_scene_files,
            prompt_index=prompt_index,
            rng=rng,
            used_scene_ids=used_scene_ids,
        )
        all_selected.extend(selected)

        room_resolution_info["laundry"] = {
            "mode": "prompt_contains_washing_machine",
            "resolved_folder": "ANY_UNDER_EMPTY_ROOT",
            "selected_count": len(selected),
        }

    rng.shuffle(all_selected)

    write_jsonl(all_selected, output_path)

    summary = {
        "seed": args.seed,
        "empty_root": str(empty_root),
        "prompt_jsonl": str(prompt_jsonl),
        "output": str(output_path),
        "total_selected": len(all_selected),
        "counts": counts,
        "room_resolution": room_resolution_info,
    }
    write_summary(summary, summary_path)

    print("=" * 80)
    print("采样完成")
    print(f"输出 jsonl: {output_path}")
    print(f"输出 summary: {summary_path}")
    print(f"总样本数: {len(all_selected)}")
    print("-" * 80)
    for room, info in room_resolution_info.items():
        print(
            f"{room:12s} -> mode={info['mode']:<32s} count={info['selected_count']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()