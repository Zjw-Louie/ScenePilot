from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path("/home2/zhangjiawei/respace/evaluate_date/evaluate_413")
FILENAME = "summary.json"

METRICS_PATHS = {
    "total_pbl_loss": ("final_metrics", "total_pbl_loss"),
    "final_dir_loss": ("final_dir_loss",),
    "final_rel_loss": ("final_rel_loss",),
    "final_func_loss": ("final_func_loss",),
    "final_score": ("final_score",),
}


def _get_by_path(d: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, (int, float)) and not (isinstance(cur, float) and (math.isnan(cur) or math.isinf(cur))):
        return float(cur)
    return None


def main() -> None:
    paths = sorted(ROOT.rglob(FILENAME))
    if not paths:
        raise SystemExit(f"未找到任何 {FILENAME} 于: {ROOT}")

    sums = {k: 0.0 for k in METRICS_PATHS}
    counts = {k: 0 for k in METRICS_PATHS}

    total_files = 0
    parsed_files = 0

    for p in paths:
        total_files += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        parsed_files += 1
        for name, path in METRICS_PATHS.items():
            v = _get_by_path(data, path)
            if v is None:
                continue
            sums[name] += v
            counts[name] += 1

    print(f"root: {ROOT}")
    print(f"found_files: {total_files}")
    print(f"parsed_json: {parsed_files}")
    print("averages:")
    for name in METRICS_PATHS:
        c = counts[name]
        avg = (sums[name] / c) if c else float("nan")
        print(f"  {name}: avg={avg:.6f}  (n={c})")


if __name__ == "__main__":
    main()