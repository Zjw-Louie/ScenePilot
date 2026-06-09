import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Set

import requests

from src.eval import eval_scene_before_after_with_delta
from scorer.parallel_actions import (
    StepActions,
    validate_step_actions,
    apply_actions_parallel_with_eval,
    parse_delta_token,
)


@dataclass
class GPTScoreResult:
    actions: Dict[str, Any]
    scores: Dict[str, Any]
    optimized_scene: Dict[str, Any]
    metrics: Dict[str, Any]


class GPTVLMScorer:
    """
    GPT 视觉动作优化器（actions+scores）：
    - 通过 YUNWU_AI 兼容接口发送：两张图片 + prompt
    - 要求模型输出 EXACTLY TWO JSON objects：
      (1) ACTIONS JSON (step/actions/default_op)
      (2) SCORES JSON (per_object[])
    - per_object 分数由模型给；global 分数由程序对 per_object 聚合计算
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key_env: str = "YUNWU_AI_API_KEY",
        timeout_s: int = 180,
    ) -> None:
        self.model = model
        self.api_base = api_base or os.environ.get("YUNWU_AI_API_BASE", "https://api.yunwu.ai/v1")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self.timeout_s = int(timeout_s)

    def _build_prompt(self, scene: Dict[str, Any], scene_summary_text: Optional[str] = None, step: int = 0) -> str:
        scene_str = json.dumps(scene, ensure_ascii=False)

        valid_jids: List[str] = []
        for obj in scene.get("objects", []):
            jid = obj.get("jid") or obj.get("sampled_jid") or obj.get("sampled_asset_jid")
            if isinstance(jid, str):
                valid_jids.append(jid)

        move_sizes = [0.02, 0.05, 0.10, 0.20]
        yaw_sizes = [2, 5, 10, 15]

        token_spec = (
            "ACTION SPACE (DISCRETE TOKENS):\n"
            f"- Translate tokens: MOVE_(dx,dy,dz) where (dx,dy,dz) is axis-aligned and step in {move_sizes}\n"
            "  Examples: MOVE_(+0.10,+0.00,+0.00), MOVE_(+0.00,-0.05,+0.00), MOVE_(+0.00,+0.00,-0.20)\n"
            f"- Yaw tokens (degrees): YAW_(+k) or YAW_(-k) where k in {yaw_sizes}\n"
            "- Noop token: noop\n"
            "RULES:\n"
            "- Output actions only for objects that need to change; others are implicitly noop (default_op).\n"
            "- Each object at most one action.\n"
            "- Prefer small moves first (0.02/0.05, yaw 2/5) unless clearly needed.\n"
            "- Goal: reduce out-of-bounds (OOB) and overlaps/collisions (BBL/MBL).\n"
        )

        actions_schema = (
            "ACTIONS OUTPUT (STRICT JSON #1):\n"
            "{\n"
            f'  "step": {int(step)},\n'
            '  "actions": [\n'
            '    {"jid": "VALID_JID", "op": "translate", "delta_token": "MOVE_(+0.05,+0.00,+0.00)"},\n'
            '    {"jid": "VALID_JID", "op": "rotate_yaw", "delta_token": "YAW_(-5)"}\n'
            "  ],\n"
            '  "default_op": "noop"\n'
            "}\n"
        )

        scores_schema = (
            "SCORES OUTPUT (STRICT JSON #2):\n"
            "{\n"
            '  "per_object": [\n'
            "    {\n"
            '      "jid": "VALID_JID",\n'
            '      "pos_score_0_10": 0.0,\n'
            '      "rot_score_0_10": 0.0,\n'
            '      "total_score_0_10": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Scoring rules:\n"
            '- For EACH object in scene.objects, output scores:\n'
            '- "pos_score_0_10" (0-10)\n'
            '- "rot_score_0_10" (0-10)\n'
            '- "total_score_0_10" = 0.5 * pos_score_0_10 + 0.5 * rot_score_0_10\n'
        )

        hard_rules = (
            "HARD OUTPUT RULES:\n"
            "1) Output EXACTLY TWO JSON objects, in this order: (1) ACTIONS JSON, then (2) SCORES JSON.\n"
            "2) No extra text. No markdown.\n"
            "3) Use double quotes only.\n"
            "4) Each JSON object must be valid: starts with '{' ends with '}'.\n"
        )

        parts: List[str] = []
        parts.append("You are a rigorous 3D indoor scene optimizer and rater.")
        parts.append("You are given TWO rendered images of the SAME scene (diag and top view) and the scene JSON.")
        parts.append("First propose ONE STEP of parallel object actions to reduce OOB and overlaps.")
        parts.append("Then rate the current scene quality by outputting per-object position/rotation scores.")
        parts.append(token_spec)
        parts.append(actions_schema)
        parts.append(scores_schema)
        parts.append(hard_rules)

        parts.append("Valid object identifiers (jid list):")
        parts.append(json.dumps(valid_jids, ensure_ascii=False))

        if scene_summary_text and scene_summary_text.strip():
            parts.append("\nCurrent scene summary:")
            parts.append(scene_summary_text.strip())

        parts.append("\nCurrent scene JSON (for reference, do NOT output it):")
        parts.append(scene_str)
        return "\n".join(parts)

    @staticmethod
    def _extract_all_json_objects_balanced(text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)

        out: List[str] = []
        n = len(text)
        i = 0
        while i < n:
            start = text.find("{", i)
            if start < 0:
                break

            depth = 0
            in_str = False
            esc = False
            for j in range(start, n):
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            out.append(text[start : j + 1])
                            i = j + 1
                            break
            else:
                break

        return out

    @staticmethod
    def _try_parse_json_relaxed(s: str) -> Dict[str, Any]:
        s0 = s.strip()
        try:
            return json.loads(s0)
        except json.JSONDecodeError:
            pass
        s2 = re.sub(r",\s*([}\]])", r"\1", s0)
        s2 = s2.replace("“", '"').replace("”", '"').replace("’", "'")
        s2 = s2.replace("'", '"')
        return json.loads(s2)

    @staticmethod
    def _scene_jid_set(scene: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for obj in scene.get("objects", []):
            jid = obj.get("jid") or obj.get("sampled_jid") or obj.get("sampled_asset_jid")
            if isinstance(jid, str):
                out.add(jid)
        return out

    @staticmethod
    def _scene_jid_to_obj(scene: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for obj in scene.get("objects", []):
            jid = obj.get("jid") or obj.get("sampled_jid") or obj.get("sampled_asset_jid")
            if isinstance(jid, str):
                out[jid] = obj
        return out

    @staticmethod
    def _object_importance(obj: Dict[str, Any]) -> float:
        size = obj.get("size")
        if not (isinstance(size, (list, tuple)) and len(size) == 3):
            return 0.0
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        footprint = max(sx, 0.0) * max(sz, 0.0)
        volume = max(sx, 0.0) * max(sy, 0.0) * max(sz, 0.0)
        return float(footprint + 0.25 * volume)

    def _priority_by_remaining_from_actions(
        self,
        scene: Dict[str, Any],
        step_actions: StepActions,
        *,
        k_yaw: float = 0.10,
        k_size: float = 1.00,
    ) -> Dict[str, float]:
        jid_to_obj = self._scene_jid_to_obj(scene)
        out: Dict[str, float] = {}
        for a in step_actions.actions:
            (dx, dy, dz), dyaw = parse_delta_token(a)
            move_mag = float(abs(dx) + abs(dy) + abs(dz))
            yaw_mag = float(abs(dyaw))
            imp = self._object_importance(jid_to_obj.get(a.jid, {}))
            out[a.jid] = k_size * imp + move_mag + k_yaw * yaw_mag
        return out

    @staticmethod
    def _validate_action_schema(step_actions: StepActions, scene: Dict[str, Any]) -> None:
        valid = GPTVLMScorer._scene_jid_set(scene)
        for a in step_actions.actions:
            if a.jid not in valid:
                raise ValueError(f"Unknown jid in actions: {a.jid}")

    @staticmethod
    def _clamp_0_10(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v != v:
            return 0.0
        return float(max(0.0, min(10.0, v)))

    def _validate_and_complete_scores(self, scores_json: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_jids = [
            (obj.get("jid") or obj.get("sampled_jid") or obj.get("sampled_asset_jid"))
            for obj in scene.get("objects", [])
        ]
        scene_jids = [j for j in scene_jids if isinstance(j, str)]
        valid_set = set(scene_jids)

        per = scores_json.get("per_object", [])
        if not isinstance(per, list):
            per = []

        by_jid: Dict[str, Dict[str, Any]] = {}
        for it in per:
            if not isinstance(it, dict):
                continue
            jid = it.get("jid")
            if not isinstance(jid, str) or jid not in valid_set:
                continue
            pos_s = self._clamp_0_10(it.get("pos_score_0_10", 0.0))
            rot_s = self._clamp_0_10(it.get("rot_score_0_10", 0.0))
            tot_s = 0.5 * pos_s + 0.5 * rot_s
            by_jid[jid] = {
                "jid": jid,
                "pos_score_0_10": float(round(pos_s, 4)),
                "rot_score_0_10": float(round(rot_s, 4)),
                "total_score_0_10": float(round(tot_s, 4)),
            }

        completed: List[Dict[str, Any]] = []
        for jid in scene_jids:
            if jid in by_jid:
                completed.append(by_jid[jid])
            else:
                completed.append({"jid": jid, "pos_score_0_10": 0.0, "rot_score_0_10": 0.0, "total_score_0_10": 0.0})

        if completed:
            global_pos = sum(x["pos_score_0_10"] for x in completed) / len(completed)
            global_rot = sum(x["rot_score_0_10"] for x in completed) / len(completed)
        else:
            global_pos = 0.0
            global_rot = 0.0
        global_total = 0.5 * global_pos + 0.5 * global_rot

        return {
            "per_object": completed,
            "global_pos_score_0_10": float(round(global_pos, 4)),
            "global_rot_score_0_10": float(round(global_rot, 4)),
            "global_total_score_0_10": float(round(global_total, 4)),
        }

    @staticmethod
    def _img_to_data_url(p: Path) -> str:
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _chat_completions(self, diag_image_path: Path, top_image_path: Path, prompt: str, *, temperature: float, max_tokens: int) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._img_to_data_url(diag_image_path)}},
                        {"type": "image_url", "image_url": {"url": self._img_to_data_url(top_image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    def score_and_optimize(
        self,
        diag_image_path: Path,
        top_image_path: Path,
        scene: Dict[str, Any],
        scene_summary_text: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        debug_dump_path: Optional[Path] = None,
        max_parse_retries: int = 2,
        step: int = 0,
        k_yaw: float = 0.10,
        k_size: float = 1.00,
    ) -> GPTScoreResult:
        prompt = self._build_prompt(scene, scene_summary_text=scene_summary_text, step=step)

        last_text: Optional[str] = None
        last_err: Optional[Exception] = None
        last_json_snips: List[str] = []

        for attempt in range(max_parse_retries + 1):
            temp = temperature if attempt == 0 else 0.0
            try:
                text = self._chat_completions(
                    diag_image_path=diag_image_path,
                    top_image_path=top_image_path,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=max_new_tokens,
                )
                last_text = text

                snips = self._extract_all_json_objects_balanced(text)
                last_json_snips = snips
                if len(snips) < 2:
                    raise ValueError(f"Expected 2 JSON objects (actions, scores), got {len(snips)}")

                actions_json: Optional[Dict[str, Any]] = None
                step_actions: Optional[StepActions] = None
                scores_json_raw: Optional[Dict[str, Any]] = None

                for sn in snips:
                    try:
                        parsed = self._try_parse_json_relaxed(sn)
                        sa = validate_step_actions(parsed)
                        self._validate_action_schema(sa, scene)
                        actions_json = parsed
                        step_actions = sa
                        break
                    except Exception:
                        continue
                if actions_json is None or step_actions is None:
                    raise ValueError("Could not find a valid ACTIONS JSON in model output")

                for sn in snips:
                    try:
                        parsed = self._try_parse_json_relaxed(sn)
                        if isinstance(parsed, dict) and isinstance(parsed.get("per_object", None), list):
                            scores_json_raw = parsed
                            break
                    except Exception:
                        continue
                if scores_json_raw is None:
                    raise ValueError("Could not find a valid SCORES JSON in model output")

                priority_by_remaining = self._priority_by_remaining_from_actions(scene, step_actions, k_yaw=k_yaw, k_size=k_size)

                scene_after = apply_actions_parallel_with_eval(
                    scene,
                    step_actions,
                    max_backoff_iters=50,
                    only_backoff_if_worse=True,
                    priority_by_remaining=priority_by_remaining,
                    debug=False,
                )

                scores_json = self._validate_and_complete_scores(scores_json_raw, scene)
                delta_metrics = eval_scene_before_after_with_delta(scene, scene_after, is_debug=False)

                return GPTScoreResult(actions=actions_json, scores=scores_json, optimized_scene=scene_after, metrics=delta_metrics)
            except Exception as e:
                last_err = e
                continue

        if debug_dump_path is not None:
            debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
            debug_dump_path.write_text(last_text or "", encoding="utf-8")
            (debug_dump_path.parent / "gpt_json_snips.txt").write_text("\n\n---\n\n".join(last_json_snips), encoding="utf-8")

        raise RuntimeError(f"Failed to parse GPT output after retries. Last error: {last_err}")