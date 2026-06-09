#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Importable Planning RAG module for scene planning.

This module converts the previous CLI-style planning_rag_hook into a directly
importable Python module that fits the user's current inference architecture.

Main entry points
-----------------
1) PlanningRAGRetriever(...)
   - reusable retriever object
   - load once, use many times

2) build_planning_rag_hint(...)
   - one-shot helper
   - returns a prompt-ready prior hint string and the raw retrieval bundle

3) build_augmented_extra_hints(...)
   - appends the retrieved prior hint to your existing extra_hints_text

4) infer_anchor_candidates(...)
   - heuristic anchor extraction from scene / role categories / prompt text

Recommended integration
-----------------------
In the user's current architecture, `extra_hints_text` is passed into
`optimize_scene_refactored_v13(...)` and then combined by
`_compose_step_extra_context(...)`. Therefore the minimal-intrusion way
to integrate RAG is:

    from src.planning_rag import PlanningRAGRetriever, build_augmented_extra_hints

    rag = PlanningRAGRetriever(
        index_dir="/path/to/faiss_index_augmented",
        model_name_or_path="/path/to/qwen3-embedding-8B",
    )

    role_graph = infer_role_graph(current_scene)
    extra_hints_text = build_augmented_extra_hints(
        retriever=rag,
        base_extra_hints_text=extra_hints_text,
        scene=current_scene,
        user_prompt=room_prompt,
        room_type=current_room_type,
        role_categories=role_graph.categories,
    )

Then continue your existing pipeline unchanged.

Outputs
-------
- prompt-ready planning hint
- structured retrieval JSON bundle
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss is required for PlanningRAGRetriever.") from e


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL line {line_no} in {path}: {e}") from e
    return rows


# ============================================================
# small utils
# ============================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def norm_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip().lower()


def coarse_anchor(anchor: Optional[str]) -> Optional[str]:
    if anchor is None:
        return None
    a = norm_text(anchor)
    if a is None:
        return None

    if "bed" in a:
        return "bed"
    if "sofa" in a or "couch" in a:
        return "sofa"
    if "dining table" in a:
        return "dining table"
    if "coffee table" in a:
        return "coffee table"
    if "side table" in a or "end table" in a:
        return "side table"
    if "nightstand" in a or "bedside" in a:
        return "nightstand"
    if "desk" in a:
        return "desk"
    if "tv stand" in a or "media console" in a:
        return "tv stand"
    if "bookshelf" in a or "bookcase" in a:
        return "bookshelf"
    if "cabinet" in a or "sideboard" in a:
        return "cabinet"
    if "wardrobe" in a:
        return "wardrobe"
    if "dresser" in a or "drawer chest" in a:
        return "dresser"
    if "washing machine" in a or "washer" in a:
        return "washing machine"
    return a


def doc_anchor_matches(doc_anchor: Optional[str], query_anchor: Optional[str], allow_coarse: bool = True) -> bool:
    if query_anchor is None:
        return True
    da = norm_text(doc_anchor)
    qa = norm_text(query_anchor)
    if da == qa:
        return True
    if allow_coarse and coarse_anchor(da) == coarse_anchor(qa):
        return True
    return False


def build_query_text(user_prompt: str, room_type: Optional[str], anchor: Optional[str]) -> str:
    parts = []
    if room_type:
        parts.append(room_type)
    if anchor:
        parts.append(anchor)
    parts.append(user_prompt)
    return " ".join(parts).strip()


# ============================================================
# dataclasses
# ============================================================

@dataclass
class RetrievalDoc:
    doc_id: str
    title: str
    scope: Optional[str]
    room_type: Optional[str]
    anchor: Optional[str]
    text: str
    score: float
    top_members: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    source_fine_anchors: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalDoc":
        return cls(
            doc_id=str(d.get("doc_id", "")),
            title=str(d.get("title", "")),
            scope=d.get("scope"),
            room_type=d.get("room_type"),
            anchor=d.get("anchor"),
            text=str(d.get("text", "")),
            score=float(d.get("score", 0.0)),
            top_members=d.get("top_members"),
            keywords=d.get("keywords"),
            source_fine_anchors=d.get("source_fine_anchors"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningRAGConfig:
    top_k_per_anchor: int = 2
    top_k_room_level: int = 3
    max_docs_per_anchor_in_prompt: int = 2
    allow_coarse_anchor_match: bool = True
    batch_size: int = 16
    device: Optional[str] = None


# ============================================================
# embedding backend
# ============================================================

class EmbeddingBackend:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, batch_size: int = 16):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.backend_type = None
        self.model = None
        self.tokenizer = None

        # sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name_or_path, device=device)
            self.backend_type = "sentence_transformers"
            return
        except Exception:
            pass

        # transformers fallback
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            self.model.to(self.device)
            self.model.eval()
            self.backend_type = "transformers"
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model from {model_name_or_path}") from e

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend_type == "sentence_transformers":
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embs.astype(np.float32)

        if self.backend_type == "transformers":
            return self._encode_transformers(texts)

        raise RuntimeError("Embedding backend is not initialized.")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])

    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        import torch

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError("Transformers model output does not contain last_hidden_state.")

            token_embeddings = outputs.last_hidden_state.float()
            attention_mask = batch["attention_mask"].unsqueeze(-1).float()

            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1.0)
            emb = summed / counts

            emb = emb.detach().cpu().numpy().astype(np.float32)
            emb = l2_normalize(emb)
            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)


# ============================================================
# candidate anchor inference
# ============================================================

ANCHOR_KEYWORDS = [
    "King-Size Bed",
    "Queen-Size Bed",
    "Double Bed",
    "Single Bed",
    "Kids Bed",
    "Bed",
    "Loveseat Sofa",
    "Sectional Sofa",
    "L-Shaped Sofa",
    "Sofa",
    "Dining Table",
    "Coffee Table",
    "Side Table",
    "Nightstand",
    "Desk",
    "TV Stand",
    "Bookshelf",
    "Cabinet",
    "Wardrobe",
    "Dresser",
    "Washing Machine",
]


def _extract_scene_text(obj: Dict[str, Any]) -> str:
    parts = []
    for k in ("category", "super_category", "desc", "prompt", "sampled_asset_desc"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " | ".join(parts).lower()


def infer_anchor_candidates(
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    user_prompt: str = "",
    max_anchors: int = 4,
) -> List[str]:
    """
    Heuristic anchor proposal.
    Priority:
    1) role_categories if provided
    2) scene object text
    3) user prompt text
    """
    votes: List[str] = []

    if role_categories:
        for cat in role_categories:
            if not isinstance(cat, str):
                continue
            votes.append(cat)

    if scene is not None:
        for obj in scene.get("objects", []):
            text = _extract_scene_text(obj)
            for kw in ANCHOR_KEYWORDS:
                if kw.lower() in text:
                    votes.append(kw)

    prompt_low = user_prompt.lower()
    for kw in ANCHOR_KEYWORDS:
        if kw.lower() in prompt_low:
            votes.append(kw)

    # dedupe but preserve a soft preference for more specific anchors first
    # sort by specificity (longer first), then frequency
    counter = {}
    for v in votes:
        counter[v] = counter.get(v, 0) + 1

    ranked = sorted(counter.items(), key=lambda kv: (-len(kv[0]), -kv[1], kv[0]))
    return [k for k, _ in ranked[:max_anchors]]


# ============================================================
# retriever
# ============================================================

class PlanningRAGRetriever:
    """
    Reusable retriever object.
    Load once, call many times.
    """

    def __init__(
        self,
        index_dir: str,
        model_name_or_path: str,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.index_dir = Path(index_dir)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size

        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.jsonl"
        config_path = self.index_dir / "config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = load_jsonl(metadata_path)
        self.config = load_json(config_path)
        self.embedder = EmbeddingBackend(
            model_name_or_path=model_name_or_path,
            device=device,
            batch_size=batch_size,
        )

    def _search(
        self,
        query_text: str,
        top_k: int,
        room_type: Optional[str] = None,
        anchor: Optional[str] = None,
        scope: Optional[str] = "room",
        allow_coarse_anchor_match: bool = True,
    ) -> List[RetrievalDoc]:
        query_vec = l2_normalize(self.embedder.encode_query(query_text).astype(np.float32))
        room_type_n = norm_text(room_type)
        scope_n = norm_text(scope)

        candidate_ids = []
        for i, doc in enumerate(self.metadata):
            doc_room = norm_text(doc.get("room_type"))
            doc_scope = norm_text(doc.get("scope"))
            doc_anchor = doc.get("anchor")

            if room_type_n is not None and doc_room != room_type_n:
                continue
            if scope_n is not None and doc_scope != scope_n:
                continue
            if not doc_anchor_matches(doc_anchor, anchor, allow_coarse=allow_coarse_anchor_match):
                continue
            candidate_ids.append(i)

        if not candidate_ids:
            return []

        vecs = np.asarray([self.index.reconstruct(int(i)) for i in candidate_ids], dtype=np.float32)
        scores = np.matmul(vecs, query_vec[0])
        order = np.argsort(-scores)

        results: List[RetrievalDoc] = []
        used_doc_ids = set()
        for rank in order:
            idx = candidate_ids[int(rank)]
            doc = dict(self.metadata[idx])
            doc["score"] = float(scores[int(rank)])
            doc_id = doc.get("doc_id", idx)
            if doc_id in used_doc_ids:
                continue
            used_doc_ids.add(doc_id)
            results.append(RetrievalDoc.from_dict(doc))
            if len(results) >= top_k:
                break

        return results

    def retrieve(
        self,
        room_type: Optional[str],
        user_prompt: str,
        anchors: Optional[List[str]] = None,
        top_k_per_anchor: int = 2,
        top_k_room_level: int = 3,
        allow_coarse_anchor_match: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns a structured retrieval bundle.
        """
        results_by_anchor: Dict[str, List[Dict[str, Any]]] = {}

        # room-level retrieval
        room_query = build_query_text(user_prompt=user_prompt, room_type=room_type, anchor=None)
        room_docs = self._search(
            query_text=room_query,
            top_k=top_k_room_level,
            room_type=room_type,
            anchor=None,
            scope="room",
            allow_coarse_anchor_match=allow_coarse_anchor_match,
        )
        if room_docs:
            results_by_anchor["__room_level__"] = [d.to_dict() for d in room_docs]

        # anchor-level retrieval
        anchors = anchors or []
        for anchor in anchors:
            q = build_query_text(user_prompt=user_prompt, room_type=room_type, anchor=anchor)
            docs = self._search(
                query_text=q,
                top_k=top_k_per_anchor,
                room_type=room_type,
                anchor=anchor,
                scope="room",
                allow_coarse_anchor_match=allow_coarse_anchor_match,
            )
            if docs:
                results_by_anchor[anchor] = [d.to_dict() for d in docs]

        return {
            "room_type": room_type,
            "user_prompt": user_prompt,
            "anchors": anchors,
            "results_by_anchor": results_by_anchor,
        }

    def summarize_for_prompt(
        self,
        retrieval_bundle: Dict[str, Any],
        max_docs_per_anchor: int = 2,
    ) -> str:
        room_type = retrieval_bundle.get("room_type")
        user_prompt = retrieval_bundle.get("user_prompt")
        results_by_anchor = retrieval_bundle.get("results_by_anchor", {})

        lines: List[str] = []
        lines.append("RETRIEVED GROUP PRIORS FOR PLANNING:")
        if room_type:
            lines.append(f"- room_type: {room_type}")
        lines.append(f"- original prompt: {user_prompt}")

        room_docs = results_by_anchor.get("__room_level__", [])
        if room_docs:
            lines.append("")
            lines.append("Room-level priors:")
            for i, doc in enumerate(room_docs[:max_docs_per_anchor], start=1):
                lines.append(f"{i}. [{doc.get('title', '')}] score={doc.get('score', 0.0):.4f}")
                lines.append(str(doc.get("text", "")).strip())

        for anchor, docs in results_by_anchor.items():
            if anchor == "__room_level__":
                continue
            if not docs:
                continue
            lines.append("")
            lines.append(f"Anchor priors for {anchor}:")
            for i, doc in enumerate(docs[:max_docs_per_anchor], start=1):
                lines.append(f"{i}. [{doc.get('title', '')}] score={doc.get('score', 0.0):.4f}")
                lines.append(str(doc.get("text", "")).strip())

        lines.append("")
        lines.append(
            "Use these priors only as soft planning hints. "
            "They should complement the user instruction, not override it. "
            "Prioritize requested objects and room semantics first, then use retrieved priors "
            "to decide common groupings, likely companion objects, and reasonable initial relative placements."
        )
        return "\n".join(lines)

    def build_augmented_prompt(
        self,
        base_prompt: str,
        retrieval_bundle: Dict[str, Any],
        max_docs_per_anchor: int = 2,
    ) -> str:
        prior_hint = self.summarize_for_prompt(retrieval_bundle, max_docs_per_anchor=max_docs_per_anchor)
        return f"""{prior_hint}

Now perform scene planning for the following request:
{base_prompt}
"""


# ============================================================
# high-level helpers
# ============================================================

def build_planning_rag_hint(
    retriever: PlanningRAGRetriever,
    *,
    user_prompt: str,
    room_type: Optional[str],
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    anchors: Optional[List[str]] = None,
    top_k_per_anchor: int = 2,
    top_k_room_level: int = 3,
    max_docs_per_anchor_in_prompt: int = 2,
    allow_coarse_anchor_match: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    One-shot helper:
    - infer anchors if not provided
    - retrieve docs
    - summarize into a prompt-ready hint
    """
    if anchors is None or len(anchors) == 0:
        anchors = infer_anchor_candidates(
            scene=scene,
            role_categories=role_categories,
            user_prompt=user_prompt,
            max_anchors=4,
        )

    bundle = retriever.retrieve(
        room_type=room_type,
        user_prompt=user_prompt,
        anchors=anchors,
        top_k_per_anchor=top_k_per_anchor,
        top_k_room_level=top_k_room_level,
        allow_coarse_anchor_match=allow_coarse_anchor_match,
    )
    hint = retriever.summarize_for_prompt(bundle, max_docs_per_anchor=max_docs_per_anchor_in_prompt)
    return hint, bundle


def build_augmented_extra_hints(
    retriever: PlanningRAGRetriever,
    *,
    base_extra_hints_text: str,
    user_prompt: str,
    room_type: Optional[str],
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    anchors: Optional[List[str]] = None,
    top_k_per_anchor: int = 2,
    top_k_room_level: int = 3,
    max_docs_per_anchor_in_prompt: int = 2,
    allow_coarse_anchor_match: bool = True,
    separator: str = "\n\n",
) -> str:
    """
    Appends RAG hint to your existing extra_hints_text.
    This is the easiest way to integrate with the user's current architecture.
    """
    hint, _ = build_planning_rag_hint(
        retriever=retriever,
        user_prompt=user_prompt,
        room_type=room_type,
        scene=scene,
        role_categories=role_categories,
        anchors=anchors,
        top_k_per_anchor=top_k_per_anchor,
        top_k_room_level=top_k_room_level,
        max_docs_per_anchor_in_prompt=max_docs_per_anchor_in_prompt,
        allow_coarse_anchor_match=allow_coarse_anchor_match,
    )

    base = (base_extra_hints_text or "").strip()
    if not base:
        return hint
    return f"{base}{separator}{hint}"


def dump_retrieval_bundle(path: str | Path, retrieval_bundle: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(retrieval_bundle, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# optional self-test
# ============================================================

def _demo() -> None:
    print("This module is intended to be imported, not run as a standalone CLI.")


if __name__ == "__main__":
    _demo()
