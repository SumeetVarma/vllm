# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree: Diffusion Draft Tree for speculative decoding.

Implements the tree construction and traversal algorithms from:
  "Accelerating Speculative Decoding with Block Diffusion Draft Trees"
  Ringel & Romano, arXiv:2604.12989

DDTree builds a draft tree from DFlash's per-position probability
distributions using a best-first heap, then verifies the whole tree
in a single target-model forward pass using ancestor-only attention.
"""

import heapq
import os
import pathlib
import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import torch
import triton
import triton.language as tl
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


@contextmanager
def _ddtree_profile_range(name: str, **fields):
    if os.environ.get("DTREE_PROFILE") != "1":
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("DTREE_PROFILE ddtree_%s %.3fms %s", name, elapsed_ms, field_text)


@contextmanager
def _ddtree_cpu_profile_range(name: str, **fields):
    """CPU wall timer for async-boundary profiling.

    Unlike DTREE_PROFILE, this intentionally does not synchronize CUDA. If a
    .cpu(), .numpy(), or .tolist() boundary blocks on queued GPU work, the wait
    shows up here without globally changing the stream schedule.
    """
    if os.environ.get("DTREE_CPU_PROFILE") != "1":
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("DTREE_CPU_PROFILE ddtree_%s %.3fms %s", name, elapsed_ms, field_text)


_SGL_DTREE_OPS_LOADED = False
_STATIC_SIBLING_TOPOLOGY_CACHE: dict[
    tuple[int, ...],
    tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor],
] = {}
_RETRIEVE_TABLE_CACHE: dict[
    tuple[str, int, int, tuple[tuple[int, ...], ...]],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


def _load_sgl_dtree_ops() -> bool:
    """Load SGLang's speculative tree CUDA ops if the local build exists."""
    if os.environ.get("DDTREE_DISABLE_SGL_OPS") == "1":
        return False
    global _SGL_DTREE_OPS_LOADED
    if _SGL_DTREE_OPS_LOADED:
        return True
    explicit = os.environ.get("DDTREE_SGL_OPS_SO")
    candidates: list[pathlib.Path]
    if explicit:
        candidates = [pathlib.Path(explicit)]
    else:
        candidates = sorted(
            pathlib.Path("/root/.cache/torch_extensions").glob(
                "**/sgl_dtree_ops.so"
            )
        )
    for so_path in reversed(candidates):
        if not so_path.exists():
            continue
        try:
            torch.ops.load_library(str(so_path))
            _SGL_DTREE_OPS_LOADED = True
            logger.info("Loaded SGLang DDTree ops from %s", so_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load SGLang DDTree ops from %s: %s", so_path, exc)
    return False


@triton.jit
def _ddtree_verify_greedy_kernel(
    target_predict,
    draft_token_ids,
    retrieve_next_token,
    retrieve_next_sibling,
    output,
    gdn_state_indices,
    budget: tl.constexpr,
    out_stride: tl.constexpr,
):
    req = tl.program_id(0)

    offs = tl.arange(0, 128)
    if budget + 1 <= 128:
        tl.store(output + req * out_stride + offs, -1, mask=offs < budget + 1)

    current = tl.full((), 0, dtype=tl.int32)
    out_pos = tl.full((), 0, dtype=tl.int32)
    done = tl.full((), False, dtype=tl.int1)

    for _ in range(0, budget + 1):
        next_token = tl.load(target_predict + req * (budget + 1) + current)
        child = tl.load(retrieve_next_token + req * (budget + 1) + current)
        matched = tl.full((), -1, dtype=tl.int32)

        for _sibling in range(0, budget):
            has_child = (child >= 0) & (~done) & (matched < 0)
            child_token = tl.load(
                draft_token_ids + req * budget + child - 1,
                mask=has_child,
                other=-2147483648,
            )
            is_match = has_child & (child_token == next_token)
            matched = tl.where(is_match & (matched < 0), child, matched)
            child = tl.load(
                retrieve_next_sibling + req * (budget + 1) + child,
                mask=has_child & (matched < 0),
                other=-1,
            )

        no_match = (matched < 0) & (~done)
        if no_match:
            tl.store(output + req * out_stride + out_pos, next_token)
            tl.store(gdn_state_indices + req, current + 1)
            done = True

        has_match = (matched >= 0) & (~done)
        if has_match:
            tl.store(output + req * out_stride + out_pos, next_token)
            out_pos += 1
            current = matched


def build_retrieve_from_child_maps(
    child_maps: list[list[dict[int, int]]],
    budget: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SGLang-style linked-child retrieval tables for DDTree verify."""
    batch_size = len(child_maps)
    retrieve_next_token = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    retrieve_next_sibling = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    for r, maps in enumerate(child_maps):
        for parent_idx, children in enumerate(maps[: budget + 1]):
            previous_child = -1
            for child_idx in children.values():
                child_idx = int(child_idx)
                if previous_child < 0:
                    retrieve_next_token[r, parent_idx] = child_idx
                else:
                    retrieve_next_sibling[r, previous_child] = child_idx
                previous_child = child_idx
    return retrieve_next_token, retrieve_next_sibling


def get_cached_retrieve_tables(
    child_maps: list[list[dict[int, int]]],
    budget: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return cached child/sibling tables for repeated static DDTree topology."""
    key = (
        str(device),
        budget,
        len(child_maps),
        tuple(
            tuple(tuple(children.values()) for children in maps[: budget + 1])
            for maps in child_maps
        ),
    )
    cached = _RETRIEVE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    retrieve_next_token_i32, retrieve_next_sibling_i32 = build_retrieve_from_child_maps(
        child_maps, budget, device
    )
    retrieve_next_token_i64 = retrieve_next_token_i32.to(dtype=torch.int64)
    retrieve_next_sibling_i64 = retrieve_next_sibling_i32.to(dtype=torch.int64)
    retrieve_index = torch.arange(
        len(child_maps) * (budget + 1), dtype=torch.int64, device=device
    ).view(len(child_maps), budget + 1)
    row_offsets = (
        torch.arange(len(child_maps), dtype=torch.int32, device=device) * (budget + 1)
    ).view(len(child_maps), 1)
    cached = (
        retrieve_next_token_i32,
        retrieve_next_sibling_i32,
        retrieve_next_token_i64,
        retrieve_next_sibling_i64,
        retrieve_index,
        row_offsets,
    )
    _RETRIEVE_TABLE_CACHE[key] = cached
    return cached


def _build_visibility_from_parents(parents: np.ndarray) -> torch.Tensor:
    # visibility[i, j] == True iff j is an ancestor of i or j == i.
    current_length = int(len(parents))
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for idx in range(1, current_length):
        p_idx = int(parents[idx])
        visibility_np[idx, :idx] = visibility_np[p_idx, :idx]
        visibility_np[idx, idx] = True
    return torch.from_numpy(visibility_np)


def _build_static_sibling_child_maps(
    chain_len: int,
    branch_count: int,
) -> list[dict[int, int]]:
    """Index-only child maps for the static chain+siblings topology.

    The GPU verifier uses only child indices from these maps; token matching is
    done against the GPU draft-token tensor.  Dummy keys deliberately avoid a
    proposal-time GPU->CPU token sync in the static fast path.
    """
    total_nodes = chain_len + branch_count
    child_maps: list[dict[int, int]] = [{} for _ in range(total_nodes + 1)]
    parent_index = 0
    for depth in range(1, chain_len + 1):
        current_index = depth
        child_maps[parent_index][-current_index] = current_index
        parent_index = current_index
    for depth in range(1, branch_count + 1):
        current_index = chain_len + depth
        parent_index = depth - 1
        child_maps[parent_index][-current_index] = current_index
    return child_maps


def _static_interleaved_chain_index(depth: int, branch_count: int) -> int:
    return depth + min(depth - 1, branch_count)


def _build_static_interleaved_sibling_child_maps(
    chain_len: int,
    branch_count: int,
) -> list[dict[int, int]]:
    total_nodes = chain_len + branch_count
    child_maps: list[dict[int, int]] = [{} for _ in range(total_nodes + 1)]
    for depth in range(1, chain_len + 1):
        chain_index = _static_interleaved_chain_index(depth, branch_count)
        parent_index = (
            0
            if depth == 1
            else _static_interleaved_chain_index(depth - 1, branch_count)
        )
        child_maps[parent_index][-chain_index] = chain_index
        if depth <= branch_count:
            sibling_index = chain_index + 1
            child_maps[parent_index][-sibling_index] = sibling_index
    return child_maps


def _build_static_sibling_topology(
    chain_len: int,
    branch_count: int,
    interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    total_nodes = chain_len + branch_count
    topology_key = (chain_len, branch_count, total_nodes, int(interleave))
    cached = _STATIC_SIBLING_TOPOLOGY_CACHE.get(topology_key)
    if cached is not None:
        return cached

    node_depths_np = np.empty(total_nodes, dtype=np.int64)
    node_ranks_np = np.empty(total_nodes, dtype=np.int64)
    parents_np = np.empty(total_nodes + 1, dtype=np.int32)
    parents_np[0] = -1
    if interleave:
        for depth in range(1, chain_len + 1):
            chain_index = _static_interleaved_chain_index(depth, branch_count)
            parent_index = (
                0
                if depth == 1
                else _static_interleaved_chain_index(depth - 1, branch_count)
            )
            node_depths_np[chain_index - 1] = depth
            node_ranks_np[chain_index - 1] = 0
            parents_np[chain_index] = parent_index
            if depth <= branch_count:
                sibling_index = chain_index + 1
                node_depths_np[sibling_index - 1] = depth
                node_ranks_np[sibling_index - 1] = 1
                parents_np[sibling_index] = parent_index
    else:
        node_depths_np[:] = np.concatenate(
            [
                np.arange(1, chain_len + 1, dtype=np.int64),
                np.arange(1, branch_count + 1, dtype=np.int64),
            ]
        )
        node_ranks_np[:] = np.concatenate(
            [
                np.zeros(chain_len, dtype=np.int64),
                np.ones(branch_count, dtype=np.int64),
            ]
        )
        for depth in range(1, chain_len + 1):
            parents_np[depth] = depth - 1
        for depth in range(1, branch_count + 1):
            parents_np[chain_len + depth] = depth - 1

    cached = (
        torch.from_numpy(node_depths_np),
        torch.from_numpy(node_ranks_np),
        parents_np.tolist(),
        _build_visibility_from_parents(parents_np),
    )
    _STATIC_SIBLING_TOPOLOGY_CACHE[topology_key] = cached
    return cached


def build_static_sibling_tree_from_token_ids(
    top1_ids: torch.Tensor,
    branch_top2_ids: torch.Tensor,
    budget: int,
    chain_len: int,
    branch_count: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    """Build the static chain+siblings tree from already-selected tokens."""
    chain_len = max(0, min(int(chain_len), int(budget), int(top1_ids.shape[0])))
    branch_count = max(0, min(int(branch_count), int(budget) - chain_len, chain_len))
    total_nodes = chain_len + branch_count
    if total_nodes <= 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long, device=top1_ids.device),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [{}],
            visibility,
        )

    interleave = os.environ.get("DDTREE_STATIC_INTERLEAVE") == "1"
    if interleave:
        pieces = []
        for depth in range(chain_len):
            pieces.append(top1_ids[depth : depth + 1])
            if depth < branch_count:
                pieces.append(branch_top2_ids[depth : depth + 1])
        node_token_ids = torch.cat(pieces, dim=0).to(dtype=torch.long)
    else:
        node_token_ids = torch.cat(
            [top1_ids[:chain_len], branch_top2_ids[:branch_count]],
            dim=0,
        ).to(dtype=torch.long)

    node_depths, node_ranks, parents_list, visibility = _build_static_sibling_topology(
        chain_len, branch_count, interleave
    )
    return (
        node_token_ids,
        node_depths,
        node_ranks,
        parents_list,
        (
            _build_static_interleaved_sibling_child_maps(chain_len, branch_count)
            if interleave
            else _build_static_sibling_child_maps(chain_len, branch_count)
        ),
        visibility,
    )


def resolve_ddtree_candidate_topk(
    budget: int,
    vocab_size: int,
    force_chain_prefix: int = 0,
) -> int:
    """Resolve how many draft candidates to materialize per depth.

    Explicit DDTREE_CANDIDATE_TOPK keeps its old override semantics.  For the
    latency-critical DDTree-15 shape used by the competition stack, the tree
    reserves a long top-1 chain and spends only a few nodes on branches.  A
    full top-15 slice is unnecessary proposer work and empirically slower than
    top-12 while producing the same exact verifier semantics.
    """
    candidate_topk = int(os.environ.get("DDTREE_CANDIDATE_TOPK", "0") or "0")
    if candidate_topk <= 0 and budget == 15 and force_chain_prefix >= 12:
        candidate_topk = 12
    if candidate_topk <= 0:
        candidate_topk = budget
    return min(candidate_topk, budget, vocab_size)


def should_use_ddtree_for_prompt_lens(prompt_lens) -> bool:
    """Return whether DDTree should run for this request batch.

    The current DDTree path wins on short prompts but loses on medium/long
    prompts on the A10 stack.  Keep the threshold environment-controlled so the
    gate is easy to disable or retune without changing code.
    """
    threshold = int(os.environ.get("DDTREE_MAX_PROMPT_TOKENS", "0") or "0")
    if threshold <= 0:
        return True
    return all(int(length) <= threshold for length in prompt_lens)


def _build_forced_chain_tail_tree_fast(
    top_token_ids_np: np.ndarray,
    top_log_probs_np: np.ndarray,
    budget: int,
    depth_limit: int,
    force_chain_prefix: int,
    topk: int,
):
    """Specialized builder for the competition's chain+small-tail shape."""
    if (
        os.environ.get("DDTREE_ENABLE_FAST_CHAIN") != "1"
        or budget != 15
        or force_chain_prefix < 12
        or topk <= 1
        or depth_limit < 12
    ):
        return None

    chain_len = max(0, min(int(force_chain_prefix), budget, depth_limit))
    if budget - chain_len != 3:
        return None

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_ranks_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    path_logw_by_index = [0.0]
    path_ranks_by_index: list[tuple[int, ...]] = [()]
    added_keys: set[tuple[int, int, int]] = set()
    node_count = 0

    def add_node(
        parent_index: int,
        depth: int,
        rank: int,
        logw: float,
        ranks: tuple[int, ...],
    ) -> int:
        nonlocal node_count
        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        node_ranks_np[node_count] = rank
        parents_np[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        path_logw_by_index.append(logw)
        path_ranks_by_index.append(ranks)
        added_keys.add((parent_index, depth, rank))
        node_count += 1
        return current_index

    parent_index = 0
    logw = 0.0
    ranks: tuple[int, ...] = ()
    forced_node_indices: list[int] = []
    for depth in range(1, chain_len + 1):
        logw += float(top_log_probs_np[depth - 1, 0])
        ranks = ranks + (0,)
        parent_index = add_node(parent_index, depth, 0, logw, ranks)
        forced_node_indices.append(parent_index)

    candidates: list[tuple[float, tuple[int, ...], int, int, int, float]] = []

    def push_candidate(
        parent: int,
        depth: int,
        rank: int,
        cand_logw: float,
        cand_ranks: tuple[int, ...],
    ) -> None:
        if depth > depth_limit or rank >= topk:
            return
        if (parent, depth, rank) in added_keys:
            return
        candidates.append((-cand_logw, cand_ranks, parent, depth, rank, cand_logw))

    for node_index in forced_node_indices:
        depth = int(node_depths_np[node_index - 1])
        parent = int(parents_np[node_index])
        parent_logw = path_logw_by_index[parent]
        parent_ranks = path_ranks_by_index[parent]
        push_candidate(
            parent,
            depth,
            1,
            parent_logw + float(top_log_probs_np[depth - 1, 1]),
            parent_ranks + (1,),
        )
    if chain_len < depth_limit:
        push_candidate(
            parent_index,
            chain_len + 1,
            0,
            logw + float(top_log_probs_np[chain_len, 0]),
            ranks + (0,),
        )

    while candidates and node_count < budget:
        best = min(range(len(candidates)), key=candidates.__getitem__)
        _, ranks, parent_index, depth, rank, logw = candidates.pop(best)
        token_id = int(top_token_ids_np[depth - 1, rank])
        if (parent_index, depth, rank) in added_keys:
            continue
        if token_id in child_maps[parent_index]:
            continue
        current_index = add_node(parent_index, depth, rank, logw, ranks)

        if rank + 1 < topk:
            parent_logw = path_logw_by_index[parent_index]
            push_candidate(
                parent_index,
                depth,
                rank + 1,
                parent_logw + float(top_log_probs_np[depth - 1, rank + 1]),
                path_ranks_by_index[parent_index] + (rank + 1,),
            )
        if depth < depth_limit:
            push_candidate(
                current_index,
                depth + 1,
                0,
                logw + float(top_log_probs_np[depth, 0]),
                ranks + (0,),
            )

    current_length = 1 + node_count
    parents_view = parents_np[:current_length]
    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        torch.from_numpy(node_ranks_np[:node_count]),
        parents_view.tolist(),
        child_maps,
        _build_visibility_from_parents(parents_view),
    )


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    force_chain_prefix: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    """Build a draft tree from DFlash per-position logits.

    Uses a best-first heap to select the ``budget`` most-probable token
    paths according to the draft model's output distributions.

    Args:
        draft_logits: Float tensor of shape ``[depth, vocab_size]``.
            Raw (un-softmaxed) logits for each speculative position,
            as produced by the DFlash draft model.
        budget: Maximum number of non-root tree nodes to expand.
        force_chain_prefix: Number of leading top-1 chain nodes to reserve
            before spending the remaining budget on best-first branches.

    Returns:
        node_token_ids: int64[num_nodes] — token id at each non-root node.
        node_depths:    int64[num_nodes] — 1-based depth (root=0, children=1, …).
        node_ranks:     int64[num_nodes] — top-k rank at each node's depth position.
        parents:    list[num_nodes+1] — parent index per node; parents[0]==-1 (root).
        child_maps: list[num_nodes+1] of dicts —
            ``child_maps[i][token_id]`` = child index.
        visibility:     bool[num_nodes+1, num_nodes+1] — ancestor-only attention mask.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [{}],
            visibility,
        )

    depth_limit = int(draft_logits.shape[0])
    static_siblings = int(os.environ.get("DDTREE_STATIC_SIBLINGS", "0") or "0")
    if static_siblings > 0:
        chain_len = max(0, min(int(force_chain_prefix), budget, depth_limit))
        branch_count = max(0, min(static_siblings, budget - chain_len, chain_len))
        total_nodes = chain_len + branch_count
        interleave = os.environ.get("DDTREE_STATIC_INTERLEAVE") == "1"
        topology_key = (chain_len, branch_count, total_nodes, int(interleave))
        gpu_static_tree = (
            os.environ.get("DDTREE_GPU_STATIC_TREE") == "1"
            and os.environ.get("DDTREE_SGL_VERIFY") == "1"
            and total_nodes == budget
            and branch_count > 0
        )
        with _ddtree_cpu_profile_range("static_top1_argmax", rows=depth_limit):
            top1_ids = draft_logits.argmax(dim=-1)
        if branch_count > 0 and draft_logits.shape[-1] > 1:
            with _ddtree_cpu_profile_range("static_branch_top2", rows=branch_count):
                branch_top2_ids = torch.topk(
                    draft_logits[:branch_count].float(), k=2, dim=-1
                ).indices
        else:
            branch_top2_ids = None
        if gpu_static_tree and branch_top2_ids is not None:
            if interleave:
                pieces = []
                for depth in range(chain_len):
                    pieces.append(top1_ids[depth : depth + 1])
                    if depth < branch_count:
                        pieces.append(branch_top2_ids[depth : depth + 1, 1])
                node_token_ids = torch.cat(pieces, dim=0).to(dtype=torch.long)
            else:
                node_token_ids = torch.cat(
                    [top1_ids[:chain_len], branch_top2_ids[:branch_count, 1]], dim=0
                ).to(dtype=torch.long)
            node_depths, node_ranks, parents_list, visibility = (
                _build_static_sibling_topology(chain_len, branch_count, interleave)
            )
            return (
                node_token_ids,
                node_depths,
                node_ranks,
                parents_list,
                (
                    _build_static_interleaved_sibling_child_maps(
                        chain_len, branch_count
                    )
                    if interleave
                    else _build_static_sibling_child_maps(chain_len, branch_count)
                ),
                visibility,
            )
        with _ddtree_cpu_profile_range("static_top1_d2h", rows=depth_limit):
            top1_ids_np = top1_ids.to(device="cpu", dtype=torch.long).numpy()
        with _ddtree_cpu_profile_range("static_branch_d2h", rows=branch_count):
            branch_top2_ids_np = (
                branch_top2_ids.to(device="cpu", dtype=torch.long).numpy()
                if branch_top2_ids is not None
                else None
            )
        node_token_ids_np = np.empty(total_nodes, dtype=np.int64)
        node_depths_np = np.empty(total_nodes, dtype=np.int64)
        node_ranks_np = np.empty(total_nodes, dtype=np.int64)
        parents_np = np.empty(total_nodes + 1, dtype=np.int32)
        parents_np[0] = -1
        child_maps: list[dict[int, int]] = [{}]

        parent_index = 0
        with _ddtree_cpu_profile_range("static_python_tree", rows=total_nodes):
            for depth in range(1, chain_len + 1):
                current_index = depth
                token_id = int(top1_ids_np[depth - 1])
                node_token_ids_np[current_index - 1] = token_id
                node_depths_np[current_index - 1] = depth
                node_ranks_np[current_index - 1] = 0
                parents_np[current_index] = parent_index
                child_maps.append({})
                child_maps[parent_index][token_id] = current_index
                parent_index = current_index

            node_count = chain_len
            for depth in range(1, branch_count + 1):
                if branch_top2_ids_np is None:
                    break
                parent_index = depth - 1
                token_id = int(branch_top2_ids_np[depth - 1, 1])
                if token_id in child_maps[parent_index]:
                    continue
                node_count += 1
                current_index = node_count
                node_token_ids_np[current_index - 1] = token_id
                node_depths_np[current_index - 1] = depth
                node_ranks_np[current_index - 1] = 1
                parents_np[current_index] = parent_index
                child_maps.append({})
                child_maps[parent_index][token_id] = current_index

        if node_count == total_nodes:
            cached = _STATIC_SIBLING_TOPOLOGY_CACHE.get(topology_key)
            if cached is None:
                node_depths = torch.from_numpy(node_depths_np[:node_count].copy())
                node_ranks = torch.from_numpy(node_ranks_np[:node_count].copy())
                parents_list = parents_np[: node_count + 1].tolist()
                visibility = _build_visibility_from_parents(
                    parents_np[: node_count + 1]
                )
                cached = (node_depths, node_ranks, parents_list, visibility)
                _STATIC_SIBLING_TOPOLOGY_CACHE[topology_key] = cached
            node_depths, node_ranks, parents_list, visibility = cached
        else:
            parents_view = parents_np[: node_count + 1]
            node_depths = torch.from_numpy(node_depths_np[:node_count])
            node_ranks = torch.from_numpy(node_ranks_np[:node_count])
            parents_list = parents_view.tolist()
            visibility = _build_visibility_from_parents(parents_view)
        return (
            torch.from_numpy(node_token_ids_np[:node_count]),
            node_depths,
            node_ranks,
            parents_list,
            child_maps,
            visibility,
        )

    topk = resolve_ddtree_candidate_topk(
        budget=budget,
        vocab_size=draft_logits.shape[-1],
        force_chain_prefix=force_chain_prefix,
    )

    # Compute normalised log-probabilities for the top-k tokens at each
    # position.  Move to CPU immediately; the heap runs on the CPU.  For the
    # top-k-logZ fast path, avoid upcasting the full vocab matrix to fp32.
    if os.environ.get("DDTREE_TOPK_LOGZ") == "1" or os.environ.get("DDTREE_SKIP_LOGZ") == "1":
        with _ddtree_cpu_profile_range("generic_topk", rows=depth_limit, topk=topk):
            top_logits, top_token_ids = torch.topk(draft_logits, k=topk, dim=-1)
        top_logits = top_logits.float()
        if os.environ.get("DDTREE_SKIP_LOGZ") == "1":
            with _ddtree_cpu_profile_range("generic_logprob_d2h", rows=depth_limit, topk=topk):
                top_log_probs_np = top_logits.to(device="cpu", dtype=torch.float32).numpy()
        else:
            # Fast proposal-only approximation: use the already materialized top-k
            # slice to normalize branch scores instead of scanning the full vocab.
            # Verification remains exact; this only affects which candidate nodes
            # are selected for the fixed tree budget.
            with _ddtree_cpu_profile_range("generic_topk_logz", rows=depth_limit, topk=topk):
                log_z = torch.logsumexp(top_logits, dim=-1, keepdim=True)
                top_log_probs = top_logits - log_z
            with _ddtree_cpu_profile_range("generic_logprob_d2h", rows=depth_limit, topk=topk):
                top_log_probs_np = top_log_probs.to(device="cpu", dtype=torch.float32).numpy()
    else:
        with _ddtree_cpu_profile_range("generic_float_full", rows=depth_limit):
            logits = draft_logits.float()
        with _ddtree_cpu_profile_range("generic_topk", rows=depth_limit, topk=topk):
            top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
        with _ddtree_cpu_profile_range("generic_full_logz", rows=depth_limit):
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            top_log_probs = top_logits - log_z
        with _ddtree_cpu_profile_range("generic_logprob_d2h", rows=depth_limit, topk=topk):
            top_log_probs_np = top_log_probs.to(device="cpu", dtype=torch.float32).numpy()
    with _ddtree_cpu_profile_range("generic_token_d2h", rows=depth_limit, topk=topk):
        top_token_ids_np = top_token_ids.to(device="cpu", dtype=torch.long).numpy()

    fast_tree = _build_forced_chain_tail_tree_fast(
        top_token_ids_np=top_token_ids_np,
        top_log_probs_np=top_log_probs_np,
        budget=budget,
        depth_limit=depth_limit,
        force_chain_prefix=force_chain_prefix,
        topk=topk,
    )
    if fast_tree is not None:
        return fast_tree

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    node_ranks_np = np.empty(budget, dtype=np.int64)
    # parents_np[0] == -1 (root); parents_np[i] for i >= 1 is the
    # parent node index.
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    path_logw_by_index = [0.0]
    path_ranks_by_index: list[tuple[int, ...]] = [()]
    added_keys: set[tuple[int, int, int]] = set()
    node_count = 0

    # Best-first heap. Each entry is a candidate node not yet added to the tree.
    # Entry: (-logw, ranks, parent_index, depth, rank, logw)
    #
    #   -logw        — negated accumulated path log-prob; min-heap pops the
    #                  highest log-prob candidate first
    #   ranks        — tuple of top-k indices taken at each depth along this
    #                  path, e.g. (0, 1) = rank-0 at depth 1, rank-1 at depth 2;
    #                  used only as a tiebreaker when two entries have equal logw
    #   parent_index — insertion-order index of this candidate's parent node
    #                  (root=0, first inserted node=1, second=2, ...)
    #   depth        — 1-based depth of this candidate (root is depth 0)
    #   rank         — index k into top_token_ids_np[depth-1, k] for this token;
    #                  rank 0 = most probable token at this depth position
    #   logw         — accumulated path log-prob (non-negated); kept separately
    #                  so sibling/child expansions can do arithmetic on the raw
    #                  sum without re-extracting it from the negated first field
    heap: list[tuple] = []

    def add_node(
        parent_index: int,
        depth: int,
        rank: int,
        logw: float,
        ranks: tuple[int, ...],
    ) -> int:
        nonlocal node_count
        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1

        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        node_ranks_np[node_count] = rank
        parents_np[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        path_logw_by_index.append(logw)
        path_ranks_by_index.append(ranks)
        added_keys.add((parent_index, depth, rank))
        node_count += 1
        return current_index

    def push_candidate(
        parent_index: int,
        depth: int,
        rank: int,
        logw: float,
        ranks: tuple[int, ...],
    ) -> None:
        if depth > depth_limit or rank >= topk:
            return
        if (parent_index, depth, rank) in added_keys:
            return
        heapq.heappush(heap, (-logw, ranks, parent_index, depth, rank, logw))

    force_chain_len = max(0, min(int(force_chain_prefix), budget, depth_limit))
    parent_index = 0
    logw = 0.0
    ranks: tuple[int, ...] = ()
    forced_node_indices: list[int] = []
    for depth in range(1, force_chain_len + 1):
        logw += float(top_log_probs_np[depth - 1, 0])
        ranks = ranks + (0,)
        parent_index = add_node(parent_index, depth, 0, logw, ranks)
        forced_node_indices.append(parent_index)

    root_siblings = int(os.environ.get("DDTREE_ROOT_SIBLINGS", "0") or "0")
    if (
        root_siblings > 0
        and force_chain_len > 0
        and depth_limit > 0
        and node_count < budget
    ):
        max_rank = min(topk, root_siblings + 1)
        for rank in range(1, max_rank):
            if node_count >= budget:
                break
            token_id = int(top_token_ids_np[0, rank])
            if token_id in child_maps[0]:
                continue
            add_node(
                0,
                1,
                rank,
                float(top_log_probs_np[0, rank]),
                (rank,),
            )
        return (
            torch.from_numpy(node_token_ids_np[:node_count]),
            torch.from_numpy(node_depths_np[:node_count]),
            torch.from_numpy(node_ranks_np[:node_count]),
            parents_np[: node_count + 1].tolist(),
            child_maps,
            _build_visibility_from_parents(parents_np[: node_count + 1]),
        )

    suffix_branch_depths_env = os.environ.get("DDTREE_SUFFIX_BRANCH_DEPTHS", "")
    if (
        suffix_branch_depths_env
        and force_chain_len > 0
        and node_count < budget
        and topk > 1
    ):
        prefix_logw: list[float] = [0.0]
        running_logw = 0.0
        for depth in range(1, force_chain_len + 1):
            running_logw += float(top_log_probs_np[depth - 1, 0])
            prefix_logw.append(running_logw)

        branch_depths: list[int] = []
        for item in suffix_branch_depths_env.split(","):
            item = item.strip()
            if not item:
                continue
            depth = int(item)
            if 1 <= depth <= force_chain_len:
                branch_depths.append(depth)

        for branch_depth in branch_depths:
            if node_count >= budget:
                break
            parent_index = 0 if branch_depth == 1 else forced_node_indices[branch_depth - 2]
            branch_token_id = int(top_token_ids_np[branch_depth - 1, 1])
            if branch_token_id in child_maps[parent_index]:
                continue

            branch_logw = (
                prefix_logw[branch_depth - 1]
                + float(top_log_probs_np[branch_depth - 1, 1])
            )
            branch_ranks = (0,) * (branch_depth - 1) + (1,)
            branch_parent = add_node(
                parent_index,
                branch_depth,
                1,
                branch_logw,
                branch_ranks,
            )
            for depth in range(branch_depth + 1, force_chain_len + 1):
                if node_count >= budget:
                    break
                branch_logw += float(top_log_probs_np[depth - 1, 0])
                branch_ranks = branch_ranks + (0,)
                branch_parent = add_node(
                    branch_parent,
                    depth,
                    0,
                    branch_logw,
                    branch_ranks,
                )
        return (
            torch.from_numpy(node_token_ids_np[:node_count]),
            torch.from_numpy(node_depths_np[:node_count]),
            torch.from_numpy(node_ranks_np[:node_count]),
            parents_np[: node_count + 1].tolist(),
            child_maps,
            _build_visibility_from_parents(parents_np[: node_count + 1]),
        )

    alt_root_chains = int(os.environ.get("DDTREE_ALT_ROOT_CHAINS", "0") or "0")
    if os.environ.get("DDTREE_ALT_ROOT_CHAIN") == "1":
        alt_root_chains = max(alt_root_chains, 1)
    if (
        alt_root_chains > 0
        and node_count < budget
        and depth_limit > 0
        and topk > 1
    ):
        max_root_rank = min(topk, alt_root_chains + 1)
        for root_rank in range(1, max_root_rank):
            if node_count >= budget:
                break
            token_id = int(top_token_ids_np[0, root_rank])
            if token_id in child_maps[0]:
                continue
            alt_parent = 0
            alt_logw = float(top_log_probs_np[0, root_rank])
            alt_ranks: tuple[int, ...] = (root_rank,)
            alt_parent = add_node(alt_parent, 1, root_rank, alt_logw, alt_ranks)
            for depth in range(2, depth_limit + 1):
                if node_count >= budget:
                    break
                alt_logw += float(top_log_probs_np[depth - 1, 0])
                alt_ranks = alt_ranks + (0,)
                alt_parent = add_node(alt_parent, depth, 0, alt_logw, alt_ranks)
        heap.clear()
        return (
            torch.from_numpy(node_token_ids_np[:node_count]),
            torch.from_numpy(node_depths_np[:node_count]),
            torch.from_numpy(node_ranks_np[:node_count]),
            parents_np[: node_count + 1].tolist(),
            child_maps,
            _build_visibility_from_parents(parents_np[: node_count + 1]),
        )

    if force_chain_len == 0:
        first_logw = float(top_log_probs_np[0, 0])
        push_candidate(0, 1, 0, first_logw, (0,))
    else:
        # Seed branches off every forced-chain parent, then let the same
        # best-first expansion fill the remaining budget. This preserves the
        # flat DFlash top-1 chain floor while still allocating extra nodes to
        # high-probability alternatives.
        for node_index in forced_node_indices:
            depth = int(node_depths_np[node_index - 1])
            parent = int(parents_np[node_index])
            parent_logw = path_logw_by_index[parent]
            parent_ranks = path_ranks_by_index[parent]
            if 1 < topk:
                push_candidate(
                    parent,
                    depth,
                    1,
                    parent_logw + float(top_log_probs_np[depth - 1, 1]),
                    parent_ranks + (1,),
                )
        if force_chain_len < depth_limit:
            push_candidate(
                parent_index,
                force_chain_len + 1,
                0,
                logw + float(top_log_probs_np[force_chain_len, 0]),
                ranks + (0,),
            )

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(top_token_ids_np[depth - 1, rank])
        if (parent_index, depth, rank) in added_keys:
            continue
        if token_id in child_maps[parent_index]:
            continue
        current_index = add_node(parent_index, depth, rank, logw, ranks)

        # Push sibling: same parent, next rank at the same depth.
        if rank + 1 < topk:
            parent_logw = path_logw_by_index[parent_index]
            sibling_logw = parent_logw + float(
                top_log_probs_np[depth - 1, rank + 1]
            )
            push_candidate(
                parent_index,
                depth,
                rank + 1,
                sibling_logw,
                path_ranks_by_index[parent_index] + (rank + 1,),
            )

        # Push first child: go one level deeper, take rank-0 token.
        if depth < depth_limit:
            child_logw = logw + float(top_log_probs_np[depth, 0])
            push_candidate(current_index, depth + 1, 0, child_logw, ranks + (0,))

    current_length = 1 + node_count
    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        torch.from_numpy(node_ranks_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        _build_visibility_from_parents(parents_np[:current_length]),
    )

def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_token_ids: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree to find the longest accepted path.

    After the target model runs a forward pass over the whole tree, this
    function greedily follows the path of accepted tokens from the root.

    Args:
        child_maps: As returned by :func:`build_ddtree_tree`.
        posterior_token_ids: List of token ids sampled from the target
            model's logits, one per tree node (root included, in node
            index order).

    Returns:
        accepted_indices: Node indices (including root at 0) that form
            the accepted prefix.
        bonus_token_id: The next token id to emit after the accepted
            prefix (sampled by the target model at the last accepted
            node).
    """
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_token_ids[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_token_ids[current_index])

    return accepted_indices, next_token



@triton.jit
def _ddtree_lm_head_argmax_kernel(
    hidden_ptr,
    weight_ptr,
    local_max_ptr,
    local_idx_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    org_vocab_size: tl.constexpr,
    hidden_stride0: tl.constexpr,
    hidden_stride1: tl.constexpr,
    weight_stride0: tl.constexpr,
    weight_stride1: tl.constexpr,
    local_stride0: tl.constexpr,
    block_v: tl.constexpr,
    block_h: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs_v = block * block_v + tl.arange(0, block_v)
    offs_h = tl.arange(0, block_h)
    acc = tl.zeros((block_v,), dtype=tl.float32)

    for h0 in range(0, hidden_size, block_h):
        h_idx = h0 + offs_h
        h = tl.load(
            hidden_ptr + row * hidden_stride0 + h_idx * hidden_stride1,
            mask=h_idx < hidden_size,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr
            + offs_v[:, None] * weight_stride0
            + h_idx[None, :] * weight_stride1,
            mask=(offs_v[:, None] < vocab_size) & (h_idx[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * h[None, :], axis=1)

    acc = tl.where(offs_v < org_vocab_size, acc, -float("inf"))
    max_val = tl.max(acc, axis=0)
    winner = tl.min(tl.where(acc == max_val, offs_v, vocab_size + block_v), axis=0)
    tl.store(local_max_ptr + row * local_stride0 + block, max_val)
    tl.store(local_idx_ptr + row * local_stride0 + block, winner)


def ddtree_fused_lm_head_argmax(
    lm_head: torch.nn.Module,
    logits_processor: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    weight = getattr(lm_head, "weight", None)
    bias = getattr(lm_head, "bias", None)
    if weight is None or bias is not None:
        return None
    if not hidden_states.is_cuda or not weight.is_cuda:
        return None
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return None
    if weight.dtype not in (torch.float16, torch.bfloat16):
        return None
    if hidden_states.shape[-1] != weight.shape[-1]:
        return None
    scale = getattr(logits_processor, "scale", 1.0)
    if scale <= 0.0 and scale != 1.0:
        return None

    hidden = hidden_states.contiguous()
    weight = weight.contiguous()
    n_rows = int(hidden.shape[0])
    hidden_size = int(hidden.shape[1])
    vocab_size = int(weight.shape[0])
    org_vocab_size = int(getattr(logits_processor, "org_vocab_size", vocab_size))
    if n_rows == 0:
        return torch.empty(0, dtype=torch.long, device=hidden.device)

    block_v = int(os.environ.get("DDTREE_FUSED_ARGMAX_BLOCK_V", "64"))
    block_h = int(os.environ.get("DDTREE_FUSED_ARGMAX_BLOCK_H", "64"))
    num_blocks = triton.cdiv(vocab_size, block_v)
    local_max = torch.empty((n_rows, num_blocks), dtype=torch.float32, device=hidden.device)
    local_idx = torch.empty((n_rows, num_blocks), dtype=torch.int64, device=hidden.device)
    _ddtree_lm_head_argmax_kernel[(n_rows, num_blocks)](
        hidden,
        weight,
        local_max,
        local_idx,
        n_rows,
        hidden_size,
        vocab_size,
        org_vocab_size,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        local_max.stride(0),
        block_v,
        block_h,
        num_warps=4,
    )
    block_ids = local_max.argmax(dim=-1, keepdim=True)
    return local_idx.gather(1, block_ids).squeeze(1).to(torch.long)


def ddtree_verify_argmax_ids(
    posterior_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    child_maps: list[list[dict[int, int]]],
    budget: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    posterior_cpu = posterior_token_ids.view(batch_size, budget + 1).cpu().tolist()
    draft_tokens_cpu = draft_token_ids.view(batch_size, budget).cpu().tolist()
    output = torch.full((batch_size, budget + 1), -1, dtype=torch.int32)
    gdn_state_indices = torch.ones(batch_size, dtype=torch.int32)
    for r in range(batch_size):
        accepted_indices, bonus_token = follow_verified_tree(child_maps[r], posterior_cpu[r])
        gdn_state_indices[r] = int(accepted_indices[-1]) + 1
        out_pos = 0
        for node_idx in accepted_indices[1:]:
            output[r, out_pos] = draft_tokens_cpu[r][node_idx - 1]
            out_pos += 1
        output[r, out_pos] = int(bonus_token)
    return output.to(device), gdn_state_indices.to(device)

def ddtree_verify(
    logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    child_maps: list[list[dict[int, int]]],
    budget: int,
    batch_size: int,
    device: torch.device,
    return_gdn_state_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Tree-aware verification for DDTree speculative decoding.

    After the target model's forward pass over all tree nodes (with
    ancestor-only attention), this function traces the longest accepted
    path through the tree per request using the target model's per-node
    greedy predictions.

    In contrast to flat spec-decode rejection sampling (which scans
    draft tokens left-to-right), DDTree must follow tree edges: after
    accepting a node, the next comparison is against that node's
    children, not its siblings.

    Args:
        logits:                float[num_logits, vocab_size] — target model logits.
        target_logits_indices: int[batch * budget] — logit row per draft node.
        bonus_logits_indices:  int[batch] — logit row for the last node per request.
        draft_token_ids:       int[batch * budget] — draft tokens in tree-node order.
        child_maps:            list[batch] of per-request dicts from build_ddtree_tree.
        budget:                number of draft nodes per request (root excluded).
        batch_size:            number of requests in the batch.
        device:                target device for the returned tensor.

    Returns:
        int32[batch, budget+1] — accepted path tokens + bonus token, -1 padded.
    """

    # posterior[r][i]   = target's argmax at node i's position, req r
    # posterior[r][B]   = target's argmax at the last node (bonus), req r
    # Example (batch=2, budget=3, vocab=3):
    #   target_logits_indices = [0, 1, 2, 3, 4, 5]   # 6 rows, one per (req, node) pair
    #
    #   logits[0] = [0.1, 0.9, 0.1] - token 1   (req 0, node 0)
    #   logits[1] = [0.5, 0.1, 0.3] - token 0   (req 0, node 1)
    #   logits[2] = [0.1, 0.1, 0.6] - token 2   (req 0, node 2)
    #   logits[3] = [0.8, 0.1, 0.1] - token 0   (req 1, node 0)
    #   logits[4] = [0.1, 0.6, 0.1] - token 1   (req 1, node 1)
    #   logits[5] = [0.2, 0.1, 0.7] - token 2   (req 1, node 2)
    #
    #   .argmax()  = [1, 0, 2, 0, 1, 2]
    #   .view(2,3) = [[1, 0, 2],   <- req 0
    #                 [0, 1, 2]]   <- req 1
    node_posterior = (
        logits[target_logits_indices].argmax(dim=-1).view(batch_size, budget)
    )
    bonus_posterior = logits[bonus_logits_indices].argmax(dim=-1)

    with _ddtree_cpu_profile_range("verify_node_posterior_d2h", batch=batch_size, budget=budget):
        node_posterior_cpu = node_posterior.cpu().tolist()
    with _ddtree_cpu_profile_range("verify_bonus_d2h", batch=batch_size):
        bonus_posterior_cpu = bonus_posterior.cpu().tolist()
    with _ddtree_cpu_profile_range("verify_draft_tokens_d2h", batch=batch_size, budget=budget):
        draft_tokens_cpu = draft_token_ids.view(batch_size, budget).cpu().tolist()

    _dbg = os.environ.get("DDTREE_DEBUG") == "1"

    output = torch.full((batch_size, budget + 1), -1, dtype=torch.int32)
    gdn_state_indices = torch.ones(batch_size, dtype=torch.int32)

    for r in range(batch_size):
        posterior = node_posterior_cpu[r] + [bonus_posterior_cpu[r]]

        accepted_indices, bonus_token = follow_verified_tree(child_maps[r], posterior)
        # GDN/Mamba speculative state slots are indexed by tree node, not by
        # accepted-path length. For a flat chain those are identical; for a
        # branched tree, the accepted leaf can live at any node index.
        gdn_state_indices[r] = int(accepted_indices[-1]) + 1

        if _dbg and r == 0:
            print(
                f"[ddtree_verify] draft={draft_tokens_cpu[r]}"
                f" posterior={posterior[:budget]}"
                f" acc_len={len(accepted_indices)}"
                f" gdn_state_idx={int(gdn_state_indices[r])}"
                f" maps={child_maps[r]}"
            )

        out_pos = 0
        for node_idx in accepted_indices[1:]:
            output[r, out_pos] = draft_tokens_cpu[r][node_idx - 1]
            out_pos += 1
        output[r, out_pos] = bonus_token

        with _ddtree_cpu_profile_range("verify_output_h2d", batch=batch_size, budget=budget):
            output = output.to(device)
        if return_gdn_state_indices:
            with _ddtree_cpu_profile_range("verify_gdn_h2d", batch=batch_size):
                gdn_state_indices = gdn_state_indices.to(device)
            return output, gdn_state_indices
        return output


def ddtree_verify_sglang_greedy(
    logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    child_maps: list[list[dict[int, int]]],
    budget: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang greedy tree verification using linked child/sibling tables.

    The preferred path loads SGLang's CUDA implementation of
    ``verify_tree_greedy``.  The local Triton kernel is kept only as a
    development fallback because the CUDA op is the behavior we want to port.
    """
    node_posterior = logits[target_logits_indices].argmax(dim=-1).view(
        batch_size, budget
    )
    bonus_posterior = logits[bonus_logits_indices].argmax(dim=-1).view(batch_size, 1)
    target_predict = torch.cat([node_posterior, bonus_posterior], dim=1).to(
        dtype=torch.int64
    )
    draft_tokens = draft_token_ids.view(batch_size, budget).to(dtype=torch.int64)
    (
        retrieve_next_token_i32,
        retrieve_next_sibling_i32,
        retrieve_next_token_i64,
        retrieve_next_sibling_i64,
        retrieve_index,
        row_offsets,
    ) = get_cached_retrieve_tables(child_maps, budget, device)

    output = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    gdn_state_indices = torch.empty(batch_size, dtype=torch.int32, device=device)

    if _load_sgl_dtree_ops():
        candidates = torch.cat(
            [
                torch.full(
                    (batch_size, 1),
                    -1,
                    dtype=torch.int64,
                    device=device,
                ),
                draft_tokens,
            ],
            dim=1,
        )
        accept_index = torch.full(
            (batch_size, budget + 1), -1, dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty(batch_size, dtype=torch.int32, device=device)
        predicts = torch.full(
            (batch_size * (budget + 1),), -1, dtype=torch.int32, device=device
        )
        torch.ops.sgl_dtree.verify_tree_greedy(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token_i64,
            retrieve_next_sibling_i64,
            target_predict,
        )
        # SGLang writes accepted row ids into ``accept_index``. Convert back
        # to vLLM's [accepted draft tokens..., bonus token, -1...] convention
        # without CPU synchronization.
        accepted_rows_i32 = accept_index - row_offsets
        accepted_rows = accepted_rows_i32.clamp_min(0).to(dtype=torch.long)
        gathered_candidates = candidates.gather(1, accepted_rows).to(dtype=torch.int32)
        output[:, :budget] = torch.where(
            torch.arange(budget, device=device).view(1, budget)
            < accept_token_num.view(batch_size, 1),
            gathered_candidates[:, 1 : budget + 1],
            output[:, :budget],
        )
        bonus_cols = accept_token_num.view(batch_size, 1).to(dtype=torch.long)
        bonus_rows = accepted_rows.gather(1, bonus_cols)
        bonus_tokens = target_predict.gather(1, bonus_rows).to(dtype=torch.int32)
        output.scatter_(1, bonus_cols, bonus_tokens)
        gdn_state_indices = bonus_rows.squeeze(1).to(dtype=torch.int32) + 1
        return output, gdn_state_indices

    _ddtree_verify_greedy_kernel[(batch_size,)](
        target_predict,
        draft_tokens,
        retrieve_next_token_i32,
        retrieve_next_sibling_i32,
        output,
        gdn_state_indices,
        budget,
        output.stride(0),
    )
    return output, gdn_state_indices


@triton.jit
def _ddtree_static_siblings_verify_kernel(
    target_predict,
    draft_token_ids,
    output,
    gdn_state_indices,
    budget: tl.constexpr,
    chain_len: tl.constexpr,
    branch_count: tl.constexpr,
    out_stride: tl.constexpr,
):
    req = tl.program_id(0)
    offs = tl.arange(0, 128)
    if budget + 1 <= 128:
        tl.store(output + req * out_stride + offs, -1, mask=offs < budget + 1)

    out_pos = tl.full((), 0, dtype=tl.int32)
    done = tl.full((), False, dtype=tl.int1)
    current = tl.full((), 0, dtype=tl.int32)

    for depth in range(0, chain_len):
        active = ~done
        next_token = tl.load(
            target_predict + req * (budget + 1) + current,
            mask=active,
            other=-2147483648,
        )
        chain_node = depth + 1
        chain_token = tl.load(
            draft_token_ids + req * budget + chain_node - 1,
            mask=active,
            other=-2147483648,
        )
        chain_match = active & (next_token == chain_token)
        if chain_match:
            tl.store(output + req * out_stride + out_pos, next_token)
            out_pos += 1
            current = chain_node

        sibling_match = tl.full((), False, dtype=tl.int1)
        sibling_node = chain_len + depth + 1
        if depth < branch_count:
            sib_active = active & (~chain_match)
            sibling_token = tl.load(
                draft_token_ids + req * budget + sibling_node - 1,
                mask=sib_active,
                other=-2147483648,
            )
            sibling_match = sib_active & (next_token == sibling_token)
            if sibling_match:
                tl.store(output + req * out_stride + out_pos, next_token)
                out_pos += 1
                bonus = tl.load(target_predict + req * (budget + 1) + sibling_node)
                tl.store(output + req * out_stride + out_pos, bonus)
                tl.store(gdn_state_indices + req, sibling_node + 1)
                done = True

        reject = active & (~chain_match) & (~sibling_match)
        if reject:
            tl.store(output + req * out_stride + out_pos, next_token)
            tl.store(gdn_state_indices + req, current + 1)
            done = True

    if ~done:
        bonus = tl.load(target_predict + req * (budget + 1) + chain_len)
        tl.store(output + req * out_stride + out_pos, bonus)
        tl.store(gdn_state_indices + req, chain_len + 1)


def ddtree_verify_static_siblings_greedy(
    logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    budget: int,
    batch_size: int,
    device: torch.device,
    chain_len: int,
    branch_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy verifier for the fixed chain+tail-siblings DDTree layout."""
    node_posterior = logits[target_logits_indices].argmax(dim=-1).view(
        batch_size, budget
    )
    bonus_posterior = logits[bonus_logits_indices].argmax(dim=-1).view(batch_size, 1)
    target_predict = torch.cat([node_posterior, bonus_posterior], dim=1).to(
        dtype=torch.int64
    )
    draft_tokens = draft_token_ids.view(batch_size, budget).to(dtype=torch.int64)
    output = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    gdn_state_indices = torch.empty(batch_size, dtype=torch.int32, device=device)
    _ddtree_static_siblings_verify_kernel[(batch_size,)](
        target_predict,
        draft_tokens,
        output,
        gdn_state_indices,
        budget,
        chain_len,
        branch_count,
        output.stride(0),
    )
    return output, gdn_state_indices


def ddtree_verify_static_siblings_lazy_logits(
    compute_logits: Callable[[torch.Tensor], torch.Tensor],
    sample_hidden_states: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    budget: int,
    batch_size: int,
    device: torch.device,
    chain_len: int,
    branch_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazy greedy verifier for the fixed chain+siblings layout.

    This avoids the generic child-map walker because the GPU-static topology
    intentionally uses dummy child-map keys to skip proposal-time token syncs.
    """

    def predict_one(req: int, pred_index: int) -> int:
        if pred_index < budget:
            row = int(target_logits_indices[req * budget + pred_index].item())
        else:
            row = int(bonus_logits_indices[req].item())
        logits = compute_logits(sample_hidden_states[row : row + 1])
        return int(logits.argmax(dim=-1).item())

    output = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    gdn_state_indices = torch.empty(batch_size, dtype=torch.int32, device=device)
    draft_tokens = draft_token_ids.view(batch_size, budget)

    for req in range(batch_size):
        out_tokens: list[int] = []
        current = 0
        done = False
        for depth in range(chain_len):
            next_token = predict_one(req, current)
            chain_node = depth + 1
            chain_token = int(draft_tokens[req, chain_node - 1].item())
            if next_token == chain_token:
                out_tokens.append(next_token)
                current = chain_node
                continue

            sibling_node = chain_len + depth + 1
            if depth < branch_count:
                sibling_token = int(draft_tokens[req, sibling_node - 1].item())
                if next_token == sibling_token:
                    out_tokens.append(next_token)
                    out_tokens.append(predict_one(req, sibling_node))
                    gdn_state_indices[req] = sibling_node + 1
                    done = True
                    break

            out_tokens.append(next_token)
            gdn_state_indices[req] = current + 1
            done = True
            break

        if not done:
            out_tokens.append(predict_one(req, chain_len))
            gdn_state_indices[req] = chain_len + 1

        if out_tokens:
            output[req, : len(out_tokens)] = torch.tensor(
                out_tokens, dtype=torch.int32, device=device
            )

    return output, gdn_state_indices


@triton.jit
def _ddtree_multi_root_chains_verify_kernel(
    target_predict,
    draft_token_ids,
    output,
    gdn_state_indices,
    budget: tl.constexpr,
    chain_len: tl.constexpr,
    num_chains: tl.constexpr,
    out_stride: tl.constexpr,
):
    req = tl.program_id(0)
    offs = tl.arange(0, 128)
    if budget + 1 <= 128:
        tl.store(output + req * out_stride + offs, -1, mask=offs < budget + 1)

    first_token = tl.load(target_predict + req * (budget + 1))
    selected_chain = tl.full((), -1, dtype=tl.int32)

    for chain in range(0, num_chains):
        first_node = chain * chain_len + 1
        draft_token = tl.load(draft_token_ids + req * budget + first_node - 1)
        matched = (selected_chain < 0) & (first_token == draft_token)
        selected_chain = tl.where(matched, chain, selected_chain)

    if selected_chain < 0:
        tl.store(output + req * out_stride, first_token)
        tl.store(gdn_state_indices + req, 1)
        return

    out_pos = tl.full((), 0, dtype=tl.int32)
    current_node = selected_chain * chain_len + 1
    tl.store(output + req * out_stride + out_pos, first_token)
    out_pos += 1

    rejected = tl.full((), False, dtype=tl.int1)
    for depth in range(1, chain_len):
        if not rejected:
            next_token = tl.load(target_predict + req * (budget + 1) + current_node)
            next_node = selected_chain * chain_len + depth + 1
            draft_token = tl.load(draft_token_ids + req * budget + next_node - 1)
            tl.store(output + req * out_stride + out_pos, next_token)
            out_pos += 1
            matched = next_token == draft_token
            current_node = tl.where(matched, next_node, current_node)
            rejected = ~matched

    if rejected:
        tl.store(gdn_state_indices + req, current_node + 1)
    else:
        bonus = tl.load(target_predict + req * (budget + 1) + current_node)
        tl.store(output + req * out_stride + out_pos, bonus)
        tl.store(gdn_state_indices + req, current_node + 1)


def ddtree_verify_multi_root_chains_greedy(
    logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    budget: int,
    batch_size: int,
    device: torch.device,
    chain_len: int,
    num_chains: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy verifier for static multiple root-to-leaf chain layout."""
    node_posterior = logits[target_logits_indices].argmax(dim=-1).view(
        batch_size, budget
    )
    bonus_posterior = logits[bonus_logits_indices].argmax(dim=-1).view(batch_size, 1)
    target_predict = torch.cat([node_posterior, bonus_posterior], dim=1).to(
        dtype=torch.int64
    )
    draft_tokens = draft_token_ids.view(batch_size, budget).to(dtype=torch.int64)
    output = torch.full(
        (batch_size, budget + 1), -1, dtype=torch.int32, device=device
    )
    gdn_state_indices = torch.empty(batch_size, dtype=torch.int32, device=device)
    _ddtree_multi_root_chains_verify_kernel[(batch_size,)](
        target_predict,
        draft_tokens,
        output,
        gdn_state_indices,
        budget,
        chain_len,
        num_chains,
        output.stride(0),
    )
    return output, gdn_state_indices


def ddtree_verify_lazy_logits(
    compute_logits: Callable[[torch.Tensor], torch.Tensor],
    sample_hidden_states: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    child_maps: list[list[dict[int, int]]],
    budget: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify a DDTree while projecting logits only for rows that matter.

    The target transformer still runs all tree nodes. For greedy verification,
    full-vocab logits are needed for the root and tree nodes that have children;
    leaf logits are only needed for the single leaf on the accepted path to
    produce the bonus token. This avoids projecting every dead leaf.
    """
    batch_size = len(child_maps)
    target_logits_indices = target_logits_indices.view(batch_size, budget)
    draft_tokens_cpu = draft_token_ids.view(batch_size, budget).cpu().tolist()

    internal_rows: list[int] = []
    internal_nodes_by_req: list[list[int]] = []
    for r, maps in enumerate(child_maps):
        nodes = [
            node_idx
            for node_idx, children in enumerate(maps[: budget + 1])
            if children
        ]
        internal_nodes_by_req.append(nodes)
        for node_idx in nodes:
            if node_idx < budget:
                internal_rows.append(int(target_logits_indices[r, node_idx].item()))
            else:
                internal_rows.append(int(bonus_logits_indices[r].item()))

    posterior_by_req: list[dict[int, int]] = [dict() for _ in range(batch_size)]
    if internal_rows:
        rows_t = torch.tensor(internal_rows, dtype=torch.long, device=device)
        internal_posterior = (
            compute_logits(sample_hidden_states[rows_t]).argmax(dim=-1).cpu().tolist()
        )
        cursor = 0
        for r, nodes in enumerate(internal_nodes_by_req):
            for node_idx in nodes:
                posterior_by_req[r][node_idx] = int(internal_posterior[cursor])
                cursor += 1

    accepted_paths: list[list[int]] = []
    bonus_tokens: list[int | None] = []
    leaf_bonus_rows: list[int] = []
    leaf_bonus_req_order: list[int] = []

    for r, maps in enumerate(child_maps):
        accepted_indices = [0]
        current_index = 0
        bonus_token: int | None = None

        while True:
            next_token = posterior_by_req[r].get(current_index)
            if next_token is None:
                leaf_bonus_req_order.append(r)
                if current_index < budget:
                    leaf_bonus_rows.append(
                        int(target_logits_indices[r, current_index].item())
                    )
                else:
                    leaf_bonus_rows.append(int(bonus_logits_indices[r].item()))
                break
            if next_token not in maps[current_index]:
                bonus_token = int(next_token)
                break
            current_index = maps[current_index][next_token]
            accepted_indices.append(current_index)

        accepted_paths.append(accepted_indices)
        bonus_tokens.append(bonus_token)

    if leaf_bonus_rows:
        rows_t = torch.tensor(leaf_bonus_rows, dtype=torch.long, device=device)
        leaf_bonus = (
            compute_logits(sample_hidden_states[rows_t]).argmax(dim=-1).cpu().tolist()
        )
        for req_idx, token_id in zip(leaf_bonus_req_order, leaf_bonus):
            bonus_tokens[req_idx] = int(token_id)

    output = torch.full((batch_size, budget + 1), -1, dtype=torch.int32)
    gdn_state_indices = torch.ones(batch_size, dtype=torch.int32)
    for r, accepted_indices in enumerate(accepted_paths):
        gdn_state_indices[r] = int(accepted_indices[-1]) + 1
        out_pos = 0
        for node_idx in accepted_indices[1:]:
            output[r, out_pos] = draft_tokens_cpu[r][node_idx - 1]
            out_pos += 1
        output[r, out_pos] = int(bonus_tokens[r])

    return output.to(device), gdn_state_indices.to(device)


class DDTreeProposer(DFlashProposer):
    """DFlash proposer with a dynamic best-first draft tree.

    Each proposal step:
    1. Runs the DFlash draft model to obtain per-position logits.
    2. Calls :func:`build_ddtree_tree` per request on its own draft logits
       to select the ``budget`` most-probable tree nodes via a best-first heap.
    3. Updates the target model's ``TreeAttentionMetadataBuilder`` with
       the new visibility mask so the next verification pass uses the
       correct ancestor-only attention bias.
    4. Returns per-request draft tokens in tree-node order.

    Each request gets its own tree topology derived from its own draft logits.

    Requirements:
    - attention_config.backend = "TREE_ATTN": needed for per-request
      ancestor-only attention masking over the draft tree.
    - speculative_config.method = "ddtree": selects this proposer.

    Example (budget=4, num_speculative_tokens=4):

        DFlash produces marginal logits for 4 depth positions.
        For this example, assume probabilities are concentrated enough
        that the tree only branches to depth 2.
        root token = "The"

        Top-5 most probable sequences by cumulative log-prob:
          rank 1: "The" -> cat -> sat
          rank 2: "The" -> cat -> ran
          rank 3: "The" -> dog -> sat
          rank 4: "The" -> cat -> red
          rank 5: "The" -> dog -> ran

        budget=4 means: expand top-4 nodes from the heap:
          pop 1: cat   (root->cat)       -> node 1
          pop 2: sat   (root->cat->sat)  -> node 2  child of node 1
          pop 3: dog   (root->dog)       -> node 3
          pop 4: sat   (root->dog->sat)  -> node 4  child of node 3
          stop.  rank 5 (dog->ran) NOT in tree — out of budget.

                    root ("The")          <- node 0
                   /            \\
                cat               dog     <- node 1, node 3
                 |                 |
                sat               sat    <- node 2, node 4

        child_maps:
          child_maps[0] = {cat: 1, dog: 3}
          child_maps[1] = {sat: 2}
          child_maps[2] = {}                  <- leaf
          child_maps[3] = {sat: 4}
          child_maps[4] = {}                  <- leaf

        draft output: [batch, budget=4] in node-index order
          [cat, sat, dog, sat]
           ^1   ^2   ^3   ^4

        visibility mask [5, 5]  (budget+1 nodes including root)
        rule: row i attends to col j iff j is an ancestor of i (or j==i)

              0  1  2  3  4
          0 [ T  F  F  F  F ]  root
          1 [ T  T  F  F  F ]  cat           (sees root, itself)
          2 [ T  T  T  F  F ]  sat under cat (sees root, cat, itself)
          3 [ T  F  F  T  F ]  dog           (sees root, itself)
          4 [ T  F  F  T  T ]  sat under dog (sees root, dog, itself)

        note: nodes 2 and 4 are both "sat" but attend to different contexts.

        verification examples:
          target predicts [cat, sat, ...]:
            node 0 -> cat in child_maps[0] -> node 1  accepted
            node 1 -> sat in child_maps[1] -> node 2  accepted
            node 2 -> no children          -> stop
            result: "The cat sat" + bonus token

          target predicts [dog, sat, ...]:
            node 0 -> dog in child_maps[0] -> node 3  accepted
            node 3 -> sat in child_maps[3] -> node 4  accepted
            node 4 -> no children          -> stop
            result: "The dog sat" + bonus token
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "ddtree"
        super().__init__(vllm_config, device, runner)

        self._runner = runner
        self._budget = self.num_speculative_tokens
        self._child_maps: list[list[dict[int, int]]] | None = None
        self._node_depths: list[torch.Tensor] | None = None
        self._tree_req_ids: list[str] | None = None
        self._draft_token_ids_cpu_cache: list[list[int]] | None = None
        self._uploaded_static_visibility_key: tuple[int, int, int] | None = None

    @override
    def build_per_group_and_layer_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        # Skip DFlashProposer's causal=False assertion; tree attention enforces
        # non-causal masking via qq_bias, not via the causal flag.
        return SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata(
            self, cad, draft_index
        )

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Build a per-request dynamic tree and return draft tokens.

        Overrides the DFlash greedy argmax with a heap-based tree built
        independently for each request from its own logits.  Each request
        gets its own tree topology (child_maps) and visibility mask, allowing
        heterogeneous batches to have different speculative paths.

        Args:
            hidden_states: [batch * depth, hidden_size] — the DFlash
                model's output hidden states for the speculative positions.

        Returns:
            [batch * budget] int64 — flattened draft token IDs in
            tree-node order, varying per request.
        """
        depth = self.dflash_draft_depth  # DFlash depth can be below tree budget.
        batch_size = hidden_states.shape[0] // depth

        if self._runner is not None:
            prompt_lens = self._runner.input_batch.num_prompt_tokens[:batch_size]
            if not should_use_ddtree_for_prompt_lens(prompt_lens):
                self._child_maps = None
                self._node_depths = None
                self._tree_req_ids = None
                self._draft_token_ids_cpu_cache = None
                self._reset_target_flat_chain_visibility()
                return SpecDecodeBaseProposer._greedy_sample(self, hidden_states)

        force_chain_prefix = int(
            os.environ.get("DDTREE_FORCE_CHAIN_PREFIX", "0") or "0"
        )
        static_siblings = int(os.environ.get("DDTREE_STATIC_SIBLINGS", "0") or "0")
        static_chain_len = max(0, min(force_chain_prefix, self._budget, depth))
        static_branch_count = max(
            0, min(static_siblings, self._budget - static_chain_len, static_chain_len)
        )
        use_static_fast_proposer = (
            os.environ.get("DDTREE_STATIC_FAST_PROPOSER", "0") == "1"
            and static_branch_count > 0
            and static_chain_len + static_branch_count == self._budget
            and self.use_local_argmax_reduction
            and hasattr(self.model, "get_top_tokens")
        )

        logits_per_req = None
        top1_per_req = None
        branch_top2_per_req = None
        if use_static_fast_proposer:
            with _ddtree_profile_range("static_top1_tokens", rows=hidden_states.shape[0]):
                top1_flat = SpecDecodeBaseProposer._greedy_sample(self, hidden_states)
                top1_per_req = top1_flat.view(batch_size, depth)
            hidden_by_req = hidden_states.view(batch_size, depth, hidden_states.shape[-1])
            branch_hidden = hidden_by_req[:, :static_branch_count, :].reshape(
                batch_size * static_branch_count, hidden_states.shape[-1]
            )
            with _ddtree_profile_range("static_branch_logits", rows=branch_hidden.shape[0]):
                branch_logits = self.model.compute_logits(branch_hidden)
            with _ddtree_profile_range("static_branch_top2", rows=branch_hidden.shape[0]):
                branch_top2_per_req = torch.topk(
                    branch_logits.float(), k=2, dim=-1
                ).indices[:, 1].view(batch_size, static_branch_count)
        else:
            with _ddtree_profile_range("draft_logits", rows=hidden_states.shape[0]):
                logits = self.model.compute_logits(hidden_states)
                vocab_size = logits.shape[-1]
                logits_per_req = logits.view(batch_size, depth, vocab_size)

        if self._runner is not None:
            self._tree_req_ids = list(self._runner.input_batch.req_ids)[:batch_size]
        else:
            self._tree_req_ids = None

        all_child_maps: list[list[dict[int, int]]] = []
        all_draft_tokens: list[torch.Tensor] = []
        all_draft_token_lists: list[list[int]] = []
        all_node_depths: list[torch.Tensor] = []
        all_visibility: list[torch.Tensor] = []
        target_size = self._budget + 1  # [N+1, N+1] per request
        gpu_static_tree = (
            os.environ.get("DDTREE_GPU_STATIC_TREE") == "1"
            and os.environ.get("DDTREE_SGL_VERIFY") == "1"
        )

        with _ddtree_profile_range("tree_build", batch=batch_size, budget=self._budget):
            for r in range(batch_size):
                if use_static_fast_proposer:
                    assert top1_per_req is not None
                    assert branch_top2_per_req is not None
                    (
                        node_token_ids,
                        node_depths,
                        _,
                        _,
                        child_maps,
                        visibility,
                    ) = build_static_sibling_tree_from_token_ids(
                        top1_per_req[r],
                        branch_top2_per_req[r],
                        budget=self._budget,
                        chain_len=static_chain_len,
                        branch_count=static_branch_count,
                    )
                else:
                    assert logits_per_req is not None
                    (
                        node_token_ids,
                        node_depths,
                        _,
                        _,
                        child_maps,
                        visibility,
                    ) = build_ddtree_tree(
                        logits_per_req[r],
                        budget=self._budget,
                        force_chain_prefix=force_chain_prefix,
                    )
                all_child_maps.append(child_maps)

                # Draft tokens for this request, padded to budget.
                budget_actual = node_token_ids.shape[0]
                if gpu_static_tree:
                    token_list = []
                else:
                    with _ddtree_cpu_profile_range("greedy_node_tokens_tolist", budget=budget_actual):
                        token_list = [int(token_id) for token_id in node_token_ids.tolist()]
                with _ddtree_cpu_profile_range("greedy_node_tokens_h2d", budget=budget_actual):
                    tokens = node_token_ids.to(self.device)
                if budget_actual < self._budget:
                    if not gpu_static_tree:
                        token_list.extend([0] * (self._budget - budget_actual))
                    with _ddtree_cpu_profile_range("greedy_node_tokens_pad", budget=self._budget):
                        tokens = torch.cat(
                            [tokens, tokens.new_zeros(self._budget - budget_actual)]
                        )
                if not gpu_static_tree:
                    all_draft_token_lists.append(token_list)
                all_draft_tokens.append(tokens)
                if os.environ.get("DDTREE_STATIC_INTERLEAVE") == "1":
                    depths = node_depths.to(self.device, dtype=torch.long)
                    if budget_actual < self._budget:
                        depths = torch.cat(
                            [depths, depths.new_zeros(self._budget - budget_actual)]
                        )
                    all_node_depths.append(depths)

                # Visibility mask padded to [target_size, target_size].
                vis_size = visibility.shape[0]  # budget_actual + 1
                if vis_size < target_size:
                    padded = torch.zeros(target_size, target_size, dtype=torch.bool)
                    padded[:vis_size, :vis_size] = visibility
                    visibility = padded
                all_visibility.append(visibility)

        self._child_maps = all_child_maps
        self._node_depths = (
            all_node_depths
            if os.environ.get("DDTREE_STATIC_INTERLEAVE") == "1"
            else None
        )
        self._draft_token_ids_cpu_cache = None if gpu_static_tree else all_draft_token_lists

        # Stack per-request masks. For the batch-1 latency harness, a 2D bias
        # is exact and uses the cheaper TreeAttention path. Batched requests may
        # have different DDTree topologies, so they still require 3D per-request
        # bias.
        with _ddtree_cpu_profile_range("greedy_visibility_stack", batch=batch_size, budget=self._budget):
            stacked = (
                all_visibility[0]
                if batch_size == 1
                else torch.stack(all_visibility, dim=0)
            )
        static_visibility_key = None
        if static_siblings > 0 and batch_size == 1:
            static_visibility_key = (
                static_chain_len,
                static_branch_count,
                static_chain_len + static_branch_count,
                int(os.environ.get("DDTREE_STATIC_INTERLEAVE") == "1"),
            )

        with _ddtree_profile_range("bias_update", batch=batch_size, budget=self._budget):
            if (
                static_visibility_key is not None
                and self._uploaded_static_visibility_key == static_visibility_key
            ):
                pass
            elif batch_size == 1:
                self._update_target_tree_visibility(stacked)
                self._uploaded_static_visibility_key = static_visibility_key
            else:
                tree_attn_bias = torch.where(
                    stacked.to(self.device),
                    torch.zeros(1, dtype=torch.float32, device=self.device),
                    torch.full(
                        (1,), float("-inf"), dtype=torch.float32, device=self.device
                    ),
                )
                self._update_target_tree_attn_bias(tree_attn_bias)
                self._uploaded_static_visibility_key = None

        with _ddtree_profile_range("draft_pack", batch=batch_size, budget=self._budget):
            if batch_size == 1:
                return all_draft_tokens[0].to(torch.long)
            draft = torch.stack(all_draft_tokens, dim=0)  # [batch, budget]
            return draft.reshape(-1).to(torch.long)  # [batch * budget]

    def _reset_target_flat_chain_visibility(self) -> None:
        if self._runner is None:
            return
        self._uploaded_static_visibility_key = None
        found_tree_attn = False
        for attn_groups in self._runner.attn_groups:
            for attn_group in attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, TreeAttentionMetadataBuilder):
                    builder.tree_attn_bias = torch.empty(0, device=self.device)
                    builder.reorder_batch_threshold = self._budget
                    builder._tree_decode_threshold = self._budget + 1
                    found_tree_attn = True
        assert found_tree_attn, (
            "DDTreeProposer requires the target model to use the TREE_ATTN "
            "attention backend. Set attention_config.backend = 'TREE_ATTN' "
            "in your vllm config."
        )

    def _update_target_tree_visibility(self, visibility: torch.Tensor) -> None:
        if self._runner is None:
            return
        found_tree_attn = False
        visibility_gpu = visibility.to(self.device)
        target_size = visibility_gpu.shape[-1]
        for attn_groups in self._runner.attn_groups:
            for attn_group in attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, TreeAttentionMetadataBuilder):
                    current_bias = builder.tree_attn_bias
                    if (
                        current_bias is None
                        or current_bias.shape != visibility_gpu.shape
                        or current_bias.device != visibility_gpu.device
                    ):
                        current_bias = torch.empty(
                            visibility_gpu.shape,
                            dtype=torch.float32,
                            device=visibility_gpu.device,
                        )
                        builder.tree_attn_bias = current_bias
                    current_bias.fill_(float("-inf"))
                    current_bias.masked_fill_(visibility_gpu, 0.0)
                    builder.reorder_batch_threshold = target_size - 1
                    builder._tree_decode_threshold = target_size
                    found_tree_attn = True
        assert found_tree_attn, (
            "DDTreeProposer requires the target model to use the TREE_ATTN "
            "attention backend. Set attention_config.backend = 'TREE_ATTN' "
            "in your vllm config."
        )

    def _update_target_tree_attn_bias(self, tree_attn_bias: torch.Tensor) -> None:
        """Push a new tree_attn_bias to all TreeAttentionMetadataBuilders.

        Called after each propose step so the target model's next
        verification pass uses the freshly-built tree topology.
        """
        if self._runner is None:
            return
        found_tree_attn = False
        for attn_groups in self._runner.attn_groups:
            for attn_group in attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, TreeAttentionMetadataBuilder):
                    current_bias = builder.tree_attn_bias
                    if (
                        current_bias is not None
                        and current_bias.shape == tree_attn_bias.shape
                        and current_bias.dtype == tree_attn_bias.dtype
                        and current_bias.device == tree_attn_bias.device
                    ):
                        current_bias.copy_(tree_attn_bias)
                    else:
                        builder.tree_attn_bias = tree_attn_bias
                    # N = budget; reorder threshold = N so spec-decodes (q_len=N+1)
                    # land in region 2 (long_extend) after regular decodes (region 0),
                    # giving forward() a clean split point.
                    builder.reorder_batch_threshold = tree_attn_bias.shape[-2] - 1
                    # build() still needs to include spec-decodes in the decode
                    # bucket (not prefill), so keep decode threshold at N+1.
                    builder._tree_decode_threshold = tree_attn_bias.shape[-2]
                    found_tree_attn = True
        assert found_tree_attn, (
            "DDTreeProposer requires the target model to use the TREE_ATTN "
            "attention backend. Set attention_config.backend = 'TREE_ATTN' "
            "in your vllm config."
        )
