# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree import (
    build_retrieve_from_child_maps,
    build_ddtree_tree,
    build_static_sibling_tree_from_token_ids,
    ddtree_verify,
    ddtree_verify_lazy_logits,
    ddtree_verify_multi_root_chains_greedy,
    ddtree_verify_sglang_greedy,
    ddtree_verify_static_siblings_lazy_logits,
    ddtree_verify_static_siblings_greedy,
    resolve_ddtree_candidate_topk,
    should_use_ddtree_for_prompt_lens,
)


def _logits_for_argmax(tokens: list[int], vocab_size: int = 128) -> torch.Tensor:
    logits = torch.full((len(tokens), vocab_size), -1000.0)
    for row, token_id in enumerate(tokens):
        logits[row, token_id] = 1000.0
    return logits


def test_ddtree_verify_returns_accepted_leaf_state_slot_for_branch() -> None:
    # Tree node order:
    #
    #   root
    #    ├─ 10 -> node 1
    #    │   └─ 30 -> node 2
    #    └─ 20 -> node 3
    #        └─ 40 -> node 4
    #
    # The accepted path is root -> node 3 -> node 4. Its accepted-token length
    # is 2, but its GDN state slot is node index 4. Flat-chain bookkeeping would
    # incorrectly pick slot 2 here.
    child_maps = [[{10: 1, 20: 3}, {30: 2}, {}, {40: 4}, {}]]
    draft_token_ids = torch.tensor([10, 30, 20, 40], dtype=torch.int64)

    # Posterior rows correspond to tree nodes 0..3, plus bonus row 4.
    # row 0 predicts 20, so node 3 is accepted.
    # row 3 predicts 40, so node 4 is accepted.
    # row 4 is the bonus token produced from node 4.
    logits = _logits_for_argmax([20, 99, 99, 40, 77])

    output, gdn_state_indices = ddtree_verify(
        logits=logits,
        target_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        bonus_logits_indices=torch.tensor([4], dtype=torch.long),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        batch_size=1,
        device=torch.device("cpu"),
        return_gdn_state_indices=True,
    )

    assert output.tolist() == [[20, 40, 77, -1, -1]]
    assert int((output != -1).sum().item()) == 3
    assert gdn_state_indices.tolist() == [5]


def test_ddtree_verify_state_slot_matches_flat_chain_count() -> None:
    # Flat chain: root -> node 1 -> node 2. Here node slot and accepted length
    # remain aligned, so the DDTree state-slot path is backward-compatible with
    # ordinary DFlash-style chain semantics.
    child_maps = [[{10: 1}, {30: 2}, {}]]
    draft_token_ids = torch.tensor([10, 30], dtype=torch.int64)
    logits = _logits_for_argmax([10, 30, 77])

    output, gdn_state_indices = ddtree_verify(
        logits=logits,
        target_logits_indices=torch.tensor([0, 1], dtype=torch.long),
        bonus_logits_indices=torch.tensor([2], dtype=torch.long),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=2,
        batch_size=1,
        device=torch.device("cpu"),
        return_gdn_state_indices=True,
    )

    assert output.tolist() == [[10, 30, 77]]
    assert int((output != -1).sum().item()) == 3
    assert gdn_state_indices.tolist() == [3]


def test_ddtree_verify_default_return_is_output_tensor() -> None:
    logits = _logits_for_argmax([11, 77])

    output = ddtree_verify(
        logits=logits,
        target_logits_indices=torch.tensor([0], dtype=torch.long),
        bonus_logits_indices=torch.tensor([1], dtype=torch.long),
        draft_token_ids=torch.tensor([10], dtype=torch.int64),
        child_maps=[[{10: 1}, {}]],
        budget=1,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert isinstance(output, torch.Tensor)
    assert output.tolist() == [[11, -1]]


def test_ddtree_lazy_logits_matches_full_verify_and_skips_dead_leaf() -> None:
    child_maps = [[{10: 1, 20: 3}, {30: 2}, {}, {40: 4}, {}]]
    draft_token_ids = torch.tensor([10, 30, 20, 40], dtype=torch.int64)

    # Rows are [root, node1, node2, node3, node4]. The accepted path is
    # root -> node3 -> node4. Lazy verification should never project node2,
    # because it is a dead leaf on the unaccepted branch.
    row_to_token = [20, 99, 88, 40, 77]
    hidden = torch.arange(len(row_to_token), dtype=torch.float32).view(-1, 1)
    projected_rows: list[int] = []

    def compute_logits(rows: torch.Tensor) -> torch.Tensor:
        row_ids = [int(x) for x in rows[:, 0].tolist()]
        projected_rows.extend(row_ids)
        return _logits_for_argmax([row_to_token[i] for i in row_ids])

    lazy_output, lazy_gdn_state_indices = ddtree_verify_lazy_logits(
        compute_logits=compute_logits,
        sample_hidden_states=hidden,
        target_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        bonus_logits_indices=torch.tensor([4], dtype=torch.long),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        device=torch.device("cpu"),
    )
    full_output, full_gdn_state_indices = ddtree_verify(
        logits=_logits_for_argmax(row_to_token),
        target_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        bonus_logits_indices=torch.tensor([4], dtype=torch.long),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        batch_size=1,
        device=torch.device("cpu"),
        return_gdn_state_indices=True,
    )

    assert lazy_output.tolist() == full_output.tolist()
    assert lazy_gdn_state_indices.tolist() == full_gdn_state_indices.tolist()
    assert projected_rows == [0, 1, 3, 4]


def test_ddtree_lazy_logits_uses_spec_metadata_row_indices() -> None:
    child_maps = [[{10: 1, 20: 3}, {30: 2}, {}, {40: 4}, {}]]
    draft_token_ids = torch.tensor([10, 30, 20, 40], dtype=torch.int64)

    # sample_hidden_states is intentionally not arranged as tree-node order.
    # Lazy verification must follow target_logits_indices/bonus_logits_indices,
    # matching the full-logits path.
    row_to_token = [99, 40, 99, 77, 20, 88]
    hidden = torch.arange(len(row_to_token), dtype=torch.float32).view(-1, 1)
    projected_rows: list[int] = []

    def compute_logits(rows: torch.Tensor) -> torch.Tensor:
        row_ids = [int(x) for x in rows[:, 0].tolist()]
        projected_rows.extend(row_ids)
        return _logits_for_argmax([row_to_token[i] for i in row_ids])

    target_indices = torch.tensor([4, 0, 5, 1], dtype=torch.long)
    bonus_indices = torch.tensor([3], dtype=torch.long)
    lazy_output, lazy_gdn_state_indices = ddtree_verify_lazy_logits(
        compute_logits=compute_logits,
        sample_hidden_states=hidden,
        target_logits_indices=target_indices,
        bonus_logits_indices=bonus_indices,
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        device=torch.device("cpu"),
    )
    full_output, full_gdn_state_indices = ddtree_verify(
        logits=_logits_for_argmax(row_to_token),
        target_logits_indices=target_indices,
        bonus_logits_indices=bonus_indices,
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        batch_size=1,
        device=torch.device("cpu"),
        return_gdn_state_indices=True,
    )

    assert lazy_output.tolist() == full_output.tolist()
    assert lazy_gdn_state_indices.tolist() == full_gdn_state_indices.tolist()
    assert projected_rows == [4, 0, 1, 3]


def test_ddtree_sglang_retrieve_tables_encode_child_sibling_links() -> None:
    child_maps = [[{10: 1, 20: 3}, {30: 2}, {}, {40: 4}, {}]]

    retrieve_next_token, retrieve_next_sibling = build_retrieve_from_child_maps(
        child_maps, budget=4, device=torch.device("cpu")
    )

    assert retrieve_next_token.tolist() == [[1, 2, -1, 4, -1]]
    assert retrieve_next_sibling.tolist() == [[-1, 3, -1, -1, -1]]


def test_ddtree_sglang_greedy_verify_matches_full_verify_cuda() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    child_maps = [[{10: 1, 20: 3}, {30: 2}, {}, {40: 4}, {}]]
    draft_token_ids = torch.tensor([10, 30, 20, 40], dtype=torch.int64, device=device)
    logits = _logits_for_argmax([20, 99, 99, 40, 77]).to(device)

    sgl_output, sgl_gdn_state_indices = ddtree_verify_sglang_greedy(
        logits=logits,
        target_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
        bonus_logits_indices=torch.tensor([4], dtype=torch.long, device=device),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        batch_size=1,
        device=device,
    )
    full_output, full_gdn_state_indices = ddtree_verify(
        logits=logits,
        target_logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
        bonus_logits_indices=torch.tensor([4], dtype=torch.long, device=device),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=4,
        batch_size=1,
        device=device,
        return_gdn_state_indices=True,
    )

    assert sgl_output.cpu().tolist() == full_output.cpu().tolist()
    assert sgl_gdn_state_indices.cpu().tolist() == full_gdn_state_indices.cpu().tolist()


def test_ddtree_alt_root_chain_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_ALT_ROOT_CHAIN", "1")
    logits = torch.full((4, 8), -10.0)
    logits[:, 1] = 10.0
    logits[0, 2] = 9.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    )

    assert node_token_ids.tolist() == [1, 1, 1, 1, 2, 1]
    assert node_depths.tolist() == [1, 2, 3, 4, 1, 2]
    assert node_ranks.tolist() == [0, 0, 0, 0, 1, 0]
    assert parents == [-1, 0, 1, 2, 3, 0, 5]
    assert child_maps[0][1] == 1
    assert child_maps[0][2] == 5
    assert visibility[6, 5]
    assert not visibility[6, 1]


def test_ddtree_multi_alt_root_chain_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_ALT_ROOT_CHAINS", "2")
    logits = torch.full((4, 8), -10.0)
    logits[:, 1] = 10.0
    logits[0, 2] = 9.0
    logits[0, 3] = 8.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=10, force_chain_prefix=4)
    )

    assert node_token_ids.tolist() == [1, 1, 1, 1, 2, 1, 1, 1, 3, 1]
    assert node_depths.tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    assert node_ranks.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 2, 0]
    assert parents == [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9]
    assert child_maps[0][1] == 1
    assert child_maps[0][2] == 5
    assert child_maps[0][3] == 9
    assert visibility[8, 5]
    assert not visibility[8, 1]


def test_ddtree_suffix_branch_chain_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_SUFFIX_BRANCH_DEPTHS", "2,3")
    logits = torch.full((5, 8), -10.0)
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=12, force_chain_prefix=5)
    )

    assert node_token_ids.tolist() == [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1]
    assert node_depths.tolist() == [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5]
    assert node_ranks.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert parents == [-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 2, 10, 11]
    assert child_maps[1][2] == 6
    assert child_maps[2][2] == 10
    assert visibility[8, 6]
    assert not visibility[8, 2]


def test_ddtree_static_siblings_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_STATIC_SIBLINGS", "2")
    logits = torch.full((4, 8), -10.0)
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    )

    assert node_token_ids.tolist() == [1, 1, 1, 1, 2, 2]
    assert node_depths.tolist() == [1, 2, 3, 4, 1, 2]
    assert node_ranks.tolist() == [0, 0, 0, 0, 1, 1]
    assert parents == [-1, 0, 1, 2, 3, 0, 1]
    assert child_maps[0][2] == 5
    assert child_maps[1][2] == 6
    assert visibility[6, 1]
    assert not visibility[6, 2]


def test_ddtree_static_siblings_token_id_builder_matches_logits_builder(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DDTREE_STATIC_SIBLINGS", "2")
    monkeypatch.setenv("DDTREE_GPU_STATIC_TREE", "1")
    monkeypatch.setenv("DDTREE_SGL_VERIFY", "1")
    logits = torch.full((4, 8), -10.0)
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0

    expected = build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    top1_ids = logits.argmax(dim=-1)
    branch_top2_ids = torch.topk(logits[:2].float(), k=2, dim=-1).indices[:, 1]
    actual = build_static_sibling_tree_from_token_ids(
        top1_ids,
        branch_top2_ids,
        budget=6,
        chain_len=4,
        branch_count=2,
    )

    for expected_tensor, actual_tensor in zip(expected[:3], actual[:3]):
        assert actual_tensor.tolist() == expected_tensor.tolist()
    assert actual[3] == expected[3]
    assert actual[4] == expected[4]
    assert actual[5].tolist() == expected[5].tolist()


def test_ddtree_static_interleaved_siblings_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_STATIC_SIBLINGS", "2")
    monkeypatch.setenv("DDTREE_GPU_STATIC_TREE", "1")
    monkeypatch.setenv("DDTREE_SGL_VERIFY", "1")
    monkeypatch.setenv("DDTREE_STATIC_INTERLEAVE", "1")
    logits = torch.full((4, 8), -10.0)
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    )

    assert node_token_ids.tolist() == [1, 2, 1, 2, 1, 1]
    assert node_depths.tolist() == [1, 1, 2, 2, 3, 4]
    assert node_ranks.tolist() == [0, 1, 0, 1, 0, 0]
    assert parents == [-1, 0, 0, 1, 1, 3, 5]
    assert child_maps[0][-1] == 1
    assert child_maps[0][-2] == 2
    assert child_maps[1][-3] == 3
    assert child_maps[1][-4] == 4
    assert visibility[6, 5]
    assert not visibility[6, 2]


def test_ddtree_gpu_static_siblings_matches_sglang_verify(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("DDTREE_STATIC_SIBLINGS", "2")
    monkeypatch.setenv("DDTREE_GPU_STATIC_TREE", "1")
    monkeypatch.setenv("DDTREE_SGL_VERIFY", "1")
    logits = torch.full((4, 8), -10.0, device="cuda")
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    )

    assert node_token_ids.is_cuda
    assert node_token_ids.cpu().tolist() == [1, 1, 1, 1, 2, 2]
    assert node_depths.tolist() == [1, 2, 3, 4, 1, 2]
    assert node_ranks.tolist() == [0, 0, 0, 0, 1, 1]
    assert parents == [-1, 0, 1, 2, 3, 0, 1]
    assert visibility[6, 1]
    assert not visibility[6, 2]

    target_logits = _logits_for_argmax([2, 77, 77, 77, 77, 2, 99]).to("cuda")
    output, gdn_state_indices = ddtree_verify_sglang_greedy(
        logits=target_logits,
        target_logits_indices=torch.arange(6, dtype=torch.long, device="cuda"),
        bonus_logits_indices=torch.tensor([6], dtype=torch.long, device="cuda"),
        draft_token_ids=node_token_ids,
        child_maps=[child_maps],
        budget=6,
        batch_size=1,
        device=torch.device("cuda"),
    )

    assert output.cpu().tolist() == [[2, 2, -1, -1, -1, -1, -1]]
    assert gdn_state_indices.cpu().tolist() == [6]


def test_ddtree_static_siblings_fast_verify_matches_full_verify_cuda() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    child_maps = [[{10: 1, 20: 5}, {30: 2, 40: 6}, {50: 3}, {60: 4}, {}, {}, {}]]
    draft_token_ids = torch.tensor(
        [10, 30, 50, 60, 20, 40], dtype=torch.int64, device=device
    )
    logits = _logits_for_argmax([10, 40, 99, 99, 99, 77, 88]).to(device)

    fast_output, fast_gdn_state_indices = ddtree_verify_static_siblings_greedy(
        logits=logits,
        target_logits_indices=torch.arange(6, dtype=torch.long, device=device),
        bonus_logits_indices=torch.tensor([6], dtype=torch.long, device=device),
        draft_token_ids=draft_token_ids,
        budget=6,
        batch_size=1,
        device=device,
        chain_len=4,
        branch_count=2,
    )
    full_output, full_gdn_state_indices = ddtree_verify(
        logits=logits,
        target_logits_indices=torch.arange(6, dtype=torch.long, device=device),
        bonus_logits_indices=torch.tensor([6], dtype=torch.long, device=device),
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=6,
        batch_size=1,
        device=device,
        return_gdn_state_indices=True,
    )

    assert fast_output.cpu().tolist() == full_output.cpu().tolist()
    assert fast_gdn_state_indices.cpu().tolist() == full_gdn_state_indices.cpu().tolist()


def test_ddtree_static_siblings_lazy_logits_matches_full_verify() -> None:
    child_maps = [[{10: 1, 20: 5}, {30: 2, 40: 6}, {50: 3}, {60: 4}, {}, {}, {}]]
    draft_token_ids = torch.tensor([10, 30, 50, 60, 20, 40], dtype=torch.int64)
    row_to_token = [10, 40, 99, 99, 99, 77, 88]
    hidden = torch.arange(len(row_to_token), dtype=torch.float32).view(-1, 1)
    projected_rows: list[int] = []

    def compute_logits(rows: torch.Tensor) -> torch.Tensor:
        row_ids = [int(x) for x in rows[:, 0].tolist()]
        projected_rows.extend(row_ids)
        return _logits_for_argmax([row_to_token[i] for i in row_ids])

    target_logits_indices = torch.arange(6, dtype=torch.long)
    bonus_logits_indices = torch.tensor([6], dtype=torch.long)
    lazy_output, lazy_gdn_state_indices = ddtree_verify_static_siblings_lazy_logits(
        compute_logits=compute_logits,
        sample_hidden_states=hidden,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        draft_token_ids=draft_token_ids,
        budget=6,
        batch_size=1,
        device=torch.device("cpu"),
        chain_len=4,
        branch_count=2,
    )
    full_output, full_gdn_state_indices = ddtree_verify(
        logits=_logits_for_argmax(row_to_token),
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        draft_token_ids=draft_token_ids,
        child_maps=child_maps,
        budget=6,
        batch_size=1,
        device=torch.device("cpu"),
        return_gdn_state_indices=True,
    )

    assert lazy_output.tolist() == full_output.tolist()
    assert lazy_gdn_state_indices.tolist() == full_gdn_state_indices.tolist()
    assert projected_rows == [0, 1, 6]


def test_ddtree_multi_root_chains_fast_verify_matches_full_verify_cuda() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    # Three chains of length 3:
    # root -> 10 -> 11 -> 12
    # root -> 20 -> 21 -> 22
    # root -> 30 -> 31 -> 32
    child_maps = [[
        {10: 1, 20: 4, 30: 7},
        {11: 2},
        {12: 3},
        {},
        {21: 5},
        {22: 6},
        {},
        {31: 8},
        {32: 9},
        {},
    ]]
    draft_token_ids = torch.tensor(
        [10, 11, 12, 20, 21, 22, 30, 31, 32],
        dtype=torch.int64,
        device=device,
    )
    cases = [
        # Root rejection: emit only the target token from root logits.
        [99, 88, 88, 88, 88, 88, 88, 88, 88, 77],
        # Full accept of the first chain uses node 3 logits for the bonus.
        [10, 11, 12, 77, 88, 88, 88, 88, 88, 66],
        # Middle rejection on the second chain emits accepted tokens + reject token.
        [20, 88, 88, 88, 21, 99, 88, 88, 88, 66],
        # Full accept of the third chain uses the appended bonus row.
        [30, 88, 88, 88, 88, 88, 88, 31, 32, 77],
    ]
    target_logits_indices = torch.arange(9, dtype=torch.long, device=device)
    bonus_logits_indices = torch.tensor([9], dtype=torch.long, device=device)

    for target_tokens in cases:
        logits = _logits_for_argmax(target_tokens).to(device)

        fast_output, fast_gdn_state_indices = ddtree_verify_multi_root_chains_greedy(
            logits=logits,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            draft_token_ids=draft_token_ids,
            budget=9,
            batch_size=1,
            device=device,
            chain_len=3,
            num_chains=3,
        )
        full_output, full_gdn_state_indices = ddtree_verify(
            logits=logits,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            draft_token_ids=draft_token_ids,
            child_maps=child_maps,
            budget=9,
            batch_size=1,
            device=device,
            return_gdn_state_indices=True,
        )

        assert fast_output.cpu().tolist() == full_output.cpu().tolist()
        assert fast_gdn_state_indices.cpu().tolist() == full_gdn_state_indices.cpu().tolist()


def test_ddtree_topk_logz_builds_valid_forced_chain_tree(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_TOPK_LOGZ", "1")
    logits = torch.full((4, 16), -20.0)
    logits[:, 1] = 10.0
    logits[:, 2] = 9.0
    logits[:, 3] = 8.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=6, force_chain_prefix=4)
    )

    assert node_token_ids[:4].tolist() == [1, 1, 1, 1]
    assert node_depths[:4].tolist() == [1, 2, 3, 4]
    assert node_ranks[:4].tolist() == [0, 0, 0, 0]
    assert parents[:5] == [-1, 0, 1, 2, 3]
    assert child_maps[0][1] == 1
    assert visibility[4, 0]
    assert visibility[4, 3]


def test_ddtree_candidate_topk_auto_cap_and_override(monkeypatch) -> None:
    monkeypatch.delenv("DDTREE_CANDIDATE_TOPK", raising=False)

    assert resolve_ddtree_candidate_topk(
        budget=15, vocab_size=248320, force_chain_prefix=12
    ) == 12
    assert resolve_ddtree_candidate_topk(
        budget=15, vocab_size=248320, force_chain_prefix=0
    ) == 15

    monkeypatch.setenv("DDTREE_CANDIDATE_TOPK", "15")
    assert resolve_ddtree_candidate_topk(
        budget=15, vocab_size=248320, force_chain_prefix=12
    ) == 15

    monkeypatch.setenv("DDTREE_CANDIDATE_TOPK", "99")
    assert resolve_ddtree_candidate_topk(
        budget=15, vocab_size=8, force_chain_prefix=12
    ) == 8


def test_ddtree_fast_chain_builder_matches_generic(monkeypatch) -> None:
    for key in (
        "DDTREE_STATIC_SIBLINGS",
        "DDTREE_ALT_ROOT_CHAIN",
        "DDTREE_ROOT_SIBLINGS",
        "DDTREE_SKIP_LOGZ",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DDTREE_TOPK_LOGZ", "1")

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((15, 64), generator=generator)

    monkeypatch.setenv("DDTREE_ENABLE_FAST_CHAIN", "1")
    fast = build_ddtree_tree(logits, budget=15, force_chain_prefix=12)
    monkeypatch.delenv("DDTREE_ENABLE_FAST_CHAIN", raising=False)
    generic = build_ddtree_tree(logits, budget=15, force_chain_prefix=12)

    assert fast[0].tolist() == generic[0].tolist()
    assert fast[1].tolist() == generic[1].tolist()
    assert fast[2].tolist() == generic[2].tolist()
    assert fast[3] == generic[3]
    assert fast[4] == generic[4]
    assert torch.equal(fast[5], generic[5])


def test_ddtree_prompt_length_gate(monkeypatch) -> None:
    monkeypatch.delenv("DDTREE_MAX_PROMPT_TOKENS", raising=False)
    assert should_use_ddtree_for_prompt_lens([74])
    assert should_use_ddtree_for_prompt_lens([2058])

    monkeypatch.setenv("DDTREE_MAX_PROMPT_TOKENS", "512")
    assert should_use_ddtree_for_prompt_lens([74])
    assert not should_use_ddtree_for_prompt_lens([2058])

    monkeypatch.setenv("DDTREE_MAX_PROMPT_TOKENS", "0")
    assert should_use_ddtree_for_prompt_lens([8202])

    monkeypatch.setenv("DDTREE_MAX_PROMPT_TOKENS", "4096")
    assert should_use_ddtree_for_prompt_lens([74, 2058])
    assert not should_use_ddtree_for_prompt_lens([74, 8202])


def test_ddtree_root_siblings_topology(monkeypatch) -> None:
    monkeypatch.setenv("DDTREE_ROOT_SIBLINGS", "3")
    logits = torch.full((4, 16), -20.0)
    logits[:, 1] = 10.0
    logits[0, 2] = 9.0
    logits[0, 3] = 8.0
    logits[0, 4] = 7.0

    node_token_ids, node_depths, node_ranks, parents, child_maps, visibility = (
        build_ddtree_tree(logits, budget=7, force_chain_prefix=4)
    )

    assert node_token_ids.tolist() == [1, 1, 1, 1, 2, 3, 4]
    assert node_depths.tolist() == [1, 2, 3, 4, 1, 1, 1]
    assert node_ranks.tolist() == [0, 0, 0, 0, 1, 2, 3]
    assert parents == [-1, 0, 1, 2, 3, 0, 0, 0]
    assert child_maps[0][2] == 5
    assert child_maps[0][3] == 6
    assert child_maps[0][4] == 7
    assert visibility[7, 0]
    assert not visibility[7, 1]
