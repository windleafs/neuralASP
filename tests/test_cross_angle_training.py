import torch

from train_phase_screen_cross_angle import split_context_target


def test_cross_angle_split_is_disjoint_and_within_train_partition():
    train_idx = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16])
    torch.manual_seed(7)

    for _ in range(20):
        ctx_pos, tgt_pos, ctx_idx, tgt_idx = split_context_target(
            train_idx, min_context=3, target_angles=2)

        assert len(tgt_pos) == 2
        assert len(ctx_pos) >= 3
        assert set(ctx_pos.tolist()).isdisjoint(set(tgt_pos.tolist()))
        assert set(ctx_idx.tolist()).issubset(set(train_idx.tolist()))
        assert set(tgt_idx.tolist()).issubset(set(train_idx.tolist()))
        assert torch.equal(ctx_idx, train_idx[ctx_pos])
        assert torch.equal(tgt_idx, train_idx[tgt_pos])


def test_cross_angle_split_rejects_impossible_partition():
    train_idx = torch.arange(4)
    try:
        split_context_target(train_idx, min_context=3, target_angles=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected impossible split to raise ValueError")
