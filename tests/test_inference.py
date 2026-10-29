"""Frozen inference caches must preserve predictions, masks, RNG and tensor ownership."""
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.common import Policy
from training.inference import FrozenOpponentGraphs


def test_graphs_reject_trainable_policy_and_invalid_limits():
    model = Policy(hidden=32, embedding=8)
    with pytest.raises(ValueError, match="frozen"):
        FrozenOpponentGraphs(model)
    model.eval().requires_grad_(False)
    for limit in (0, -1, 65, True):
        with pytest.raises(ValueError, match="limit"):
            FrozenOpponentGraphs(model, limit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression needs GPU")
def test_graphs_preserve_changed_inputs_masks_rng_and_owned_outputs():
    torch.manual_seed(713)
    model = Policy(hidden=32, embedding=8).cuda().eval().requires_grad_(False)
    cached = FrozenOpponentGraphs(model, max_batch=3)
    previous = None
    with torch.inference_mode():
        for n in (1, 3, 1, 2, 3, 4):
            f = torch.randn(n, tg.OBS_FLOATS, device="cuda")
            c = torch.randint(tg.N_CARDS, (n, tg.OBS_CARDS), device="cuda", dtype=torch.int32)
            m = torch.rand(n, tg.N_ACTIONS, device="cuda") > .5
            m[:, 0] = True
            rng = torch.cuda.get_rng_state().clone()
            expected = model(f, c, m)
            actual = cached(f, c, m)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            assert torch.equal(rng, torch.cuda.get_rng_state())
            assert torch.all(actual[0][~m] == -1e9)
            if previous is not None:
                assert all(torch.equal(a, b) for a, b in zip(*previous))
            previous = (actual, tuple(x.clone() for x in actual))
        assert cached.inference_stats() == {
            "captured_batches": 3, "graph_calls": 5, "eager_calls": 1, "max_batch": 3}
        with pytest.raises(ValueError, match="contract"):
            cached(f[:1], c[:1].long(), m[:1])
