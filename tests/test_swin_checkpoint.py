"""Swin checkpointing must stay compatible with torch.autograd.grad."""
import ast
from pathlib import Path

import torch

from frbench.backbones.swin_v2 import BasicLayerV2


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_calls_without_use_reentrant():
    missing = []
    for path in (REPO_ROOT / "frbench").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "checkpoint"
                and isinstance(func.value, ast.Name)
                and func.value.id == "checkpoint"
            ):
                continue
            if not any(
                isinstance(keyword, ast.keyword) and keyword.arg == "use_reentrant"
                for keyword in node.keywords
            ):
                missing.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return missing


def test_all_activation_checkpoints_pass_use_reentrant():
    missing = _checkpoint_calls_without_use_reentrant()
    assert missing == [], (
        "checkpoint.checkpoint(...) must pass use_reentrant explicitly; "
        f"missing at {missing}"
    )


def test_swinv2_checkpoint_supports_autograd_grad():
    layer = BasicLayerV2(
        dim=32,
        input_resolution=(8, 8),
        depth=2,
        num_heads=4,
        window_size=4,
        use_checkpoint=True,
    )
    layer.eval()
    x = torch.randn(2, 64, 32, requires_grad=True)

    y = layer(x)
    (g,) = torch.autograd.grad(y.pow(2).mean(), x)

    assert y.shape == (2, 64, 32)
    assert g.shape == x.shape
    assert torch.isfinite(g).all()


def test_swinv2_checkpoint_matches_eager_eval_grads():
    ckpt = BasicLayerV2(
        dim=32,
        input_resolution=(8, 8),
        depth=2,
        num_heads=4,
        window_size=4,
        use_checkpoint=True,
    )
    eager = BasicLayerV2(
        dim=32,
        input_resolution=(8, 8),
        depth=2,
        num_heads=4,
        window_size=4,
        use_checkpoint=False,
    )
    eager.load_state_dict(ckpt.state_dict())
    ckpt.eval()
    eager.eval()

    x = torch.randn(2, 64, 32, requires_grad=True)
    y_ckpt = ckpt(x)
    (g_ckpt,) = torch.autograd.grad(y_ckpt.pow(2).mean(), x, retain_graph=True)

    x_eager = x.detach().clone().requires_grad_(True)
    y_eager = eager(x_eager)
    (g_eager,) = torch.autograd.grad(y_eager.pow(2).mean(), x_eager)

    assert torch.equal(y_ckpt, y_eager)
    assert torch.equal(g_ckpt, g_eager)
