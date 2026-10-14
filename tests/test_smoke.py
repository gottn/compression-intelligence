from cilab.smoke import TOLERANCE_BPB, dtype_for, verdict


def test_verdict_within_tolerance_is_ok() -> None:
    assert verdict(1.243 + TOLERANCE_BPB / 2, "enwik8_ctrl") == "ok"


def test_verdict_outside_tolerance_is_mismatch() -> None:
    assert verdict(1.30, "enwik8_ctrl") == "mismatch"


def test_verdict_without_reference() -> None:
    assert verdict(0.9, "wiki2026") == "unreferenced"


def test_dtype_cpu_is_float32() -> None:
    import torch
    assert dtype_for("cpu") is torch.float32
    assert dtype_for("cuda") is torch.float16
