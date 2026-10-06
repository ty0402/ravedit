"""Load the complete rank-128 LoRA + learned role-attention adapter."""
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file

from ltx_trainer.model_loader import load_transformer

ROLE_VERSION = "affine_query_sigma_v1"
TARGET_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0",
    "ff.net.0.proj", "ff.net.2", "audio_ff.net.0.proj", "audio_ff.net.2",
]


def attach_adapter(base, checkpoint: Path):
    """Strict loading: a plain LoRA or a partial role checkpoint is NOT accepted."""
    with safe_open(str(checkpoint), framework="pt") as saved:
        metadata = saved.metadata() or {}
        if metadata.get("role_aware_av_attention") != ROLE_VERSION:
            raise ValueError(f"Expected a role-aware {ROLE_VERSION} checkpoint: {checkpoint}")
    base.requires_grad_(False)
    base.enable_role_aware_av_attention()
    policy = get_peft_model(base, LoraConfig(
        r=128, lora_alpha=128, lora_dropout=0.0,
        target_modules=TARGET_MODULES, modules_to_save=["av_role_bias"],
    ))
    # Preserve FP32 NFT adapter weights; do not call policy.bfloat16() afterwards.
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    state = {k.removeprefix("diffusion_model."): v for k, v in load_file(str(checkpoint)).items()}
    expected = get_peft_model_state_dict(policy.get_base_model())
    if state.keys() != expected.keys():
        raise ValueError(
            f"Adapter key mismatch: missing={sorted(expected.keys()-state.keys())}, "
            f"unexpected={sorted(state.keys()-expected.keys())}"
        )
    for key, value in state.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    # PEFT rewrites modules_to_save keys in-place; keep canonical saved keys for the audit.
    set_peft_model_state_dict(policy.get_base_model(), dict(state))
    # Check actual loaded values, not only the checkpoint's metadata.
    loaded = get_peft_model_state_dict(policy.get_base_model())
    for key, value in state.items():
        if not torch.equal(loaded[key].cpu(), value.to(loaded[key].dtype)):
            raise RuntimeError(f"Checkpoint tensor did not load exactly: {key}")
    role_keys = [k for k in state if ".av_role_bias." in k]
    report = {
        "checkpoint": str(checkpoint.resolve()), "role_version": ROLE_VERSION,
        "role_tensors": len(role_keys),
        "lora_tensors": sum(".lora_" in k for k in state),
        "role_modules": len(role_keys) // 3,
        "loaded_values_verified": True,
    }
    return policy.requires_grad_(False).eval(), report


def load_editor(model_path: str, checkpoint: Path):
    base = load_transformer(model_path, device="cpu", dtype=torch.bfloat16)
    return attach_adapter(base, checkpoint)
