"""Native joint-AV denoising, with optional reference/role correctness checks."""
import torch

from ltx_core.model.transformer.av_role_attention import AVRoleBias
from ltx_trainer.validation_runner import ValidationRunner


class EditingRunner(ValidationRunner):
    def __init__(self, *args, audit=False, **kwargs):
        self.audit_enabled = audit
        self.audits = []
        super().__init__(*args, **kwargs)

    def _run_denoising(self, **kwargs):
        if not self.audit_enabled:
            return super()._run_denoising(**kwargs)
        initial = {name: kwargs[name + "_state"].latent.detach().clone() for name in ("video", "audio")}
        counts = {name: kwargs[name + "_reference_count"] for name in initial}
        calls = 0
        role_calls = {}

        def check_forward(module, args, inputs):
            nonlocal calls
            for i, name in enumerate(("video", "audio")):
                value = args[i] if len(args) > i else inputs[name]
                n = counts[name]
                assert n > 0 and value.latent.shape[1] == 2 * n
                assert not kwargs[name + "_frozen"]
                assert value.reference_mask[:, :n].all() and not value.reference_mask[:, n:].any()
                assert torch.count_nonzero(value.timesteps[:, :n]) == 0
                assert (value.timesteps[:, n:] > 0).all()
                assert torch.equal(value.latent[:, :n], initial[name][:, n:])
                assert torch.equal(value.positions[:, :, :n], value.positions[:, :, n:])
            calls += 1

        def role_hook(name):
            def record(module, inputs, output):
                role_calls[name] = role_calls.get(name, 0) + 1
            return record

        transformer = kwargs["transformer"]
        handles = [transformer.register_forward_pre_hook(check_forward, with_kwargs=True)]
        for name, module in transformer.named_modules():
            if isinstance(module, AVRoleBias) and ".original_module" not in name:
                role_calls[name] = 0
                handles.append(module.register_forward_hook(role_hook(name)))
        try:
            states = super()._run_denoising(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        assert calls > 0 and role_calls and all(role_calls.values())
        report = {"forward_calls": calls, "role_modules_called": len(role_calls),
                  "role_call_counts": role_calls, "reference_unchanged": True,
                  "shared_rope_verified": True, "source_timestep_zero": True}
        for name, state in zip(("video", "audio"), states, strict=True):
            n = counts[name]
            assert torch.equal(state.latent[:, n:], initial[name][:, n:])
            change = (state.latent[:, :n].float() - initial[name][:, :n].float()).abs().max().item()
            assert change > 0
            report[name] = {"target_shape": list(state.latent[:, :n].shape),
                            "reference_shape": list(state.latent[:, n:].shape),
                            "target_max_change": change}
        self.audits.append(report)
        return states
