"""CPU regression checks; no model downloads or training required."""
import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend

from ltx_core.model.transformer.attention import PytorchAttention
from ltx_core.model.transformer.av_role_attention import AVRoleBias, role_biased_attention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import TransformerOpsConfig
from ltx_trainer.validation_runner import reorder_modality_reference_first, restore_prediction_target_first
from ravedit.checkpoint import ROLE_VERSION, TARGET_MODULES, attach_adapter
from ravedit.inference import read_samples
from ravedit.media import aligned_frames


class InferenceTests(unittest.TestCase):
    def test_csv_relative_paths_and_repeated_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inputs.csv"
            prompt = 'Replace the dialogue with: "This is the truth."'
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "source_video", "prompt"])
                writer.writerow(["first", "clip.mp4", prompt])
                writer.writerow(["second", "clip.mp4", "Change the background to a beach."])
            rows = read_samples(SimpleNamespace(input_csv=path, prompt=None))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["source_video"], str(Path(folder) / "clip.mp4"))
            self.assertEqual(rows[0]["source_video"], rows[1]["source_video"])
            self.assertEqual(rows[0]["prompt"], prompt)
            self.assertNotEqual(rows[0]["sample_id"], rows[1]["sample_id"])

    def test_dense_role_attention_equivalence(self):
        torch.manual_seed(10)
        q, k, v = [torch.randn(2, n, 16, dtype=torch.float64) for n in (5, 7, 7)]
        bias = torch.randn(2, 5, 2, dtype=torch.float64)
        qm = torch.tensor([[True, False, True, False, False]]).expand(2, -1)
        km = torch.tensor([[False, True, True, False, True, False, False]]).expand(2, -1)
        attention = PytorchAttention([SDPBackend.MATH])
        actual = role_biased_attention(q, k, v, 2, bias, qm, km, attention)
        dense = bias.transpose(1, 2)[..., None] * (~qm)[:, None, :, None] * km[:, None, None, :]
        torch.testing.assert_close(actual, attention(q, k, v, 2, dense), atol=1e-12, rtol=1e-12)

    def test_explicit_roles_and_shared_positions(self):
        target = torch.randn(1, 3, 8)
        source = torch.randn_like(target)
        positions = torch.randn(1, 3, 6, 2)
        x = Modality(latent=torch.cat([target, source], 1), sigma=torch.tensor([.5]),
                     timesteps=torch.tensor([[[.5], [.5], [.5], [0.], [0.], [0.]]]),
                     positions=positions, context=torch.randn(1, 2, 8))
        actual = reorder_modality_reference_first(x, 3)
        self.assertTrue(torch.equal(actual.latent[:, :3], source))
        self.assertTrue(torch.equal(actual.positions[:, :, :3], actual.positions[:, :, 3:]))
        self.assertTrue(actual.reference_mask[:, :3].all())
        self.assertFalse(actual.reference_mask[:, 3:].any())
        self.assertTrue(torch.equal(restore_prediction_target_first(actual.latent, 3), x.latent))

    def test_sigma_bias_and_zero_initialization(self):
        role = AVRoleBias(8, 2, device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(1, 3, 8)
        self.assertEqual(torch.count_nonzero(role(x, torch.tensor([.7]))).item(), 0)
        with torch.no_grad():
            role.sigma_weight.fill_(2)
        torch.testing.assert_close(role(x, torch.tensor([.7])), torch.full((1, 3, 2), 1.4))

    def test_frame_alignment(self):
        self.assertEqual(aligned_frames(4.84, 4.864, 25, None), 121)
        self.assertEqual(aligned_frames(8., 7.9, 25, None), 193)
        with self.assertRaises(ValueError):
            aligned_frames(4., 4., 25, 121)
        with self.assertRaises(ValueError):
            aligned_frames(4.84, 4.84, 25, 120)

    def test_complete_checkpoint_roundtrip(self):
        attention = PytorchAttention([SDPBackend.MATH])
        base = LTXModel(num_attention_heads=2, audio_num_attention_heads=2,
                        attention_head_dim=32, audio_attention_head_dim=32,
                        cross_attention_dim=64, audio_cross_attention_dim=64, num_layers=2,
                        cross_attention_adaln=True, ff_bias=False, audio_ff_bias=False,
                        apply_gated_attention=True,
                        ops=TransformerOpsConfig.from_functions(attention=attention, masked_attention=attention))
        torch.manual_seed(11)
        with torch.no_grad():
            for p in base.parameters():
                p.normal_(0, .1)
        reloaded_base = copy.deepcopy(base)
        broken_base = copy.deepcopy(base)
        base.enable_role_aware_av_attention()
        policy = get_peft_model(base, LoraConfig(r=128, lora_alpha=128, lora_dropout=0,
                                               target_modules=TARGET_MODULES, modules_to_save=["av_role_bias"]))
        with torch.no_grad():
            for p in policy.parameters():
                if p.requires_grad:
                    p.normal_(0, .01)
        state = {"diffusion_model." + k: v.contiguous() for k, v in get_peft_model_state_dict(policy.get_base_model()).items()}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.safetensors"
            save_file(state, str(checkpoint), metadata={"role_aware_av_attention": ROLE_VERSION})
            loaded, report = attach_adapter(reloaded_base, checkpoint)
            self.assertEqual(report["role_tensors"], 12)
            inputs = {}
            for name, n, axes in (("video", 4, 3), ("audio", 3, 1)):
                position = torch.rand(1, axes, n, 2)
                inputs[name] = Modality(latent=torch.randn(1, 2*n, 128), sigma=torch.tensor([.4]),
                    timesteps=torch.cat([torch.zeros(1, n), torch.full((1, n), .4)], 1),
                    positions=torch.cat([position, position], 2), context=torch.randn(1, 5, 64),
                    reference_mask=torch.arange(2*n)[None, :] < n)
            with torch.no_grad():
                first = policy.eval()(**inputs, perturbations=None)
                second = loaded(**inputs, perturbations=None)
            for a, b in zip(first, second, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            del state[next(k for k in state if ".av_role_bias." in k)]
            save_file(state, str(checkpoint), metadata={"role_aware_av_attention": ROLE_VERSION})
            with self.assertRaisesRegex(ValueError, "Adapter key mismatch"):
                attach_adapter(broken_base, checkpoint)


if __name__ == "__main__":
    unittest.main()
