# Inference verification

Date: 2026-10-06. Hardware: NVIDIA H200 (143,771 MiB reported device memory).
Runtime: Python 3.11, PyTorch 2.8.0+cu128, Transformers 5.14.1, PEFT 0.20.0.
The existing model files were reused; no model downloads or parameter training
were performed for this release.

## Automated checks

`python -m unittest discover -s tests -v`: **6 tests passed**.

1. Role attention agrees with the equivalent explicit dense bias in FP64.
2. Source-first model inputs carry the correct reference mask and shared RoPE.
3. Zero initialization and explicit sigma-dependent bias behave correctly.
4. Frame selection uses the common AV duration and the `8k+1` constraint.
5. Complete LoRA/role checkpoint reload preserves a small real LTX model's
   AV predictions exactly; a partial role checkpoint is rejected.
6. CSV parsing preserves quoted instructions and keeps different edits of the
   same source as distinct samples.

The actual PEFT loader rewrites `modules_to_save` keys in its input dictionary.
The release passes a shallow copy to it and compares loaded values using the
original saved keys; the checkpoint round-trip test covers this behavior.

## Packaging checks

- The wheel builds successfully with `pip wheel --no-build-isolation --no-deps .`.
- The wheel includes all four Gemma system-prompt assets and license notices.
- It installs into a temporary virtual environment and provides `ravedit --help`.
- All 131 loaded RAVEdit/LTX runtime modules resolve inside that installed wheel,
  not an external LTX source tree. The temporary environment inherited existing
  third-party dependencies; this is not a fresh network dependency-resolution test.
- The 199 vendored runtime/resource files match the working LTX tree, apart
  from added modification notices. Original source files were not changed.

## Real checkpoint runs

Input: JAVEditBench source `15f317acff000801944177f056d42b9e.mp4`.
Instruction: replace the dialogue with "This is the truth.", adjust lip sync,
and preserve all other visual elements.

Both checks use seed 42, 121 frames, 25 fps, 30 inference steps, CFG 3/7,
STG 1/1, modality guidance 3/3, rescale 0.7 and STG block 28.

| Checkpoint | Canvas | Result | End-to-end time | Peak allocated CUDA memory |
|---|---|---|---|---|
| Role-SFT step18836 | 768×448 | Passed | 222.65 s | 50.73 GiB |
| Role-SFT → NFT round024 (25 rounds) | 1280×736 | Passed | 561.48 s | 67.91 GiB |

The SFT checkpoint belongs to `av-role-attention-20260911`. The NFT checkpoint
belongs to `av-role-nft-from-sft18836-20260914`; it retains the learned SFT role
parameters. It is **not** the original-architecture NFT checkpoint.

The completed SFT run verified:

- 2,688 LoRA tensors and 288 role tensors, with exact loaded values.
- 96 active role modules, each called 90 times across 120 raw-model forwards.
- Reference latents unchanged, reference timestep zero, shared source/target RoPE.
- Video target and reference shapes `[1, 5376, 128]` each.
- Audio target and reference shapes `[1, 121, 128]` each.
- Both target modalities changed during denoising; output audio is generated.
- H.264 video + 48 kHz stereo AAC, 121 frames; full FFmpeg decode passed.
- Video duration 4.840 s, audio duration 4.810 s (native audio-latent rounding).
- Visual inspection of an output frame: single edited canvas, no reference panel.

The completed high-resolution NFT run passed the same checks: 2,688 LoRA
tensors, 288 role tensors, 96 active role modules (90 calls each), 120 raw-model
forwards, unchanged references and updated target AV latents. Its video target
and reference shapes are `[1, 14720, 128]` each; audio shapes are
`[1, 121, 128]` each. Output is 1280×736 H.264 with 48 kHz stereo AAC, 121 frames,
4.840 s video / 4.810 s audio, and passes full FFmpeg decoding. A decoded
frame was inspected: there is no reference panel or unexpected layout.

An additional audio-content sanity check used local Whisper large-v3 with
English transcription, temperature 0, and no reference-text hint. Both edited
outputs transcribed as **"This is the truth."**, unlike the source dialogue.
This is a single-example sanity check, not a benchmark WER evaluation.

Local artifacts are deliberately excluded from Git:

```text
outputs/verified-role-sft18836-fast/
outputs/verified-role-nft25-full/
```

Each completed directory contains `results.csv`, `checkpoint_report.json`,
`run_report.json`, the actual edited MP4 and its stream/decode verification.
`configs/inference.local.yaml` points to existing local weights and is also
excluded from Git. Public configuration uses portable relative paths.

These are runtime/correctness checks on one input, not a new benchmark score
or a guarantee that every editing instruction is followed perfectly.
