# Third-party code and local changes

This distribution includes an inference runtime derived from
[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2), package version 1.2.0,
copied from the working LTX-2.5 environment on 2026-10-06. The complete
upstream LTX-2.x license is retained in [LICENSE.md](LICENSE.md). Existing
file-level third-party copyright and license notices are retained.

`src/ltx_core/` contains the native transformer, schedulers, guiders, VAEs,
text encoder and their supporting runtime. `src/ltx_trainer/` contains only
the native validation/inference runner and its import dependencies; it does
not include the training driver. Some strategy/config classes are included
because the native validation config imports them.

Local changes carried over from the working editing implementation:

- `ltx_core/model/transformer/av_role_attention.py`: affine feature/sigma role bias.
- `attention.py`, `model.py`, `transformer.py`, `modality.py`,
  `transformer_args.py` in the same directory: role enablement, masks and routing.
- `ltx_trainer/config.py`, `training_strategies/flexible.py`,
  `validation_runner.py`: shared source/target positions, source-first model
  inputs, role masks, audio reference-length alignment, inference support.
- `ltx_core/model/video_vae/transformer/fallback_na/triton_na.py`: existing
  Triton constexpr compatibility fix for the diffusion-VAE fallback decoder.

These locally modified files carry a RAVEdit modification notice. The new
`src/ravedit/` code packages inference, strict adapter loading and AV media
handling without changing the learned role-attention formula.

Weights, third-party datasets and user media are not included. Their original
licenses and access requirements continue to apply. No alternate open-source
license is assigned to the included LTX runtime or its derivatives.
