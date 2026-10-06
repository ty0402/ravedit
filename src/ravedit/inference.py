"""Single-video or CSV-batch inference, without a training dataset or Trainer."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from ltx_trainer.config import ValidationConfig
from ltx_trainer.progress import TrainingProgress
from ravedit.checkpoint import load_editor
from ravedit.media import stage_reference, verify_output
from ravedit.runner import EditingRunner


def read_samples(args) -> list[dict]:
    if args.input_csv:
        if args.prompt:
            raise ValueError("Use the prompt column with --input-csv, not --prompt")
        with args.input_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        base = args.input_csv.resolve().parent
    else:
        if not args.prompt:
            raise ValueError("--prompt is required with --source")
        rows = [{"sample_id": "edited", "source_video": str(args.source.resolve()), "prompt": args.prompt}]
        base = Path.cwd()
    seen = set()
    for row in rows:
        name = row["sample_id"]
        if not name or name in (".", "..") or Path(name).name != name or name in seen:
            raise ValueError(f"Use a unique filename-safe sample_id: {name!r}")
        seen.add(name)
        row["source_video"] = str((base / row["source_video"]).resolve())
        if not row["prompt"].strip():
            raise ValueError(f"Empty instruction: {name}")
    if not rows:
        raise ValueError("No input samples")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Override the role-aware adapter in the config")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--input-csv", type=Path)
    parser.add_argument("--prompt", help="Editing instruction, not a target-scene caption")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int, help="8k+1 frames; default: fit the common source AV duration")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--audit", action="store_true", help="Check role calls, shared RoPE, clean references, and generated AV")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model = cfg["model"]
    for key in model:
        model[key] = str((args.config.resolve().parent / model[key]).resolve())
        if not Path(model[key]).is_file():
            raise FileNotFoundError(model[key])
    checkpoint = args.checkpoint.resolve() if args.checkpoint else (args.config.resolve().parent / cfg["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    settings = dict(cfg["inference"])
    width = args.width or settings.pop("width")
    height = args.height or settings.pop("height")
    settings.pop("width", None)
    settings.pop("height", None)
    default_frames = settings.pop("frames", None)
    frames = args.frames if args.frames is not None else default_frames
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Width and height must be positive multiples of 32")
    if args.steps is not None:
        settings["inference_steps"] = args.steps
    if args.seed is not None:
        settings["seed"] = args.seed
    fps = int(settings["frame_rate"])
    if fps != settings["frame_rate"]:
        raise ValueError("Use an integer output frame rate")
    rows = read_samples(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    started = time.monotonic()
    torch.manual_seed(settings["seed"])
    samples = []
    for row in rows:
        reference = stage_reference(Path(row["source_video"]), output / "references" / row["sample_id"],
                                    width, height, fps, frames)
        row.update(reference_video=reference["video"], reference_audio=reference["audio"], frames=reference["frames"])
        samples.append(dict(prompt=row["prompt"], seed=settings["seed"], video_dims=[width, height, row["frames"]], conditions=[
            dict(type="reference", video=reference["video"], share_positions=True, include_in_output=False),
            dict(type="reference", audio=reference["audio"], share_positions=True, include_in_output=False),
        ]))
    validation = ValidationConfig(**settings, samples=samples, generate_audio=True, generate_video=True,
                                  include_reference_in_output=False, interval=None)
    (output / "inference_config.json").write_text(json.dumps({
        "model": model, "checkpoint": str(checkpoint), "validation": validation.model_dump(mode="json"),
    }, indent=2))
    # Cache text + references before putting the transformer on GPU.
    runner = EditingRunner(config=validation, **model, audit=args.audit)
    policy, checkpoint_report = load_editor(model["model_path"], checkpoint)
    (output / "checkpoint_report.json").write_text(json.dumps(checkpoint_report, indent=2))
    policy.to("cuda").eval()
    print(f"Loaded {checkpoint_report['lora_tensors']} LoRA tensors and {checkpoint_report['role_tensors']} role tensors.", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), TrainingProgress(enabled=False, total_steps=0) as progress:
        generated = runner.run(policy, step=0, output_dir=output / "render", device=torch.device("cuda"), progress=progress)
    results = []
    for index, path in generated:
        row = rows[index]
        edited = output / "videos" / (row["sample_id"] + ".mp4")
        path.rename(edited)
        stream_info = verify_output(edited, width, height, row["frames"], fps)
        results.append({"sample_id": row["sample_id"], "source_video": row["source_video"],
                        "edited_video": str(edited), "prompt": row["prompt"], "checkpoint": str(checkpoint),
                        "seed": settings["seed"], "width": width, "height": height, "frames": row["frames"], "fps": fps})
        (output / "videos" / (row["sample_id"] + ".json")).write_text(json.dumps(stream_info, indent=2))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "run_report.json").write_text(json.dumps({
        "samples": len(results), "seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoint": checkpoint_report, "audits": runner.audits,
        "generated_audio": True, "reference_panel": False,
    }, indent=2))
    print(f"Saved {len(results)} edited AV video(s) to {output / 'videos'}", flush=True)
