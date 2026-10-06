"""Reference staging on a shared AV clock; original files are never modified."""
import json
import math
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


def aligned_frames(video_duration: float, audio_duration: float, fps: int, requested: int | None) -> int:
    available = math.floor(min(video_duration, audio_duration) * fps + 1e-6)
    frames = 1 + ((available - 1) // 8) * 8 if requested is None else requested
    if frames < 9 or (frames - 1) % 8 or frames > available:
        raise ValueError(f"Need 8k+1 frames (at least 9), within {available} available frames; got {frames}")
    return frames


def stage_reference(source: Path, folder: Path, width: int, height: int,
                    fps: int, frames: int | None) -> dict:
    metadata = probe(source)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)
    if audio is None:
        raise ValueError(f"Joint AV editing requires a source audio track: {source}")
    if abs(float(video.get("start_time", 0)) - float(audio.get("start_time", 0))) > 1 / fps:
        raise ValueError("Source stream start times differ by more than one frame; align the source first.")
    frames = aligned_frames(
        float(video.get("duration", metadata["format"]["duration"])),
        float(audio.get("duration", metadata["format"]["duration"])), fps, frames,
    )
    folder.mkdir(parents=True, exist_ok=False)
    video_path, audio_path = folder / "reference.mp4", folder / "reference.wav"
    filters = (
        f"fps={fps}:round=near,"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-threads", "1", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", filters, "-frames:v", str(frames),
        "-c:v", "libx264", "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", str(video_path),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-threads", "1", "-i", str(source),
        "-map", "0:a:0", "-vn", "-t", str(frames / fps),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(audio_path),
    ], check=True)
    staged = probe(video_path)["streams"][0]
    if int(staged["nb_frames"]) != frames:
        raise ValueError("Decoded source is shorter than its duration metadata; refusing temporal padding.")
    return {"video": str(video_path), "audio": str(audio_path), "frames": frames}


def verify_output(path: Path, width: int, height: int, frames: int, fps: int) -> dict:
    metadata = probe(path)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    if (video["width"], video["height"], int(video["nb_frames"])) != (width, height, frames):
        raise RuntimeError("Unexpected output geometry, frame count, or reference panel")
    if video["avg_frame_rate"] != f"{fps}/1" or abs(float(video["duration"]) - float(audio["duration"])) > 0.15:
        raise RuntimeError("Output AV timing mismatch")
    subprocess.run([
        "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
    ], check=True)
    return {"width": width, "height": height, "frames": frames, "fps": fps,
            "video_codec": video["codec_name"], "audio_codec": audio["codec_name"],
            "audio_sample_rate": int(audio["sample_rate"]), "audio_channels": audio["channels"],
            "video_duration": float(video["duration"]), "audio_duration": float(audio["duration"]),
            "full_decode_passed": True}
