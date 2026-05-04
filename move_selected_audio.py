#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

GENSVS_IDS = {1, 42, 35}
MSRBENCH_IDS = {240, 198, 78}
# EARS keys are (speaker folder, leading filename digits)
EARS_UTTERANCES = {("p102", 7), ("p105", 164), ("p104", 749)}


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def is_target_file(relative_path: Path) -> bool:
    parts = [p.lower() for p in relative_path.parts]
    if not parts:
        return False

    dataset = parts[0]
    stem = normalize_name(relative_path.stem)
    if dataset == "gensvs":
        # Enforce dataset/model/file layout to avoid duplicate nested matches (e.g. convert/embeddings).
        if len(parts) != 3:
            return False
        # Exact fileid match only: fileid_1_, fileid_35_, fileid_42_ (and extension boundary variant).
        m = re.search(r"fileid_(\d+)(?:_|$)", stem)
        return bool(m and int(m.group(1)) in GENSVS_IDS)

    if dataset == "msrbench":
        # Enforce dataset/model/file layout.
        if len(parts) != 3:
            return False
        # Support both naming styles: fileid_240_* and 240_DT0_*.
        m = re.search(r"fileid_(\d+)(?:_|$)", stem)
        if m and int(m.group(1)) in MSRBENCH_IDS:
            return True

        # Also support bare IDs like 240.flac and prefixed forms like 240_DT0_*.flac.
        m = re.search(r"^(\d+)(?:_[a-z0-9]+(?:_|$)|$)", stem)
        return bool(m and int(m.group(1)) in MSRBENCH_IDS)

    if dataset == "ears":
        # EARS match rule: speaker folder + first digits in filename, e.g. .../p102/0007_*.wav or .../p102/00007.wav
        if len(parts) < 4:
            return False
        speaker = parts[-2].replace("-", "_")
        m = re.match(r"(\d+)(?:_|$)", stem)
        if not m:
            return False
        utt_prefix = int(m.group(1))
        return (speaker, utt_prefix) in EARS_UTTERANCES

    return False


def find_matches(source_root: Path) -> list[Path]:
    matches: list[Path] = []
    for file_path in source_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in AUDIO_EXTS:
            continue

        rel = file_path.relative_to(source_root)
        if is_target_file(rel):
            matches.append(file_path)

    return sorted(matches)


def copy_files(
    files: list[Path],
    source_root: Path,
    dest_root: Path,
    dry_run: bool,
    convert_to_wav_16bit: bool,
) -> tuple[int, int, int, int]:
    copied = 0
    skipped_existing = 0
    converted = 0
    failed = 0

    for src in files:
        rel = src.relative_to(source_root)
        dst_rel = rel.with_suffix(".wav") if convert_to_wav_16bit else rel
        dst = dest_root / dst_rel

        if dst.exists():
            skipped_existing += 1
            continue

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if convert_to_wav_16bit:
                try:
                    # Re-encode to PCM 16-bit WAV while preserving channels/sample rate from input.
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(src),
                            "-vn",
                            "-c:a",
                            "pcm_s16le",
                            str(dst),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    converted += 1
                except (subprocess.CalledProcessError, FileNotFoundError):
                    failed += 1
                    continue
            else:
                shutil.copy2(str(src), str(dst))

        copied += 1

    return copied, skipped_existing, converted, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy selected GenSVS/MSRBench/EARS audio samples from static/se2svs_audio to static/audio."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("static/se2svs_audio"),
        help="Source root (default: static/se2svs_audio)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("static/audio"),
        help="Destination root (default: static/audio)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched files without copying.",
    )
    parser.add_argument(
        "--convert-to-wav-16bit",
        action="store_true",
        help="Convert matched audio to 16-bit PCM WAV in destination (requires ffmpeg).",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    dest_root = args.dest.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source folder not found: {source_root}")

    matches = find_matches(source_root)
    copied, skipped_existing, converted, failed = copy_files(
        matches,
        source_root,
        dest_root,
        args.dry_run,
        args.convert_to_wav_16bit,
    )

    print(f"Matched audio files: {len(matches)}")
    if args.dry_run:
        print("Dry run mode: no files were copied.")
    else:
        print(f"Copied files: {copied}")
        print(f"Skipped because destination already exists: {skipped_existing}")
        if args.convert_to_wav_16bit:
            print(f"Converted to 16-bit WAV: {converted}")
            print(f"Failed conversions: {failed}")
        print(f"Destination root: {dest_root}")


if __name__ == "__main__":
    main()
