#!/usr/bin/env python
"""Studio-quality voice enhancement, fully local and resource-capped.

Pipeline: decode (ffmpeg) -> Resemble Enhance (neural denoise + generative
restoration to 44.1 kHz) -> mastering chain (high-pass, de-esser, compressor,
EBU R128 loudness normalization to podcast standard) -> encode.

Designed to leave the machine usable while it runs: CPU-only by default,
capped thread count, small processing chunks, progress + ETA on every chunk.

Usage:
    ./enhance input.mp3                  # writes input_studio.wav
    ./enhance input.mp3 out.mp3          # output format follows extension
    ./enhance input.mp3 --denoise-only   # skip generative stage (faster, subtler)
    ./enhance input.mp3 --nfe 64         # more diffusion steps = better, slower
    ./enhance input.mp3 --threads 2      # even gentler on the machine
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RUN_DIR = PROJECT_DIR / "models" / "enhancer_stage2"

# Tuned against a CleanVoice reference (2026-07-25): tame lows, lift the
# 600-2500 Hz presence core, cut 3.5k harshness, roll off synthesis hiss
# above 13 kHz, compress gently (target LRA ~4-5), master to -19 LUFS.
MASTER_FILTERS = (
    "highpass=f=70,"
    "bass=g=-3.5:f=250,"
    "equalizer=f=1300:t=q:w=0.6:g=3.5,"
    "equalizer=f=3500:t=q:w=1.0:g=-3.5,"
    "treble=g=-5:f=8000,"
    "lowpass=f=13000:p=2,lowpass=f=13000:p=2,"
    "deesser,"
    "acompressor=threshold=-22dB:ratio=2:attack=10:release=250,"
    "loudnorm=I=-19:TP=-2:LRA=7,"
    "aresample=44100"  # loudnorm silently upsamples to 192 kHz
)


def parse_args():
    cpu_count = os.cpu_count() or 4
    parser = argparse.ArgumentParser(description="Enhance a voice recording to studio quality (local, private).")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--denoise-only", action="store_true", help="neural denoise without generative restoration")
    parser.add_argument("--no-master", action="store_true", help="skip EQ/compression/loudness mastering")
    parser.add_argument("--threads", type=int, default=max(1, cpu_count // 2),
                        help=f"CPU threads for the model (default {max(1, cpu_count // 2)} of {cpu_count}, "
                             "leaving the rest for the OS)")
    parser.add_argument("--chunk-seconds", type=float, default=10.0,
                        help="audio chunk size; smaller = less RAM, default 10")
    parser.add_argument("--nfe", type=int, default=32, help="diffusion solver steps (quality vs speed), default 32")
    parser.add_argument("--lambd", type=float, default=0.9, help="denoise strength 0..1, default 0.9")
    parser.add_argument("--tau", type=float, default=0.5, help="prior temperature 0..1, default 0.5")
    return parser.parse_args()


ARGS = parse_args()

# Cap math-library threads BEFORE torch/numpy get imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(ARGS.threads)
# CPU only, permanently: on Apple Silicon the GPU (MPS) shares unified memory
# with the OS; PyTorch can exhaust it and hard-freeze the machine, even with
# watermark caps. Do not add a GPU path back.
ARGS.device = "cpu"


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args):
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True)


def decode_to_wav(src: Path, dst: Path):
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", "44100", "-c:a", "pcm_f32le", str(dst)])


def encode_output(src: Path, dst: Path, master: bool):
    args = ["-i", str(src)]
    if master:
        args += ["-af", MASTER_FILTERS]
    if dst.suffix.lower() == ".mp3":
        args += ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif dst.suffix.lower() in (".m4a", ".aac"):
        args += ["-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-c:a", "pcm_s16le"]
    run_ffmpeg([*args, str(dst)])


def rss_gb():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def fmt_secs(s):
    return f"{int(s) // 60}m{int(s) % 60:02d}s"


def run_model(dwav, sr, args):
    """Chunked inference with progress + ETA, keeping memory bounded."""
    import torch
    from resemble_enhance.enhancer.inference import load_enhancer
    from resemble_enhance.inference import (
        inference_chunk,
        merge_chunks,
        remove_weight_norm_recursively,
    )
    from torchaudio.functional import resample

    torch.set_num_threads(args.threads)

    print(f"loading model (device={args.device}, threads={args.threads}) ...", flush=True)
    t0 = time.time()
    enhancer = load_enhancer(RUN_DIR, args.device)
    if args.denoise_only:
        model = enhancer.denoiser
    else:
        enhancer.configurate_(nfe=args.nfe, solver="midpoint", lambd=args.lambd, tau=args.tau)
        model = enhancer
    remove_weight_norm_recursively(model)
    print(f"model loaded in {time.time() - t0:.1f}s | rss {rss_gb():.1f} GB", flush=True)

    hp = model.hp
    dwav = resample(dwav, orig_freq=sr, new_freq=hp.wav_rate,
                    lowpass_filter_width=64, rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    sr = hp.wav_rate

    chunk_length = int(sr * args.chunk_seconds)
    overlap_length = int(sr * 1.0)
    hop_length = chunk_length - overlap_length
    starts = list(range(0, dwav.shape[-1], hop_length))
    total = len(starts)

    chunks = []
    t0 = time.time()
    for i, start in enumerate(starts, 1):
        chunks.append(inference_chunk(model, dwav[start : start + chunk_length], sr, args.device))
        gc.collect()
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(f"[chunk {i}/{total}] {i / total * 100:3.0f}% | elapsed {fmt_secs(elapsed)} | "
              f"eta {fmt_secs(eta)} | rss {rss_gb():.1f} GB", flush=True)

    hwav = merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])
    return hwav.cpu(), sr


def main():
    args = ARGS
    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")
    out = args.output or args.input.with_name(args.input.stem + "_studio.wav")

    import soundfile as sf
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        decoded = tmp / "decoded.wav"
        decode_to_wav(args.input, decoded)
        audio, sr = sf.read(decoded, dtype="float32")
        dwav = torch.from_numpy(audio)
        print(f"input: {args.input.name} ({len(audio) / sr:.1f}s @ {sr} Hz)", flush=True)

        hwav, new_sr = run_model(dwav, sr, args)

        enhanced = tmp / "enhanced.wav"
        sf.write(enhanced, hwav.numpy(), new_sr)
        encode_output(enhanced, out, master=not args.no_master)

    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
