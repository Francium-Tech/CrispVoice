#!/usr/bin/env python
"""GPU (Apple MPS) model runner. Executed by enhance.py as an isolated
subprocess using .venv-gpu (torch 2.7); never imported by the main pipeline.

Safety model: MPS shares unified memory with macOS and once froze this class
of machine. Two hard caps below make over-allocation raise a clean Python
error instead; combined with subprocess isolation, the worst case is "this
process dies and the caller falls back to CPU."

Correctness model: Metal silently corrupts Conv1d results beyond 65535 output
channels (the vocoder's kernel predictor has 221184). ChunkedConv1d splits
those convs; verified float-precision identical to CPU.

Usage: gpu_runner.py in.wav out.wav NFE LAMBD TAU
"""

import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.4"

import functools
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch>=2.6 defaults weights_only=True, which rejects the old checkpoint
torch.load = functools.partial(torch.load, weights_only=False)
# quantile() rejects half precision under autocast
_quantile = torch.Tensor.quantile
torch.Tensor.quantile = lambda self, *a, **k: _quantile(self.float(), *a, **k)
torch.set_num_threads(4)

from resemble_enhance.enhancer.inference import load_enhancer
from resemble_enhance.inference import (
    inference_chunk,
    merge_chunks,
    remove_weight_norm_recursively,
)
from torchaudio.functional import resample

RUN_DIR = Path(__file__).resolve().parent / "models" / "enhancer_stage2"


class ChunkedConv1d(nn.Module):
    def __init__(self, conv, max_out=32768):
        super().__init__()
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        n = math.ceil(conv.out_channels / max_out)
        self.weights = nn.ParameterList(nn.Parameter(w) for w in conv.weight.chunk(n, 0))
        self.biases = (nn.ParameterList(nn.Parameter(b) for b in conv.bias.chunk(n, 0))
                       if conv.bias is not None else None)

    def forward(self, x):
        outs = []
        for i, w in enumerate(self.weights):
            b = self.biases[i] if self.biases is not None else None
            outs.append(F.conv1d(x, w, b, self.stride, self.padding, self.dilation))
        return torch.cat(outs, dim=1)


def patch_oversized_convs(model):
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv1d) and child.out_channels > 65535:
                setattr(mod, name, ChunkedConv1d(child))


def main():
    in_wav, out_wav, nfe, lambd, tau = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    if not torch.backends.mps.is_available():
        sys.exit("mps not available")

    audio, sr = sf.read(in_wav, dtype="float32")
    dwav = torch.from_numpy(audio)

    print("gpu: loading model ...", flush=True)
    enh = load_enhancer(RUN_DIR, "mps")
    remove_weight_norm_recursively(enh)
    patch_oversized_convs(enh)
    enh.to("mps")
    enh.configurate_(nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)

    hp = enh.hp
    dwav = resample(dwav, orig_freq=sr, new_freq=hp.wav_rate,
                    lowpass_filter_width=64, rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    sr = hp.wav_rate
    chunk_length = int(sr * 10.0)
    overlap_length = int(sr * 1.0)
    hop_length = chunk_length - overlap_length
    starts = list(range(0, dwav.shape[-1], hop_length))

    chunks = []
    t0 = time.time()
    # CRISPVOICE_GPU_PRECISION=fp32 disables the fp16 fast path
    import contextlib
    if os.environ.get("CRISPVOICE_GPU_PRECISION", "fp16") == "fp32":
        amp = contextlib.nullcontext()
        print("gpu: running at fp32", flush=True)
    else:
        amp = torch.autocast("mps", dtype=torch.float16)
    with amp:
        for i, start in enumerate(starts, 1):
            chunks.append(inference_chunk(enh, dwav[start:start + chunk_length], sr, "mps").float())
            torch.mps.empty_cache()
            elapsed = time.time() - t0
            eta = elapsed / i * (len(starts) - i)
            print(f"gpu: [chunk {i}/{len(starts)}] {i / len(starts) * 100:3.0f}% | "
                  f"elapsed {int(elapsed)}s | eta {int(eta)}s | "
                  f"mps mem {torch.mps.current_allocated_memory() / 1e9:.1f} GB", flush=True)

    hwav = merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])
    out = hwav.numpy()
    if not np.isfinite(out).all():
        sys.exit("gpu produced non-finite samples")
    sf.write(out_wav, out, sr)
    print(f"gpu: done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
