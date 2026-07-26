#!/usr/bin/env python
"""Clarity front-end runner: ClearerVoice MossFormer2_SE_48K.

Executed by enhance.py as a subprocess using .venv-cv; never imported by the
main pipeline. Cleans the recording before re-synthesis, which measurably
improves word intelligibility of the final output (won the blind A/B).

Safety pattern (learned the hard way): give the model -6 dB input headroom,
capture its output array directly instead of trusting its 16-bit writer, and
peak-normalize before writing - otherwise peaks above full scale bake
crackles into the intermediate.

Usage: cv_runner.py in.wav out.wav   (both 48 kHz mono wav)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "5")
PROJECT_DIR = Path(__file__).resolve().parent
# checkpoints download to ./checkpoints relative to cwd on first run
ckpt_home = PROJECT_DIR / "models" / "clearvoice"
ckpt_home.mkdir(parents=True, exist_ok=True)
os.chdir(ckpt_home)

import numpy as np
import soundfile as sf
from clearvoice import ClearVoice


def main():
    in_wav, out_wav = sys.argv[1], sys.argv[2]
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    out = cv(input_path=in_wav, online_write=False)
    x = np.asarray(out, dtype=np.float32).squeeze()
    if not np.isfinite(x).all():
        sys.exit("clarity model produced non-finite samples")
    peak = np.abs(x).max()
    if peak > 0.98:
        x = x * (0.98 / peak)
    sf.write(out_wav, x, 48000, subtype="PCM_16")
    print(f"clarity: done, peak {np.abs(x).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
