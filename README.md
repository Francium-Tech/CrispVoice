# CrispVoice

**Studio-quality voice enhancement that never uploads your voice.**

Live demo with before/after audio: [crispvoice.francium.tech](https://crispvoice.francium.tech)

Take any voice recording - a phone memo, a laptop-mic capture, a noisy call -
and get back clean, full-bodied, podcast-ready audio. Runs 100% locally:
your audio never leaves your machine.

Inspired by the excellent [cleanvoice.ai](https://cleanvoice.ai/) and tools
like Adobe Podcast Enhance. They showed what is possible; this project open
sources *how* it is done, for anyone who wants the same result without
sending their voice to a server.

## What it does

"Studio quality" is two problems, and this pipeline solves both:

1. **Cleanup** - a neural denoiser removes background noise, hum, and room
   reverb.
2. **Restoration** - a generative model ([Resemble
   Enhance](https://github.com/resemble-ai/resemble-enhance), MIT) literally
   re-synthesizes the voice at 44.1 kHz, adding back the bandwidth and body a
   cheap microphone never captured. This is the "suddenly I sound like a
   broadcast host" effect.
3. **Mastering** - a broadcast-style ffmpeg chain: EQ tuned against a
   commercial reference, de-essing, gentle 2:1 compression, and EBU R128
   loudness normalization to -19 LUFS.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown,
including how the mastering EQ was fitted band-by-band against a commercial
tool's output.

## Hear it

A 14-second clip recorded with background noise, before and after:

| | |
|---|---|
| Before (raw recording) | [demo/noise.m4a](demo/noise.m4a) |
| After (CrispVoice) | [demo/crispvoice.mp3](demo/crispvoice.mp3) |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and curl. Tested on macOS
(Apple Silicon); Linux x86_64 should work. ~4 GB disk total.

```sh
git clone https://github.com/Francium-Tech/CrispVoice.git
cd CrispVoice
./setup.sh                    # one-time: local Python + PyTorch + model weights
./enhance recording.mp3       # writes recording_studio.wav
```

More options:

```sh
./enhance in.mp3 out.mp3          # output format follows the extension (wav/mp3/m4a)
./enhance in.mp3 --nfe 64         # higher quality, ~2x slower
./enhance in.mp3 --denoise-only   # cleanup without generative re-synthesis
./enhance in.mp3 --threads 2      # use even less CPU
./enhance in.mp3 --no-master      # skip the mastering chain
```

Progress, ETA, and memory usage are printed for every chunk.

## Design principles

- **Private by default.** No network calls after setup. Ever.
- **Self-contained.** The Python interpreter, PyTorch, model weights, and
  even the ffmpeg binary all live inside the project folder. Nothing is
  installed system-wide. `./uninstall.sh` removes every trace in one go.
- **Never hog the machine.** CPU-only (deliberately: on Apple Silicon the
  GPU shares unified memory with the OS, and PyTorch's MPS backend can
  exhaust it and freeze the machine - a lesson learned the hard way).
  Half the cores, low process priority, bounded memory via 10-second chunks.
  Throughput is roughly 1.4x real time on an M-series CPU: an 8-minute
  recording takes about 12 minutes.

## Credits

- [Resemble AI](https://github.com/resemble-ai/resemble-enhance) for the
  MIT-licensed enhancement model that does the heavy lifting.
- [cleanvoice.ai](https://cleanvoice.ai/) for the inspiration and the
  quality bar to chase.
- [FFmpeg](https://ffmpeg.org/) for decoding, mastering, and encoding.

## About

CrispVoice is an open source project by [Francium Tech](https://francium.tech).

## License

MIT. Model weights are downloaded separately from
[ResembleAI/resemble-enhance](https://huggingface.co/ResembleAI/resemble-enhance)
(also MIT) and are not part of this repository.
