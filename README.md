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

1. **Clarity front-end** - a speech-enhancement transformer
   ([ClearerVoice](https://github.com/modelscope/ClearerVoice-Studio)
   MossFormer2) pre-cleans the recording. Feeding the re-synthesis a clean
   input measurably improves word intelligibility of the final output.
2. **Restoration** - a generative model ([Resemble
   Enhance](https://github.com/resemble-ai/resemble-enhance), MIT)
   re-synthesizes the voice at 44.1 kHz, adding back the bandwidth and body a
   cheap microphone never captured.
3. **Texture blend** - 25% of a
   [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)-denoised copy is
   mixed back in. Re-synthesis alone sounds subtly synthetic; the blend
   restores natural transients, breaths, and room decay. This stage came out
   of blind A/B testing against a commercial reference and moved the quality
   from "processed" to "studio."
4. **Mastering** - a broadcast-style ffmpeg chain: chest warmth, articulation
   and air EQ, de-essing, gentle 2:1 compression, and EBU R128 loudness
   normalization to -19 LUFS.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown,
including how the mastering EQ was fitted band-by-band against a commercial
tool's output.

## Hear it

Two short clips, before and after (or use the players on the
[demo page](https://crispvoice.francium.tech)):

| Voice | Before | After |
|---|---|---|
| Interview, room noise | [demo/noise.m4a](demo/noise.m4a) | [demo/crispvoice.mp3](demo/crispvoice.mp3) |
| Broadcast clip, compressed | [demo/woman.m4a](demo/woman.m4a) | [demo/woman_crispvoice.mp3](demo/woman_crispvoice.mp3) |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and curl. Tested on macOS
(Apple Silicon); Linux x86_64 should work. ~9 GB disk total (three model
runtimes; every byte lives inside this folder and uninstalls with it).

```sh
git clone https://github.com/Francium-Tech/CrispVoice.git
cd CrispVoice
./setup.sh                    # one-time: local Python + PyTorch + model weights
./enhance recording.mp3       # writes recording_studio.wav
```

More options:

```sh
./enhance in.mp3 out.mp3          # output format follows the extension (wav/mp3/m4a)
./enhance in.mp3 --preview 30     # quick sample: process only the first 30 seconds
./enhance in.mp3 --nfe 32         # ~2x faster, slightly lower quality (default 64)
./enhance in.mp3 --blend 0.4      # more natural texture (default 0.25)
./enhance in.mp3 --preset natural # lighter compression, more dynamics
./enhance in.mp3 --device cpu     # skip the Apple GPU fast path (auto by default)
./enhance in.mp3 --no-clarity     # skip the ClearerVoice pre-clean stage
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
  On Apple Silicon the heavy model stage runs on the GPU by default - in an
  isolated, memory-capped subprocess that falls back to CPU on any failure -
  bringing an 8-minute recording to roughly 5 minutes. Pure CPU
  (`--device cpu`) is roughly 2.5x real time (~20 minutes for the same file).

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
