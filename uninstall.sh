#!/bin/sh
# Nuke button: removes everything this project installed, in one go.
# Nothing was installed system-wide; it all lives in this folder plus
# the uv download cache (shared wheel cache in ~/.cache/uv).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "This will delete the entire project folder:"
echo "  $DIR"
echo "(Python runtime, venv, model weights, scripts, and any audio files in it)"
printf "Continue? [y/N] "
read -r answer
case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "aborted"; exit 1 ;;
esac

# Drop the cached wheels uv downloaded for this project (torch etc., ~1 GB)
if command -v uv >/dev/null 2>&1; then
    uv cache clean torch torchaudio torchvision resemble-enhance librosa \
        matplotlib pandas imageio-ffmpeg numpy scipy 2>/dev/null || true
fi

cd /
rm -rf "$DIR"
echo "done - everything removed."
