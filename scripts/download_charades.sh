#!/usr/bin/env bash
# Download the Charades-Ego dataset (annotations + 480p videos) into data/.
#
# Dataset: Sigurdsson et al. (2018), "Charades-Ego: A Large-Scale Dataset of
# Paired Third and First Person Videos". Project page:
#   https://prior.allenai.org/projects/charades-ego
#
# After extraction the layout matches config.py:
#   data/CharadesEgo/CharadesEgo_v1_train.csv      (annotations, ANNOT_CSV)
#   data/CharadesEgo_v1_480/<id>.mp4               (videos, VIDEO_DIR)
#
# The video archive is large (~11 GB). Only the first-person (egocentric) videos
# are used; ~500 of them are needed for training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SDV_DATA_DIR:-$SCRIPT_DIR/../data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades"

dl() {  # url outfile
  if [ -f "$2" ]; then echo "[skip] $2 exists"; return; fi
  echo "[download] $1"
  if command -v wget >/dev/null 2>&1; then wget -c -O "$2" "$1";
  else curl -L -C - -o "$2" "$1"; fi
}

# --- annotations (~40 MB) ---
dl "$BASE/CharadesEgo.zip" "CharadesEgo.zip"
if [ ! -d "CharadesEgo" ]; then
  echo "[extract] CharadesEgo.zip"; unzip -q CharadesEgo.zip
fi

# --- 480p videos (~11 GB) ---
dl "$BASE/CharadesEgo_v1_480.tar" "CharadesEgo_v1_480.tar"
if [ ! -d "CharadesEgo_v1_480" ]; then
  echo "[extract] CharadesEgo_v1_480.tar (this takes a while)"; tar -xf CharadesEgo_v1_480.tar
fi

echo "[done] annotations: $DATA_DIR/CharadesEgo/  videos: $DATA_DIR/CharadesEgo_v1_480/"
