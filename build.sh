#!/usr/bin/env bash
# =============================================================================
# build.sh — Render.com build hook
# Downloads ML artefacts from Hugging Face Hub if they are missing.
# This runs ONCE during the build phase, before the app starts.
# =============================================================================
set -e

echo "========================================"
echo " OFF SIDE — Render Build Hook"
echo "========================================"

# Install Python dependencies
pip install -r requirements.txt

# ── Download model artefacts from Hugging Face Hub ──────────────────────────
# Files are stored at: https://huggingface.co/datasets/RishabhRana37/offside-artefacts
#
# To upload your artefacts, run once locally:
#   pip install huggingface-hub
#   huggingface-cli login
#   python deploy/upload_artefacts.py
#
HF_REPO="RishabhRana37/offside-artefacts"
HF_TYPE="dataset"

ARTEFACTS=(
  "catboost_model.cbm"
  "fitted_pipeline.pkl"
  "player_profiles.pkl"
  "te_smooth_maps.pkl"
)

echo ""
echo "Checking model artefacts..."
for fname in "${ARTEFACTS[@]}"; do
  if [ -f "$fname" ]; then
    echo "  ✅ $fname already present"
  else
    echo "  ⬇️  Downloading $fname from HuggingFace Hub..."
    python - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="$HF_REPO",
    filename="$fname",
    repo_type="$HF_TYPE",
    local_dir="."
)
# hf_hub_download may put it in a subdirectory — move to cwd if needed
if os.path.abspath(path) != os.path.abspath("$fname"):
    shutil.move(path, "$fname")
    print(f"  Moved to ./$fname")
PYEOF
    echo "  ✅ $fname downloaded"
  fi
done

echo ""
echo "Build complete! Starting Streamlit..."
