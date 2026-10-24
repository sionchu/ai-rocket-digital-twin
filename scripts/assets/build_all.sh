#!/usr/bin/env bash
# Full VTVL v1 visual asset pipeline: textures + USD, sky bake, preview renders, contact sheet.
#   bash scripts/assets/build_all.sh            (about 35 min on 4 CPU cores)
#   SAMPLES=32 bash scripts/assets/build_all.sh (quick look)
set -euo pipefail
cd "$(dirname "$0")/../.."
SAMPLES="${SAMPLES:-96}"
PYTHONPATH=src python3 scripts/assets/build_vtvl_assets.py
DUMPS=assets/vtvl_v1/preview/_dumps
mkdir -p "$DUMPS"
for f in pad vehicle_gimbal_rcs_conical_A vehicle_tri_diff_conical_A vehicle_quad_diff_conical_A vehicle_coldgas_conical_A; do
  python3 scripts/assets/usd_to_blender_dump.py "assets/vtvl_v1/$f.usd" "$DUMPS/$f"
done
blender -b --python scripts/assets/render_previews.py -- sky | grep RENDER
for s in hero aft layouts site; do
  blender -b --python scripts/assets/render_previews.py -- "$s" --samples "$SAMPLES" --ss "${SS:-2}" | grep RENDER
done
python3 scripts/assets/make_contact_sheet.py
rm -rf "$DUMPS"
