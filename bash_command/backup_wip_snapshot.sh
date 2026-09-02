#!/usr/bin/env bash
# Create a complete WIP snapshot with visible progress (avoids "git hung" confusion).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_REF="${BASE_REF:-40249cf}"
BRANCH="${BRANCH:-feat/merge-vitac-0.1-pipeline}"
WIP_BRANCH="${WIP_BRANCH:-wip/before-vitac01-merge}"
COMMIT_MSG="${COMMIT_MSG:-WIP: complete snapshot before ViTac 0.1 merge}"

TOTAL_STEPS=8
step=0
progress() {
  step=$((step + 1))
  printf '\n[%d/%d] %s\n' "$step" "$TOTAL_STEPS" "$1"
}

spinner_wait() {
  local pid=$1
  local label=$2
  local chars='|/-\'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    i=$(((i + 1) % 4))
    printf '\r  %s %s' "${chars:$i:1}" "$label"
    sleep 0.2
  done
  wait "$pid"
  local rc=$?
  printf '\r  done: %s (%ss)\n' "$label" "${SECONDS}"
  return "$rc"
}

progress "Verify branch and undo incomplete WIP commit (keep all files on disk)"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
git reset --mixed "$BASE_REF"
git branch -D "$WIP_BRANCH" 2>/dev/null || true

progress "Update .gitignore (exclude vendor / large archives)"
touch .gitignore
for pat in \
  'logs.zip' \
  'source/ViTacLab/ViTacLab/assets.zip' \
  'source/ViTacLab/ViTacLab/utils.zip' \
  'vitac_full_delivery_pack.tar.gz' \
  'vendor/' \
  'policy/**/checkpoints/' \
  'policy/**/data/' \
  'policy/**/outputs/' \
  'policy/**/info/' \
  'policy/**/wandb/' \
  'policy/**/*.ckpt' \
  'policy/**/*.pt' \
  'policy/**/*.pth' \
  'policy/**/*.zarr/' \
  ; do
  grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore
done

progress "Stage code in batches (with per-batch progress)"
STAGE_PATHS=(
  .gitignore
  README_send.md
  bash_command
  source/ViTacLab/ViTacLab/assets/sensor/tacsl_sensor
  source/ViTacLab/ViTacLab/assets/sensor/shadow_hand_grid_tactile
  source/ViTacLab/ViTacLab/assets/sensor/shadow_hand_full_tactile
  source/ViTacLab/ViTacLab/assets/sensor/wuji_hand_grid_tactile
  source/ViTacLab/ViTacLab/assets/robot
  source/ViTacLab/ViTacLab/assets/sensor/__init__.py
  source/ViTacLab/ViTacLab/tasks
  scripts/demo
  scripts/debug
  scripts/rsl_rl
  scripts/teleoperation
  scripts/diffusion_policy
)

POLICY_CODE_PATHS=(
  policy/README.md
  policy/prepare_dataset.py
  policy/train_policy.py
  policy/Diffusion_Policy/deployment.py
  policy/Diffusion_Policy/deploy_policy.py
  policy/Diffusion_Policy/DP.py
  policy/Diffusion_Policy/diffusion_policy_core
  policy/Diffusion_Policy/config
  policy/ViTacDP/deployment.py
  policy/ViTacDP/deploy_policy.py
  policy/ViTacDP/DP.py
  policy/ViTacDP/vitacdp_core
  policy/ViTacDP/config
  policy/Ours
)

batch_total=$((${#STAGE_PATHS[@]} + ${#POLICY_CODE_PATHS[@]}))
batch_i=0
for p in "${STAGE_PATHS[@]}" "${POLICY_CODE_PATHS[@]}"; do
  batch_i=$((batch_i + 1))
  if [[ -e "$p" ]]; then
    printf '  [%d/%d] git add %s ... ' "$batch_i" "$batch_total" "$p"
    git add "$p"
    echo "ok"
  else
    printf '  [%d/%d] skip missing: %s\n' "$batch_i" "$batch_total" "$p"
  fi
done

progress "Unstage excluded large/binary paths if accidentally picked up"
for p in \
  logs.zip \
  vitac_full_delivery_pack.tar.gz \
  source/ViTacLab/ViTacLab/assets.zip \
  source/ViTacLab/ViTacLab/utils.zip \
  vendor \
  policy/ViTacDP/checkpoints \
  policy/Diffusion_Policy/checkpoints \
  policy/ViTacDP/data \
  policy/Diffusion_Policy/data \
  ; do
  git reset HEAD -- "$p" 2>/dev/null || true
done
# Drop zip task bundles from index (keep source tree)
git reset HEAD -- ':(glob)**/*.zip' 2>/dev/null || true
git reset HEAD -- ':(glob)policy/**/*.ckpt' ':(glob)policy/**/*.pt' ':(glob)policy/**/*.pth' 2>/dev/null || true

progress "Show staged summary"
git diff --cached --stat | tail -20
echo "---"
echo "Staged files: $(git diff --cached --name-only | wc -l)"
echo "Must include tacsl_sensor:"
git diff --cached --name-only | rg 'tacsl_sensor' | head -5 || echo "  WARNING: tacsl_sensor not staged"

progress "Commit snapshot (spinner shows activity; large repos may take 1-3 min)"
SECONDS=0
(
  git commit -m "$COMMIT_MSG"
) &
spinner_wait $! "git commit running"

progress "Recreate safety branch pointer"
git branch -f "$WIP_BRANCH" HEAD

progress "Done"
git log -1 --oneline
git status -sb | head -20
echo ""
echo "[OK] Backup complete on branch: $BRANCH"
echo "[OK] Safety branch: $WIP_BRANCH -> $(git rev-parse --short HEAD)"
