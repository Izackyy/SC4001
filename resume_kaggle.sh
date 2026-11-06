#!/usr/bin/env bash
# Check the SC4001 Kaggle grids and pull anything that has finished.
# Safe to run repeatedly; it never launches a run.
set -u
PROJECT="$(cd "$(dirname "$0")/Project" && pwd)"
TMP="${TMPDIR:-/tmp}/sc4001_pull"
for SLUG in sc4001-phase2-kiba-seeds sc4001-phase2-kiba-splits; do
  STATUS=$(kaggle kernels status "isaacwjk21/$SLUG" 2>&1 | tail -1)
  echo "== $SLUG: $STATUS"
  case "$STATUS" in
    *COMPLETE*)
      DEST="$TMP/$SLUG"; rm -rf "$DEST"; mkdir -p "$DEST"
      kaggle kernels output "isaacwjk21/$SLUG" -p "$DEST" >/dev/null 2>&1
      if [ -d "$DEST/outputs/phase2" ]; then
        cp -r "$DEST/outputs/phase2/." "$PROJECT/outputs/phase2/"
        echo "   pulled into Project/outputs/phase2/"
      else
        echo "   WARNING: no outputs/phase2 in the download; check $DEST"
      fi ;;
    *ERROR*|*CANCEL*)
      echo "   FAILED - fetch the log: kaggle kernels output isaacwjk21/$SLUG -p <dir>" ;;
  esac
done
echo "== metrics files now present: $(find "$PROJECT/outputs/phase2" -name metrics.json | wc -l)"
