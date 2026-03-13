#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$ROOT/site"

mkdir -p "$SITE/apps/board" "$SITE/apps/insert" "$SITE/apps/hydrofoil"

cp "$ROOT/foil-board-optimizer/viewer.html" "$SITE/apps/board/"
cp "$ROOT/foil-board-optimizer/force_viewer.html" "$SITE/apps/board/"
cp "$ROOT/foil-board-optimizer/comparison.html" "$SITE/apps/board/"

cp "$ROOT/hydrofoil/mockup.html" "$SITE/apps/hydrofoil/"

cp "$ROOT/foil-insert-load-comparison/index.html" "$SITE/apps/insert/"
cp "$ROOT/foil-insert-load-comparison/viewer-prototype.html" "$SITE/apps/insert/"
cp "$ROOT/foil-insert-load-comparison/viewer-app.js" "$SITE/apps/insert/"
cp "$ROOT/foil-insert-load-comparison/viewer-solver.js" "$SITE/apps/insert/"
cp "$ROOT/foil-insert-load-comparison/viewer-data.json" "$SITE/apps/insert/"
