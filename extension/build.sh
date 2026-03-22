#!/usr/bin/env bash
# Packages the extension/ directory into agentmemory.mcpb (a ZIP archive).
# Run from the project root: bash extension/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$SCRIPT_DIR/agentmemory.mcpb"

cd "$SCRIPT_DIR"
rm -f "$OUTPUT"

if command -v zip &>/dev/null; then
  zip -r "$OUTPUT" manifest.json server/ icon.png 2>/dev/null || zip -r "$OUTPUT" manifest.json server/
else
  python3 -c "
import zipfile, os
files = ['manifest.json', 'server/index.js']
if os.path.exists('icon.png'):
    files.append('icon.png')
with zipfile.ZipFile('$OUTPUT', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
"
fi

echo "Built: $OUTPUT"
