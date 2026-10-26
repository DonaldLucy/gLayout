#!/bin/bash
# Download and unpack the frozen E1, E2 and E3 task packets into this directory.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${CBL_DATA_URL:-https://github.com/DonaldLucy/gLayout/releases/download/cbl-data-v1/cbl-data-v1.tar.gz}"
curl -fL "$URL" | tar -xz
echo "Unpacked runtime/protocol_freezes"
