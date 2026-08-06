#!/bin/bash
# Server-side deploy script. Called by GitHub Actions via AWS SSM.
set -euo pipefail

REPO=/opt/moneymaker
BRANCH=prod

echo "=== Deploy started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== From: $(git -C $REPO rev-parse --short HEAD) ==="

cd $REPO
sudo -u moneymaker git fetch origin
sudo -u moneymaker git reset --hard origin/$BRANCH

echo "=== To:   $(git -C $REPO rev-parse --short HEAD) — $(git -C $REPO log -1 --format='%s') ==="

systemctl daemon-reload
systemctl restart moneymaker

sleep 3
if systemctl is-active --quiet moneymaker; then
    echo "=== Service: running ==="
else
    echo "=== Service: FAILED — check: journalctl -u moneymaker -n 50 ==="
    exit 1
fi
