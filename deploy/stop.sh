#!/bin/bash
# BeforeInstall: stop services before code is replaced
set -euo pipefail

systemctl stop moneymaker-train.timer 2>/dev/null || true
systemctl stop moneymaker 2>/dev/null || true

echo "Services stopped."
