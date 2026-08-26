#!/bin/bash
# ValidateService: confirm the bot runner is active
set -euo pipefail

sleep 5

if systemctl is-active --quiet moneymaker; then
    echo "Validation passed: moneymaker is running."
    exit 0
else
    echo "Validation FAILED: moneymaker is not running."
    journalctl -u moneymaker -n 30 --no-pager
    exit 1
fi
