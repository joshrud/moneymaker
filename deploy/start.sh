#!/bin/bash
# ApplicationStart: reload systemd units and start services
set -euo pipefail

systemctl daemon-reload
systemctl enable moneymaker
systemctl start moneymaker
systemctl enable moneymaker-train.timer
systemctl start moneymaker-train.timer

echo "Services started."
