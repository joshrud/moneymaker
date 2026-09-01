#!/bin/bash
# ApplicationStart: install systemd units, then start services
set -euo pipefail

APP=/opt/moneymaker

# Install/update systemd unit files from repo so pipeline is self-contained
cp "$APP/deploy/moneymaker.service" /etc/systemd/system/moneymaker.service
cp "$APP/deploy/moneymaker-train.service" /etc/systemd/system/moneymaker-train.service
cp "$APP/deploy/moneymaker-train.timer" /etc/systemd/system/moneymaker-train.timer
chmod 644 /etc/systemd/system/moneymaker*.service /etc/systemd/system/moneymaker*.timer

systemctl daemon-reload
systemctl enable moneymaker
systemctl start moneymaker
systemctl enable moneymaker-train.timer
systemctl start moneymaker-train.timer

echo "Services started."
