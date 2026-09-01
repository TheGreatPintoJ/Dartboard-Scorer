#!/usr/bin/env bash
# Install the dartboard scorer as a systemd service.
#
#   sudo ./deploy/install.sh            # install and start
#   sudo ./deploy/install.sh --source 2 # ...pointing at /dev/video2
#
# Re-running it is safe: the code is refreshed, the service restarted, and
# anything in /var/lib/dart-scorer (config, calibration, throw log) is kept.
set -euo pipefail

PREFIX=/opt/dart-scorer
KINECT=0
STATE=/var/lib/dart-scorer
USERNAME=dartscorer
SOURCE=""
PORT=""
TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --token)  TOKEN="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --with-kinect) KINECT=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run this with sudo" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> user"
id -u "$USERNAME" >/dev/null 2>&1 || \
  useradd --system --home-dir "$PREFIX" --shell /usr/sbin/nologin "$USERNAME"
# Cameras are owned by the video group.
usermod -aG video "$USERNAME"

# --- optional: the Kinect v1 ------------------------------------------------
# Off by default, so a plain install still adds no apt packages at all. The
# Kinect is reached through libfreenect over raw USB rather than V4L2, so it
# needs the library, a group that can open the USB nodes, and the matching
# DeviceAllow line in the unit (which is already there).
if [[ "$KINECT" == "1" ]]; then
  echo "==> Kinect support"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends     libfreenect-dev freenect
  groupadd -f kinect
  usermod -aG kinect "$USERNAME"
  install -m 644 "$HERE/deploy/51-kinect.rules" /etc/udev/rules.d/51-kinect.rules
  # gspca_kinect claims the Kinect camera as a V4L2 device that enumerates but
  # cannot stream, which both breaks it and renumbers every other camera.
  install -m 644 "$HERE/deploy/kinect-blacklist.conf"     /etc/modprobe.d/dart-scorer-kinect.conf
  modprobe -r gspca_kinect 2>/dev/null || true
  udevadm control --reload && udevadm trigger --subsystem-match=usb
  if "$PREFIX/.venv/bin/python" -c "from dart_scorer.kinect import kinect_status as k; import sys; s=k(); print('    ' + (('ready, ' + str(s[\"plugged_in\"]) + ' plugged in') if s['available'] else s['reason'])); sys.exit(0)" 2>/dev/null; then :; else
    echo "    (could not query it yet - it is checked again when the service starts)"
  fi
fi

echo "==> code -> $PREFIX"
install -d -o "$USERNAME" -g "$USERNAME" "$PREFIX" "$STATE"
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' \
  --exclude 'config.json' --exclude 'calibration.json' --exclude '*.csv' \
  "$HERE"/ "$PREFIX"/
chown -R "$USERNAME:$USERNAME" "$PREFIX"

echo "==> python environment"
if [[ ! -x "$PREFIX/.venv/bin/python" ]]; then
  python3 -m venv "$PREFIX/.venv"
fi
"$PREFIX/.venv/bin/pip" install --quiet --upgrade pip
# opencv-python has no wheel for some ARM boards; opencv-python-headless does,
# and this service never opens a window.
"$PREFIX/.venv/bin/pip" install --quiet \
  "opencv-python-headless>=4.10" "numpy>=2.0" || \
  "$PREFIX/.venv/bin/pip" install --quiet "opencv-python>=4.10" "numpy>=2.0"
chown -R "$USERNAME:$USERNAME" "$PREFIX/.venv"

echo "==> settings"
if [[ ! -f /etc/default/dart-scorer ]]; then
  cat > /etc/default/dart-scorer <<EOF
# Settings for the dart-scorer service. systemctl restart dart-scorer to apply.
DART_HOST=0.0.0.0
DART_PORT=${PORT:-8080}
DART_SOURCE=${SOURCE:-0}
# Leave empty for no authentication. Anything else must be supplied as
# ?token=... in the URL or an X-Auth-Token header.
DART_TOKEN=${TOKEN}
EOF
else
  [[ -n "$SOURCE" ]] && sed -i "s|^DART_SOURCE=.*|DART_SOURCE=$SOURCE|" /etc/default/dart-scorer
  [[ -n "$PORT"   ]] && sed -i "s|^DART_PORT=.*|DART_PORT=$PORT|" /etc/default/dart-scorer
  [[ -n "$TOKEN"  ]] && sed -i "s|^DART_TOKEN=.*|DART_TOKEN=$TOKEN|" /etc/default/dart-scorer
fi
chmod 600 /etc/default/dart-scorer

echo "==> service"
install -m 644 "$HERE/deploy/dart-scorer.service" /etc/systemd/system/dart-scorer.service
systemctl daemon-reload
systemctl enable --now dart-scorer
sleep 2
systemctl --no-pager --lines=15 status dart-scorer || true

PORT_IN_USE="$(grep '^DART_PORT=' /etc/default/dart-scorer | cut -d= -f2)"
echo
echo "Scorer running on http://$(hostname -I | awk '{print $1}'):${PORT_IN_USE}/"
echo "  logs:    journalctl -u dart-scorer -f"
echo "  restart: systemctl restart dart-scorer"
echo "  camera:  edit /etc/default/dart-scorer, or change it in the web UI"
