#!/usr/bin/env bash
set -euo pipefail

SERVICE="ollama.service"
DROP_IN_DIR="/etc/systemd/system/${SERVICE}.d"
DROP_IN_FILE="${DROP_IN_DIR}/zz-device-mode.conf"
AMX_BACKEND="/usr/local/lib/ollama/libggml-cpu-sapphirerapids.so"

usage() {
  echo "Usage: $0 {cpu|gpu|status}"
  echo "  cpu     Hide NVIDIA, Vulkan, and ROCm GPUs from Ollama."
  echo "  gpu     Allow Ollama to auto-detect and use the NVIDIA GPU."
  echo "  status  Show the active mode, service environment, and loaded models."
}

require_root() {
  if [[ ${EUID} -ne 0 ]]; then
    exec sudo -- "$0" "$@"
  fi
}

show_status() {
  local environment mode
  environment="$(systemctl show "$SERVICE" --property=Environment --value)"

  if [[ "$environment" == *"CUDA_VISIBLE_DEVICES=-1"* ]]; then
    mode="CPU only"
  else
    mode="GPU auto-detect"
  fi

  echo "Mode: $mode"
  echo "Service: $(systemctl is-active "$SERVICE" 2>/dev/null || true)"
  echo "Environment: ${environment:-<none>}"

  if [[ -e "$AMX_BACKEND" ]]; then
    echo "AMX backend: available (this host previously crashed in this backend)"
  else
    echo "AMX backend: disabled or not installed"
  fi

  if command -v curl >/dev/null 2>&1; then
    echo -n "Ollama API: "
    curl --silent --show-error --max-time 2 \
      http://127.0.0.1:11434/api/version || true
    echo
  fi

  if command -v ollama >/dev/null 2>&1; then
    ollama ps 2>/dev/null || true
  fi
}

write_cpu_drop_in() {
  install -d -m 0755 "$DROP_IN_DIR"
  tee "$DROP_IN_FILE" >/dev/null <<'EOF'
[Service]
Environment="CUDA_VISIBLE_DEVICES=-1"
Environment="GGML_VK_VISIBLE_DEVICES=-1"
Environment="ROCR_VISIBLE_DEVICES=-1"
EOF
}

write_gpu_drop_in() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi is unavailable; NVIDIA driver is not ready." >&2
    exit 1
  fi
  nvidia-smi -L >/dev/null

  install -d -m 0755 "$DROP_IN_DIR"
  tee "$DROP_IN_FILE" >/dev/null <<'EOF'
[Service]
UnsetEnvironment=CUDA_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
EOF
}

restart_service() {
  systemctl daemon-reload
  if ! systemctl restart "$SERVICE"; then
    echo "Ollama failed to restart. Check whether another process or container owns port 11434:" >&2
    echo "  sudo ss -ltnp 'sport = :11434'" >&2
    echo "  sudo journalctl -u ollama -n 100 --no-pager" >&2
    exit 1
  fi
}

wait_for_api() {
  local attempt

  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  for attempt in {1..30}; do
    if curl --silent --fail --max-time 1 \
      http://127.0.0.1:11434/api/version >/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "Ollama restarted but its API did not become ready within 30 seconds." >&2
  systemctl status "$SERVICE" --no-pager >&2 || true
  echo "Check logs with: sudo journalctl -u ollama -n 100 --no-pager" >&2
  exit 1
}

case "${1:-}" in
  cpu)
    require_root "$@"
    write_cpu_drop_in
    restart_service
    wait_for_api
    echo "Ollama switched to CPU-only mode."
    show_status
    ;;
  gpu)
    require_root "$@"
    write_gpu_drop_in
    restart_service
    wait_for_api
    echo "Ollama switched to GPU auto-detect mode."
    show_status
    ;;
  status)
    show_status
    ;;
  *)
    usage
    exit 2
    ;;
esac
