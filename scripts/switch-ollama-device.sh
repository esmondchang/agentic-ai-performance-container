#!/usr/bin/env bash
set -euo pipefail

SERVICE="ollama.service"
DROP_IN_DIR="/etc/systemd/system/${SERVICE}.d"
DROP_IN_FILE="${DROP_IN_DIR}/zz-device-mode.conf"
AMX_BACKEND="/usr/local/lib/ollama/libggml-cpu-sapphirerapids.so"

usage() {
  echo "Usage: $0 {cpu|gpu [device-list]|status}"
  echo "  cpu     Hide NVIDIA, Vulkan, and ROCm GPUs from Ollama."
  echo "  gpu     Allow Ollama to auto-detect NVIDIA GPUs."
  echo "  gpu 0,1 Restrict Ollama to the listed NVIDIA GPU indices."
  echo "            Multiple indices also enable model spreading and one parallel"
  echo "            inference slot per selected GPU."
  echo "  status  Show the active mode, service environment, and loaded models."
}

require_root() {
  if [[ ${EUID} -ne 0 ]]; then
    exec sudo -- "$0" "$@"
  fi
}

show_status() {
  local environment mode gpu_devices
  environment="$(systemctl show "$SERVICE" --property=Environment --value)"

  if [[ "$environment" == *"CUDA_VISIBLE_DEVICES=-1"* ]]; then
    mode="CPU only"
  elif [[ "$environment" =~ CUDA_VISIBLE_DEVICES=([^[:space:]]+) ]]; then
    gpu_devices="${BASH_REMATCH[1]//\"/}"
    mode="GPU devices ${gpu_devices}"
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
  local devices="${1:-auto}" available device device_count
  local -a requested_devices

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi is unavailable; NVIDIA driver is not ready." >&2
    exit 1
  fi
  nvidia-smi -L >/dev/null

  install -d -m 0755 "$DROP_IN_DIR"

  if [[ "$devices" == "auto" ]]; then
    tee "$DROP_IN_FILE" >/dev/null <<'EOF'
[Service]
UnsetEnvironment=CUDA_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
EOF
    return
  fi

  if [[ ! "$devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: GPU device list must contain comma-separated indices, for example 0,1." >&2
    exit 2
  fi

  available="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
  IFS=',' read -r -a requested_devices <<< "$devices"
  for device in "${requested_devices[@]}"; do
    if ! grep -Fxq "$device" <<< "$available"; then
      echo "Error: NVIDIA GPU index $device is unavailable." >&2
      echo "Available GPU indices:" >&2
      echo "$available" >&2
      exit 1
    fi
  done

  device_count="${#requested_devices[@]}"
  if (( device_count > 1 )); then
    tee "$DROP_IN_FILE" >/dev/null <<EOF
[Service]
Environment="CUDA_VISIBLE_DEVICES=${devices}"
Environment="GGML_VK_VISIBLE_DEVICES=-1"
Environment="ROCR_VISIBLE_DEVICES=-1"
Environment="OLLAMA_SCHED_SPREAD=1"
Environment="OLLAMA_NUM_PARALLEL=${device_count}"
Environment="OLLAMA_MAX_LOADED_MODELS=${device_count}"
EOF
  else
    tee "$DROP_IN_FILE" >/dev/null <<EOF
[Service]
Environment="CUDA_VISIBLE_DEVICES=${devices}"
Environment="GGML_VK_VISIBLE_DEVICES=-1"
Environment="ROCR_VISIBLE_DEVICES=-1"
Environment="OLLAMA_SCHED_SPREAD=0"
EOF
  fi
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
    if [[ -n "${3:-}" ]]; then
      usage
      exit 2
    fi
    gpu_devices="${2:-auto}"
    write_gpu_drop_in "$gpu_devices"
    restart_service
    wait_for_api
    if [[ "$gpu_devices" == "auto" ]]; then
      echo "Ollama switched to GPU auto-detect mode."
    else
      echo "Ollama restricted to NVIDIA GPU devices: $gpu_devices"
      if [[ "$gpu_devices" == *,* ]]; then
        echo "Multi-GPU spreading and parallel inference enabled."
      fi
    fi
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
