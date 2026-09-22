#!/usr/bin/env bash
# Mirror all recorded experiments and generated plots to the large data disk.
#
# Usage (from ros2_ws):
#   ./scripts/backup_experiments.sh
#   ./scripts/backup_experiments.sh --dry-run
#   ./scripts/backup_experiments.sh --no-delete
#
# The default is an exact mirror: files that no longer exist in the source are
# removed from the destination. The source and destination are deliberately
# fixed to the experiment directories to avoid mirroring another workspace.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROS2_WS="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SOURCE_DIR="${ROS2_WS}/data/experiments"
DEST_DIR="/data/UnderwaterSim/experiments"
DELETE_MODE=1
DRY_RUN=0

usage() {
    sed -n '2,12p' "$0"
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-delete)
            DELETE_MODE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$SOURCE_DIR" ]]; then
    printf 'Source directory does not exist: %s\n' "$SOURCE_DIR" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    printf 'rsync is required but was not found in PATH.\n' >&2
    exit 1
fi

mkdir -p -- "$DEST_DIR"

printf 'Source:      %s\n' "$SOURCE_DIR"
printf 'Destination: %s\n' "$DEST_DIR"
if ((DELETE_MODE)); then
    printf 'Mode:        exact mirror (--delete)\n'
else
    printf 'Mode:        additive/update (destination-only files kept)\n'
fi
if ((DRY_RUN)); then
    printf 'Execution:   dry run (no files changed)\n'
fi

RSYNC_ARGS=(-a --human-readable --info=stats2)
if ((DELETE_MODE)); then
    RSYNC_ARGS+=(--delete)
fi
if ((DRY_RUN)); then
    RSYNC_ARGS+=(--dry-run)
fi

rsync "${RSYNC_ARGS[@]}" "${SOURCE_DIR}/" "${DEST_DIR}/"

if ((DRY_RUN == 0)); then
    printf '\nBackup completed.\n'
    printf 'Source size:      '
    du -sh -- "$SOURCE_DIR" | awk '{print $1}'
    printf 'Destination size: '
    du -sh -- "$DEST_DIR" | awk '{print $1}'
    printf 'Source files:     '
    find "$SOURCE_DIR" -type f -printf '.' | wc -c
    printf 'Destination files:'
    find "$DEST_DIR" -type f -printf '.' | wc -c
fi
