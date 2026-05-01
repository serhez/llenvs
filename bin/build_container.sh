#!/bin/bash
# Build the Singularity container for llenvs / value-bench.
#
# Reads cluster defaults from bin/_cluster.sh. Override with env vars:
#   LLENVS_SIF                target .sif path
#   LLENVS_VLLM_IMAGE_TAG     docker tag (default: v0.19.0)
#   LLENVS_BUILD_MODE         'remote' (singularity build --remote) or
#                             'pull'   (singularity pull)
#                             default: 'remote', falls back to 'pull' on failure.
#   LLENVS_BUILD_SCRATCH_DIR  scratch dir for singularity's temp/cache blobs
#                             (default: $TMPDIR, else /tmp). This can grow
#                             to tens of GB during the build — point it at a
#                             large local disk on clusters with small /tmp.
#
# Neither mode needs sudo or fakeroot on the host. 'remote' requires a Sylabs
# cloud auth token (run `singularity remote login` once). 'pull' works for any
# `docker://` source without auth.
#
# Usage:
#   bash bin/build_container.sh                 # run directly on a login node
#   sbatch --partition=<serial> bin/build_container.sh   # or under SLURM

set -euo pipefail

# When submitted via sbatch, $0 points at /var/spool/slurmd/.../slurm_script
# (slurm copies the script), so dirname $0 won't find _cluster.sh. Prefer
# SLURM_SUBMIT_DIR when set. Submit from the repo root OR export LLENVS_BIN_DIR.
if [ -n "${LLENVS_BIN_DIR:-}" ] && [ -f "${LLENVS_BIN_DIR}/_cluster.sh" ]; then
    BIN_DIR="${LLENVS_BIN_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/bin/_cluster.sh" ]; then
    BIN_DIR="${SLURM_SUBMIT_DIR}/bin"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/_cluster.sh" ]; then
    BIN_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT="$(readlink -f "$0")"
    BIN_DIR="$(dirname "$SCRIPT")"
fi

# shellcheck source=./_cluster.sh
source "${BIN_DIR}/_cluster.sh"

: "${LLENVS_SIF:?LLENVS_SIF is not set; edit bin/_cluster.sh or export it}"
: "${LLENVS_VLLM_IMAGE_TAG:=v0.19.0}"
: "${LLENVS_BUILD_MODE:=remote}"

SOURCE_URI="docker://vllm/vllm-openai:${LLENVS_VLLM_IMAGE_TAG}"

OUTPUT_DIR="$(dirname "${LLENVS_SIF}")"
mkdir -p "${OUTPUT_DIR}"

# Singularity needs a scratch dir with plenty of space for docker layer blobs.
# Prefer $TMPDIR, fall back to /tmp. Override via LLENVS_BUILD_SCRATCH_DIR for
# clusters where neither has enough space.
LLENVS_BUILD_SCRATCH_DIR="${LLENVS_BUILD_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
TMPDIR_BUILD="${LLENVS_BUILD_SCRATCH_DIR}/${USER}_llenvs_singularity_build_$$"
mkdir -p "${TMPDIR_BUILD}"
export SINGULARITY_TMPDIR="${TMPDIR_BUILD}"
export SINGULARITY_CACHEDIR="${TMPDIR_BUILD}/cache"

cleanup() { rm -rf "${TMPDIR_BUILD}"; }
trap cleanup EXIT

echo "==> Building ${LLENVS_SIF}"
echo "    source:  ${SOURCE_URI}"
echo "    mode:    ${LLENVS_BUILD_MODE}"
echo "    cluster: ${LLENVS_CLUSTER}"
echo "    start:   $(date)"

do_remote_build() {
    singularity build --remote "${LLENVS_SIF}" "${SOURCE_URI}"
}
do_pull() {
    singularity pull "${LLENVS_SIF}" "${SOURCE_URI}"
}

case "${LLENVS_BUILD_MODE}" in
    remote)
        if ! do_remote_build; then
            echo "==> remote build failed, falling back to singularity pull" >&2
            do_pull
        fi
        ;;
    pull)
        do_pull
        ;;
    *)
        echo "ERROR: unknown LLENVS_BUILD_MODE='${LLENVS_BUILD_MODE}' (expected 'remote' or 'pull')" >&2
        exit 1
        ;;
esac

echo "==> Done at $(date)"
ls -lh "${LLENVS_SIF}"
