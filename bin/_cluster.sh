#!/bin/bash
# llenvs cluster profile — filesystem defaults for SingularityVLLMBackend.
#
# Sets cluster-specific paths that SingularityVLLMBackend reads as fallback
# defaults when no explicit kwargs are passed:
#
#   LLENVS_SIF          path to the Singularity .sif image
#   LLENVS_HF_HOME      HuggingFace cache root (auto-bound into the container)
#   LLENVS_BINDS        space-separated singularity --bind paths
#   LLENVS_HF_OFFLINE   1 if compute nodes have no outbound internet
#                       (backend sets HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1
#                       inside the container)
#
# Source this file once in your shell (or let your shell profile auto-source
# it) so `uv run python ...` picks up the defaults with zero flags. Any
# LLENVS_* variable set by the caller wins.
#
# Cluster selection: $LLENVS_CLUSTER (explicit), or auto-detected from the
# filesystem fingerprints below, falling back to 'generic'.
#
# To add a new cluster: add a case arm with the four variables above, plus a
# detection branch above if you want auto-detect. This file is the *only*
# place in llenvs that hardcodes cluster paths.

if [ -z "${LLENVS_CLUSTER:-}" ]; then
    if [ -d /leonardo_work ]; then
        LLENVS_CLUSTER="leonardo"
    else
        LLENVS_CLUSTER="generic"
    fi
fi

case "${LLENVS_CLUSTER}" in
    leonardo)
        : "${LLENVS_SIF:=/leonardo_work/FBKLM_prj1/mmerler/containers/llenvs-vllm.sif}"
        : "${LLENVS_HF_HOME:=/leonardo_work/FBKLM_prj1/hf_cache}"
        : "${LLENVS_BINDS:=/leonardo_work /leonardo_scratch /etc/pki ${LLENVS_HF_HOME}}"
        # Compute nodes here have no outbound internet, so HF libs must run
        # in offline mode inside the container. Prefetch weights on a login
        # node via `huggingface-cli download` before launching.
        : "${LLENVS_HF_OFFLINE:=1}"
        ;;
    generic)
        : "${LLENVS_SIF:=}"
        : "${LLENVS_HF_HOME:=${HF_HOME:-}}"
        : "${LLENVS_BINDS:=}"
        : "${LLENVS_HF_OFFLINE:=0}"
        ;;
    *)
        echo "ERROR: unknown LLENVS_CLUSTER='${LLENVS_CLUSTER}'. Edit $(dirname "$0")/_cluster.sh to add it." >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export LLENVS_CLUSTER LLENVS_SIF LLENVS_HF_HOME LLENVS_BINDS LLENVS_HF_OFFLINE
