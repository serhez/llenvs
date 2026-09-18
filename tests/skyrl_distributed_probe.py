"""Native FSDP worker fixture with read-only parameter/gradient observations."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ParameterAudit:
    model: Any

    def audit_snapshot(self, phase, directory):
        import torch
        from torch.distributed.tensor import DTensor

        assert phase in ("initial", "gradient", "update")
        root = Path(directory)
        assert root.is_absolute() and root.is_dir()
        leader = torch.distributed.get_rank() == 0
        values, digest = {}, hashlib.sha256()
        if phase == "initial" and leader:
            self._audit_initial = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.grad if phase == "gradient" else parameter
            assert value is not None, f"missing {phase}: {name}"
            # ALL ranks enter full_tensor's real collective, even though only
            # rank zero retains CPU evidence. Never compare flattened shards.
            full = value.full_tensor() if isinstance(value, DTensor) else value
            if leader:
                current = full.detach().float().cpu().clone()
                assert bool(torch.isfinite(current).all()), name
                if phase == "initial":
                    self._audit_initial[name] = current
                    digest.update(json.dumps([name, list(current.shape)]).encode())
                    digest.update(current.numpy().tobytes())
                else:
                    values[name] = (
                        current - self._audit_initial.pop(name) if phase == "update" else current
                    )
            del full
        if not leader:
            return None
        if phase == "initial":
            assert self._audit_initial
            return digest.hexdigest()
        assert values
        with (root / f"{phase}.pt").open("xb") as stream:
            torch.save(values, stream)
        norm = sum(v.double().square().sum().item() for v in values.values()) ** 0.5
        assert norm > 0, f"zero {phase}: no training evidence"
        return norm


@contextmanager
def policy_workers(cfg):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl.train.utils import get_ray_pg_ready_with_timeout
    from skyrl.train.utils.utils import ResolvedPlacementGroup

    dp = cfg.trainer.placement.policy_num_gpus_per_node
    group = None
    pg = placement_group([{"GPU": dp, "CPU": dp}], strategy="PACK")
    try:
        get_ray_pg_ready_with_timeout(pg, timeout=60)
        actor = ray.remote(num_gpus=1)(
            type("AuditedPolicy", (ParameterAudit, FSDPPolicyWorkerBase), {})
        )
        group = PPORayActorGroup(
            cfg.trainer,
            num_nodes=1,
            num_gpus_per_node=dp,
            ray_actor_type=actor,
            pg=ResolvedPlacementGroup(pg),
            num_gpus_per_actor=1,
            colocate_all=False,
        )
        ray.get(group.async_init_model(cfg.trainer.policy.model.path, num_training_steps=2))
        dispatch = WorkerDispatch(cfg, policy_actor_group=group)

        def snapshot(phase, directory):
            results = ray.get(
                group.async_run_ray_method("pass_through", "audit_snapshot", phase, str(directory))
            )
            assert results[0] is not None and all(r is None for r in results[1:])
            return results[0]

        yield dispatch, snapshot
    finally:
        if group is not None:
            for info in group.actor_infos:
                ray.kill(info.handle, no_restart=True)
        remove_placement_group(pg)


def train_step(dispatch, batch, snapshot, directory):
    """No training-loop copy: native worker loss, all-reduce, clipping and Adam."""
    import math

    initial = snapshot("initial", directory)
    output = dispatch.forward_backward("policy", batch, return_per_token_outputs=False)
    assert output.metrics and all(math.isfinite(v) for v in output.metrics.values())
    gradient = snapshot("gradient", directory)
    grad_norm = dispatch.optim_step("policy")
    assert grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0
    update = snapshot("update", directory)
    return dict(
        initial=initial,
        gradient_norm=gradient,
        update_norm=update,
        clip_grad_norm=grad_norm,
        metrics=output.metrics,
    )
