# Copyright (c) Microsoft. All rights reserved.

"""Import shims for the verl layouts Agent Lightning supports.

verl 0.9 split ``verl.trainer.main_ppo`` in two: the v0 PPO path moved to
``verl.trainer.main_ppo_v0`` (``TaskRunner`` with it) and the dataset helpers to
``verl.trainer.ppo.utils``, while ``main_ppo`` became the v1 TransferQueue
trainer. Everything else Agent Lightning imports from verl kept its path.

Resolve the two moved names here so the entrypoint reads the same against 0.8
and 0.9+. 0.9 is what Intel XPU needs: the platform plugin layer
(``verl.plugin.platform``) that verl-hardware-plugin registers against landed
after 0.8.0.
"""

# Probing two layouts means one branch always names something the installed verl
# does not have, whichever version that is.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any

__all__ = ["create_rl_sampler", "task_runner_class"]


def task_runner_class() -> type[Any]:
    """Return verl's v0 ``TaskRunner``, whose worker-setup methods we delegate to."""
    try:
        from verl.trainer.main_ppo_v0 import TaskRunner
    except ImportError:  # verl < 0.9
        from verl.trainer.main_ppo import TaskRunner

    return TaskRunner


def create_rl_sampler(data_config: Any, dataset: Any) -> Any:
    """Build the training sampler verl's PPO trainer expects."""
    try:
        from verl.trainer.ppo.utils import create_rl_sampler as _create
    except ImportError:  # verl < 0.9
        from verl.trainer.main_ppo import create_rl_sampler as _create

    return _create(data_config, dataset)
