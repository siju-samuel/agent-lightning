# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest import mock

from agentlightning.verl.verl_compat import create_rl_sampler, task_runner_class


def _module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _TaskRunner09:
    pass


class _TaskRunner08:
    pass


def _sampler_09(data_config: Any, dataset: Any) -> str:
    return f"0.9:{data_config}:{dataset}"


def _sampler_08(data_config: Any, dataset: Any) -> str:
    return f"0.8:{data_config}:{dataset}"


# A None entry in sys.modules makes `import name` raise ImportError, which is how
# each layout hides the other one's modules.
_LAYOUT_09 = {
    "verl": _module("verl"),
    "verl.trainer": _module("verl.trainer"),
    "verl.trainer.main_ppo": _module("verl.trainer.main_ppo"),
    "verl.trainer.main_ppo_v0": _module("verl.trainer.main_ppo_v0", TaskRunner=_TaskRunner09),
    "verl.trainer.ppo": _module("verl.trainer.ppo"),
    "verl.trainer.ppo.utils": _module("verl.trainer.ppo.utils", create_rl_sampler=_sampler_09),
}

_LAYOUT_08 = {
    "verl": _module("verl"),
    "verl.trainer": _module("verl.trainer"),
    "verl.trainer.main_ppo": _module("verl.trainer.main_ppo", TaskRunner=_TaskRunner08, create_rl_sampler=_sampler_08),
    "verl.trainer.main_ppo_v0": None,
    "verl.trainer.ppo": _module("verl.trainer.ppo"),
    "verl.trainer.ppo.utils": None,
}


def test_resolves_the_0_9_layout() -> None:
    """verl 0.9 moved TaskRunner to main_ppo_v0 and create_rl_sampler to ppo.utils."""
    with mock.patch.dict(sys.modules, _LAYOUT_09):
        assert task_runner_class() is _TaskRunner09
        assert create_rl_sampler("cfg", "ds") == "0.9:cfg:ds"


def test_resolves_the_0_8_layout() -> None:
    """Both names still live in main_ppo on the verl versions below 0.9."""
    with mock.patch.dict(sys.modules, _LAYOUT_08):
        assert task_runner_class() is _TaskRunner08
        assert create_rl_sampler("cfg", "ds") == "0.8:cfg:ds"
