"""Feature-row BC ONNX agent for ckpt_step_00010000.

The policy network runs in ONNX Runtime. The current BC fight-feature builder is
loaded from the local rl-agent checkout and forced onto CPU for lab evaluation.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_PACKAGED_AGENT_DIR = Path("/home/dministrator/kaggle/orbit-wars-lab/agents/mine/bc-feature-row-step10000-onnx")


def _resolve_agent_dir() -> Path:
    path = globals().get("__file__")
    if path:
        candidate = Path(path).resolve().parent
        if (candidate / "models" / "manifest.json").is_file():
            return candidate
    if (_PACKAGED_AGENT_DIR / "models" / "manifest.json").is_file():
        return _PACKAGED_AGENT_DIR
    return Path(".").resolve()


_HERE = _resolve_agent_dir()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_PRELOAD_THREAD = None
_ERROR_LOGGED = False


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _prefer_local_package() -> None:
    agent_dir = str(_HERE)
    sys.path[:] = [p for p in sys.path if p != agent_dir]
    sys.path.insert(0, agent_dir)


def _clear_foreign_jax_agent_modules() -> None:
    local_root = (_HERE / "jax_agent").resolve()
    for name, module in list(sys.modules.items()):
        if name != "jax_agent" and not name.startswith("jax_agent."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            sys.modules.pop(name, None)
            continue
        try:
            resolved = Path(module_file).resolve()
        except OSError:
            sys.modules.pop(name, None)
            continue
        if not _path_is_relative_to(resolved, local_root):
            sys.modules.pop(name, None)


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _prefer_local_package()
                _clear_foreign_jax_agent_modules()
                from jax_agent.bc_onnx_runtime import BCOnnxPolicyAgent

                _RUNTIME = BCOnnxPolicyAgent.from_dir(_HERE / "models", max_fleets=512)
    return _RUNTIME


def _preload_runtime() -> None:
    try:
        runtime = _load_runtime()
        runtime.act({"planets": [], "fleets": [], "step": 0}, player=0)
    except Exception:
        global _ERROR_LOGGED
        if not _ERROR_LOGGED:
            print(traceback.format_exc(), flush=True)
            _ERROR_LOGGED = True


def _start_preload() -> None:
    global _PRELOAD_THREAD
    if _PRELOAD_THREAD is None:
        _PRELOAD_THREAD = threading.Thread(target=_preload_runtime, daemon=True)
        _PRELOAD_THREAD.start()


def agent(observation: Any, configuration: Any = None) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        player = int(_read_field(observation, "player", 0) or 0)
        return _load_runtime().act(observation, player=player)
    except Exception:
        if not _ERROR_LOGGED:
            print(traceback.format_exc(), flush=True)
            _ERROR_LOGGED = True
        return []
