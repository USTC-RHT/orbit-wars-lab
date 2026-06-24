"""Feature-row BC ONNX agent for ckpt_step_00070000.

The policy network runs in ONNX Runtime. Feature-row construction and action
formatting use the packaged NumPy runtime, so the match path does not import JAX
or Torch.
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
os.environ.setdefault("ORBIT_WARS_ONNX_INTRA_THREADS", "1")
os.environ.setdefault("ORBIT_WARS_ONNX_INTER_THREADS", "1")

_PACKAGED_AGENT_DIR = Path.cwd().resolve()


def _has_manifest(path: Path) -> bool:
    return (path / "models" / "manifest.json").is_file()


def _resolve_agent_dir() -> Path:
    path = globals().get("__file__")
    if path:
        candidate = Path(path).resolve().parent
        if _has_manifest(candidate):
            return candidate

    # Kaggle executes main.py with exec(..., env={}) and does not set
    # __file__. During that exec it temporarily appends the agent directory to
    # sys.path, so recover the package root from there.
    for entry in reversed(sys.path):
        if not entry:
            continue
        try:
            candidate = Path(entry).resolve()
        except (OSError, TypeError):
            continue
        if _has_manifest(candidate):
            return candidate

    if _has_manifest(_PACKAGED_AGENT_DIR):
        return _PACKAGED_AGENT_DIR
    return Path.cwd().resolve()


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


def _set_agent_dir_from_config(config: Any = None) -> None:
    global _HERE
    raw_path = _read_field(config, "__raw_path__", None)
    if not raw_path:
        return
    try:
        candidate = Path(raw_path).resolve().parent
    except (OSError, TypeError):
        return
    if not _has_manifest(candidate):
        return
    _HERE = candidate
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _prefer_local_package()
                _clear_foreign_jax_agent_modules()
                from jax_agent.bc_numpy_runtime import BCNumpyOnnxPolicyAgent

                _RUNTIME = BCNumpyOnnxPolicyAgent.from_dir(_HERE / "models", max_fleets=512)
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


_start_preload()


def agent(observation: Any, configuration: Any = None) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        _set_agent_dir_from_config(configuration)
        player = int(_read_field(observation, "player", 0) or 0)
        return _load_runtime().act(observation, player=player)
    except Exception:
        if not _ERROR_LOGGED:
            print(traceback.format_exc(), flush=True)
            _ERROR_LOGGED = True
        return []
