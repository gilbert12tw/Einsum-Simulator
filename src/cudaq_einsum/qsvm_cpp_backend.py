"""
Helpers for loading and initializing the optional QSVM cuTENSOR backend.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path


_CPP_BACKEND_INIT_KEYS = {}
_CPP_BACKEND_INIT_WARNED = {}


def _candidate_backend_dirs():
    here = Path(__file__).resolve().parent
    candidates = []

    env_dir = os.environ.get("QSVM_CPP_BACKEND_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            here / "cpp_backend",
            here.parent.parent / "cpp_backend",
            here.parent.parent.parent / "cpp_backend",
            Path.cwd() / "cpp_backend",
        ]
    )

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _import_cpp_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    for candidate in _candidate_backend_dirs():
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass

        for so_path in sorted(candidate.glob(f"{module_name}*.so")):
            spec = importlib.util.spec_from_file_location(module_name, so_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    searched = [str(path) for path in _candidate_backend_dirs()]
    raise ModuleNotFoundError(
        f"Could not import optional backend module {module_name!r}. "
        f"Searched: {searched}"
    )


def _contract_config_key(contract_config):
    input_subscripts = tuple(str(x) for x in contract_config["input_subscripts"])
    triple_path = tuple(int(x) for x in contract_config["triple_path"])
    stream_0 = tuple(int(x) for x in contract_config.get("stream_0", []))
    stream_1 = tuple(int(x) for x in contract_config.get("stream_1", []))
    stream_2 = tuple(int(x) for x in contract_config.get("stream_2", []))
    return input_subscripts, triple_path, stream_0, stream_1, stream_2


def init_cpp_backend(contract_config, module_name="qsvm_cutensor_backend"):
    cpp_mod = _import_cpp_module(module_name)
    if not hasattr(cpp_mod, "init_backend"):
        if not _CPP_BACKEND_INIT_WARNED.get(module_name, False):
            print(
                f"[qsvm_cpp_backend] warning: {module_name} has no init_backend(config); "
                "skipping backend initialization."
            )
            _CPP_BACKEND_INIT_WARNED[module_name] = True
        return

    key = _contract_config_key(contract_config)
    if _CPP_BACKEND_INIT_KEYS.get(module_name) == key:
        return

    cpp_mod.init_backend(contract_config)
    _CPP_BACKEND_INIT_KEYS[module_name] = key
