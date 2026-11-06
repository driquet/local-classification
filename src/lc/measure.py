"""Cost measurements: peak memory, latency and size on disk.

Training and prediction each run in their own freshly spawned process. A process's peak memory then
belongs to one job only: the peak of the prediction process is what serving the model takes, without the
training that came before or the models measured earlier.
"""

import ctypes
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any


def in_fresh_process(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """Run ``function(*args)`` in a new spawned process: (its return value, the process's peak memory in MB).

    ``function`` must be importable at module level, and its arguments and result picklable.
    """
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        return pool.submit(_with_peak, function, *args).result()


def _with_peak(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    result = function(*args)
    return result, round(peak_memory_mb(), 1)


def peak_memory_mb() -> float:
    """This process's lifetime peak physical footprint, the "Memory" column of Activity Monitor.

    On Apple silicon, CPU and GPU share one memory, and the physical footprint counts both, including the
    Metal buffers that hold model weights on the GPU. The resident set size (RSS) misses those buffers.
    """
    if sys.platform != "darwin":
        raise NotImplementedError("peak memory is measured with macOS's proc_pid_rusage")
    info = (ctypes.c_uint64 * 40)()  # rusage_info_v4: a 16-byte UUID, then 64-bit counters
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, info) != 0:
        raise OSError("proc_pid_rusage failed")
    return info[2 + 28] / 2**20  # ri_lifetime_max_phys_footprint


def in_ram(model):
    """Copy a PyTorch model's weights into the process's own memory, and return the model.

    Weights loaded from safetensors stay mapped from the file, and the physical footprint leaves such
    file-backed pages out: a 575 MB model served on the CPU counted as almost nothing. A copy makes the
    weights count, as they do in any server that holds its model in RAM. (On the GPU the weights are copied
    to Metal buffers anyway, which the footprint counts.)
    """
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    return model


def gpu_used_mb() -> float | None:
    """Memory held by live tensors on the GPU, if PyTorch is loaded; ``None`` otherwise.

    PyTorch reserves GPU memory in large blocks, and the physical footprint counts whole blocks: about
    1.5 GB for any model from 17M to 150M parameters. This is what the model and its tensors actually use.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.backends.mps.is_available():
        return None
    return round(torch.mps.current_allocated_memory() / 2**20, 1)


def disk_mb(path: Path) -> float:
    """Total size of the files under ``path`` (symbolic links followed, as in a Hugging Face cache)."""
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20, 1)
