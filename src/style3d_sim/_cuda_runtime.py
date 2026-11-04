# SPDX-FileCopyrightText: Copyright (c) 2026 Zhejiang Linctex Digital Technology Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Linux prebuilt extension's CUDA runtime without importing PyTorch."""

import ctypes
import sys
from importlib.metadata import PackageNotFoundError, distribution


def load_cuda_runtime():
    if sys.platform != "linux":
        return None

    # Respect an already loaded runtime or the system loader's configured paths.
    # Keep the handle alive in the caller; RTLD_LOCAL avoids exporting CUDA symbols
    # globally into other CUDA major versions used by optional dependencies.
    try:
        return ctypes.CDLL("libcudart.so.12")
    except OSError:
        pass

    # NVIDIA wheels do not add their library directory to the dynamic linker path.
    # Locate this distribution specifically, so a CUDA 13 wheel cannot substitute
    # an incompatible SONAME or depend on import ordering to make CUDA 12 visible.
    try:
        runtime = distribution("nvidia-cuda-runtime-cu12")
        library = runtime.locate_file("nvidia/cuda_runtime/lib/libcudart.so.12")
        return ctypes.CDLL(str(library))
    except (PackageNotFoundError, OSError) as error:
        raise ImportError(
            "Style3D Sim requires libcudart.so.12 on Linux. Install the project dependencies "
            "(including nvidia-cuda-runtime-cu12>=12.9,<13) into the active Python environment."
        ) from error
