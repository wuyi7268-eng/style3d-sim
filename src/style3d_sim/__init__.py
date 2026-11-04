# SPDX-FileCopyrightText: Copyright (c) 2026 Zhejiang Linctex Digital Technology Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Python API for Style3D Sim."""

from typing import Any, Protocol

from ._cuda_runtime import load_cuda_runtime as _load_cuda_runtime
from .assets import get_asset, get_assets

_cuda_runtime_handle = _load_cuda_runtime()

from ._core import (
    AffineBodyRigidityModel,
    Config,
    Contact,
    Control,
    InertiaMode,
    JointConstraintMode,
    LogLevel,
    Model,
    NonlinearSolver,
    Solver,
    Span,
    State,
    is_logged_in,
    login,
)
from ._core import (
    set_log_callback as _set_log_callback,
)
from .viewer_polyscope import ViewerPolyscope


class LogCallback(Protocol):
    """Callable interface used to receive native solver log messages."""

    def __call__(
        self,
        file_name: str,
        function_name: str,
        line: int,
        level: LogLevel,
        message: str,
        /,
    ) -> None:
        ...


def set_log_callback(callback: LogCallback) -> None:
    """Register a Python callback for native solver log messages."""

    # Keep the native function private so callers see the named callback protocol.
    _set_log_callback(callback)

__all__ = [
    "AffineBodyRigidityModel",
    "Config",
    "Contact",
    "Control",
    "InertiaMode",
    "JointConstraintMode",
    "LogCallback",
    "LogLevel",
    "Model",
    "ModelBuilder",
    "NewtonSolver",
    "NonlinearSolver",
    "ParticleSurface",
    "ParticleSurfaceMesh",
    "Solver",
    "Span",
    "State",
    "ViewerPolyscope",
    "configure_fluid",
    "configure_granular",
    "get_asset",
    "get_assets",
    "is_logged_in",
    "log_surface_mesh",
    "login",
    "set_log_callback",
]


def __getattr__(name: str) -> Any:
    """Load optional Newton and Warp helpers only when their public types are requested."""
    if name in ("ParticleSurface", "ParticleSurfaceMesh", "log_surface_mesh"):
        from . import particle_surface

        value = getattr(particle_surface, name)
        globals()[name] = value
        return value
    if name == "configure_granular":
        from .granular import configure_granular

        globals()[name] = configure_granular
        return configure_granular
    if name == "configure_fluid":
        from .fluid import configure_fluid

        globals()[name] = configure_fluid
        return configure_fluid
    if name == "ModelBuilder":
        from .newton_model_builder import ModelBuilder

        globals()[name] = ModelBuilder
        return ModelBuilder
    if name == "NewtonSolver":
        from .newton_solver import NewtonSolver

        globals()[name] = NewtonSolver
        return NewtonSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
