"""Thread-safe global application state container."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

try:  # pragma: no cover - optional import for Pydantic v2
    from pydantic import BaseModel, ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - fallback for Pydantic v1
    from pydantic import BaseModel  # type: ignore

    ConfigDict = None  # type: ignore


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _LOG_PATH = Path(__file__).resolve().parents[2] / "log.txt"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        _FILE_HANDLER.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        _LOGGER.addHandler(_FILE_HANDLER)
    except OSError:
        _LOGGER.addHandler(logging.NullHandler())
else:
    _LOGGER.addHandler(logging.NullHandler())


class AppState(BaseModel):
    """Structured data shared between orchestrator and API layers."""

    A_px2cnc: list[list[float]] | None = None
    ref_pts_px: list[list[float]] | None = None
    last_heightmap: dict | None = None
    allow_execute_moves: bool = False

    if ConfigDict is not None:  # pragma: no cover - executed under pydantic v2
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
        )
    else:  # pragma: no cover - executed under pydantic v1
        class Config:
            """Pydantic configuration."""

            anystr_strip_whitespace = True
            validate_assignment = True


def _copy_state(state: AppState, *, update: dict | None = None) -> AppState:
    """Return a deep copy of ``state`` optionally applying ``update``."""

    kwargs = {"deep": True}
    if update is not None:
        kwargs["update"] = update
    if hasattr(state, "model_copy"):
        return state.model_copy(**kwargs)  # type: ignore[attr-defined]
    return state.copy(**kwargs)


class AppStateStore:
    """Lock-protected wrapper around :class:`AppState`."""

    def __init__(self) -> None:
        """Initialise the state container with a default state."""

        self._lock = threading.RLock()
        self._state = AppState()
        _LOGGER.debug("AppStateStore initialised with default state")

    def read(self) -> AppState:
        """Return a deep copy of the current state.

        The copy ensures callers cannot mutate the underlying storage without
        acquiring the lock via :meth:`update` or :meth:`mutate`.
        """

        with self._lock:
            state_copy = _copy_state(self._state)
        return state_copy

    def update(self, **changes: object) -> AppState:
        """Update selected fields atomically and return the new state."""

        with self._lock:
            self._state = _copy_state(self._state, update=changes)
            new_state = _copy_state(self._state)
        _LOGGER.debug("AppState updated: %s", changes)
        return new_state

    def replace(self, new_state: AppState) -> AppState:
        """Replace the entire state with ``new_state`` atomically."""

        with self._lock:
            self._state = _copy_state(new_state)
            stored = _copy_state(self._state)
        _LOGGER.debug("AppState replaced")
        return stored

    def mutate(self, mutator: Callable[[AppState], AppState]) -> AppState:
        """Apply ``mutator`` to a copy of the state and store the result."""

        with self._lock:
            proposal = mutator(_copy_state(self._state))
            if not isinstance(proposal, AppState):  # defensive programming
                raise TypeError("mutator must return AppState instance")
            self._state = _copy_state(proposal)
            stored = _copy_state(self._state)
        _LOGGER.debug("AppState mutated via callable %s", mutator)
        return stored

    def reset(self) -> AppState:
        """Reset the store to a pristine :class:`AppState` instance."""

        with self._lock:
            self._state = AppState()
            stored = _copy_state(self._state)
        _LOGGER.info("AppState reset to defaults")
        return stored


app_state = AppStateStore()

__all__ = ["AppState", "AppStateStore", "app_state"]
