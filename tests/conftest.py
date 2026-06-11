"""Shared pytest fixtures for the test suite."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow importing from agent/ without installing
sys.path.insert(0, str(REPO_ROOT / "agent"))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def state_dir(repo_root: Path) -> Path:
    return repo_root / "state"
