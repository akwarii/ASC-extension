from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from ovito.io import import_file
from ovito.pipeline import Pipeline

from ASC_extension import ASCModifier

DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_path() -> Path:
    """Fixture providing the path to the test xyz file."""
    return DATA_DIR / "Si_traj.xyz"


@pytest.fixture
def model_path() -> str:
    """Fixture providing the path to the model checkpoint."""
    return str(DATA_DIR / "painn.pt2")


@pytest.fixture
def device() -> str:
    """Fixture providing the device to use for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def pipeline(data_path: Path) -> Generator[Pipeline, Any, None]:
    """Provides an OVITO pipeline loaded with the test file."""
    pipeline = import_file(data_path)
    yield pipeline


@pytest.fixture
def modifier() -> ASCModifier:
    """Provides a fresh instance of the modifier for every test."""
    return ASCModifier()


def test_defaults(pipeline: Pipeline, modifier: ASCModifier, device: str) -> None:
    """Test that the modifier has the expected default values."""
    assert modifier.ckpt_file == ""
    assert not hasattr(modifier, "model")
    assert modifier.device == device

    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    # By default, no property should be added to the data collection
    # until the user explicitly passes a model to the modifier.
    for prop in data.particles.keys():
        assert prop != "ASC Structure Type"


def test_ckpt_loading(pipeline: Pipeline, modifier: ASCModifier, model_path: str) -> None:
    """Test that the modifier can load a model from a checkpoint file."""
    modifier.ckpt_file = model_path
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    assert modifier.ckpt_file == model_path
    assert isinstance(modifier.model, torch.fx.GraphModule)

    # Check if the model parameters are on the correct device
    # and if the model is compiled when it should be.
    if modifier.device == "cuda":
        assert next(modifier.model.parameters()).is_cuda
        assert modifier._is_compiled
    elif modifier.device == "mps":
        assert next(modifier.model.parameters()).is_mps
        assert not modifier._is_compiled
    else:
        assert next(modifier.model.parameters()).is_cpu
        assert not modifier._is_compiled

    # After loading the model, the modifier should add the "ASC Structure Type" property to the data collection.
    assert "ASC Structure Type" in data.particles.keys()


def test_ckpt_toggle(pipeline: Pipeline, modifier: ASCModifier, model_path: str) -> None:
    """Test that changing the checkpoint file works as expected."""
    modifier.ckpt_file = model_path
    pipeline.modifiers.append(modifier)

    # Initially, the model should be loaded from the specified checkpoint.
    assert modifier.ckpt_file == model_path
    assert isinstance(modifier.model, torch.fx.GraphModule)

    # Change the checkpoint file to an empty string and check if the model is removed.
    modifier.ckpt_file = ""
    assert modifier.ckpt_file == ""
    assert not hasattr(modifier, "model")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda not available")
def test_compile_toggle(pipeline: Pipeline, modifier: ASCModifier, model_path: str) -> None:
    """Test that toggling the compile option works as expected."""
    modifier.ckpt_file = model_path
    pipeline.modifiers.append(modifier)

    # Initially, the model should be compiled.
    assert modifier._is_compiled

    # Disable compilation and check if it is still compiled.
    modifier.should_compile = False
    assert not modifier._is_compiled

    # Recompile the model and check if it is compiled.
    modifier.should_compile = True
    assert modifier._is_compiled


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="Cuda and MPS are not available",
)
def test_device_toggle(pipeline: Pipeline, modifier: ASCModifier, model_path: str) -> None:
    """Test that changing the device works as expected."""
    modifier.ckpt_file = model_path
    pipeline.modifiers.append(modifier)

    # Initially, the model should be on the default device.
    assert next(modifier.model.parameters()).device.type == modifier.device

    # Change the device and check if the model parameters are on the new device.
    new_device = "cpu"
    modifier.device = new_device
    assert next(modifier.model.parameters()).device.type == new_device

    # Change the device and check if the model parameters are on the new device.
    new_device = "cuda" if torch.cuda.is_available() else "mps"
    modifier.device = new_device
    assert next(modifier.model.parameters()).device.type == new_device
