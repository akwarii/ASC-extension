from pathlib import Path
from typing import Any
from collections.abc import Generator

from ovito.pipeline import Pipeline
import pytest
import torch
from ovito.io import import_file
from ASC_extension import PeriodicKNN


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "fixtures" / "Si_traj.xyz"


@pytest.fixture
def knn() -> Generator[PeriodicKNN, Any, None]:
    yield PeriodicKNN(num_neighbors=4)


@pytest.fixture
def pipeline(data_path: Path) -> Generator[Pipeline, Any, None]:
    yield import_file(data_path)


def test_atomic_numbers(pipeline: Pipeline, knn: PeriodicKNN) -> None:
    data = pipeline.compute()

    z = knn._get_atomic_numbers(data)
    assert z.shape == (data.particles.count,)
    assert torch.all(z == 14)


def test_knn_without_selection(pipeline: Pipeline, knn: PeriodicKNN) -> None:
    data = pipeline.compute()
    graph = knn.convert(data)

    assert graph.edge_index is not None
    assert graph.num_nodes == data.particles.count
    assert graph.edge_index.shape[1] == data.particles.count * knn.num_neighbors


def test_knn_selection(pipeline: Pipeline, knn: PeriodicKNN) -> None:
    data = pipeline.compute()

    selection = torch.tensor([0, 2])
    num_selected = selection.numel()

    graph = knn.convert(data, selection=selection)

    assert graph.x is not None
    assert graph.edge_index is not None

    assert graph.num_nodes == num_selected
    assert graph.x.shape[0] == num_selected
    assert graph.edge_index.shape[1] == num_selected * knn.num_neighbors
