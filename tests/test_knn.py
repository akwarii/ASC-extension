import pytest
import torch
from ovito.io import import_file
from ASC_extension import PeriodicKNN, get_atomic_numbers

def test_atomic_numbers():
    pipeline = import_file("tests/data.xyz")
    data = pipeline.compute()
    
    z = get_atomic_numbers(data)
    assert z.shape == (data.particles.count,)
    # In data.xyz, all atoms are Si (Z=14)
    assert torch.all(z == 14)

def test_knn_backends_consistency():
    pipeline = import_file("tests/data.xyz")
    # Frame 3 has 8 atoms and a cubic lattice
    data = pipeline.compute(3)
    
    k = 4
    knn = PeriodicKNN(k=k)
    
    # Test for first frame
    graph_ovito = knn.convert(data, backend="ovito")
    graph_freud = knn.convert(data, backend="freud")
    
    assert graph_ovito.num_nodes == data.particles.count
    assert graph_freud.num_nodes == data.particles.count
    
    assert graph_ovito.edge_index.shape[1] == data.particles.count * k
    assert graph_freud.edge_index.shape[1] == data.particles.count * k
    
    # Check if they have the same edges (order might differ)
    dist_ovito = torch.norm(graph_ovito.edge_attr, dim=1).sort().values
    dist_freud = torch.norm(graph_freud.edge_attr, dim=1).sort().values
    
    assert torch.allclose(dist_ovito, dist_freud, atol=1e-4)

def test_knn_selection():
    pipeline = import_file("tests/data.xyz")
    data = pipeline.compute()
    
    # Select 2 atoms
    selection = torch.tensor([0, 2])
    k = 2
    knn = PeriodicKNN(k=k)
    
    graph = knn.convert(data, selection=selection, backend="ovito")
    
    # x should contain ALL atoms
    assert graph.x.shape[0] == data.particles.count
    # num_nodes should be total atoms
    assert graph.num_nodes == data.particles.count
    
    # edge_index should use global indices from selection
    assert torch.all(torch.isin(graph.edge_index[0].unique(), selection))
    assert graph.edge_index.shape[1] == len(selection) * k
