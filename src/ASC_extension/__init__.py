#### Python Modifier Name ####
# Description of your Python-based modifier.

import math
from collections.abc import Generator

import torch
import torch.nn as nn
from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from ovito.traits import FilePath
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter
from traits.api import Bool, Enum, Property, Range, cached_property, observe

#TODO currently, dependencies from the main code are not used to keep a standalone modifier
# Instead, needed classes are redefined in this file. This is not ideal, but it allows us to
# test the modifier without needing to export a model eg, with torchscript. The best solution
# would be to find a way to import the model definition from the main code without importing
# all dependencies, but this is non-trivial and may require changes to the main code structure.

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
]
atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}


def type_id_to_atomic_number(type_id: torch.Tensor, type_mapping: dict[int, int]) -> torch.Tensor:
    """Convert a tensor of type ids to atomic numbers using a mapping.

    Args:
        type_id: A tensor of shape (num_atoms,) containing the type ids of the atoms.
        type_mapping: A dictionary mapping type ids to atomic numbers.

    Returns:
        A tensor of shape (num_atoms,) containing the atomic numbers of the atoms.
    """
    max_type_id = int(type_id.max().item())
    mapping_tensor = torch.zeros(max_type_id + 1, dtype=torch.long)
    for t_id, z in type_mapping.items():
        mapping_tensor[t_id - 1] = z

    atomic_numbers = mapping_tensor[type_id]

    return atomic_numbers


class PeriodicKNN:
    """Test of a periodic knn using Freud."""

    def __init__(self, k: int = 20, **kwargs) -> None:
        if k < 1:
            raise ValueError("The number of neighbors must be greater than 0.")

        self.k = k

    def _get_graph_data_freud(self, atoms: DataCollection) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from freud import AABBQuery, Box
        
        # Map OVITO particle types to atomic numbers
        ptypes = atoms.particles_.particle_types_
        type_mapper = {t.id: atomic_numbers.get(t.name, 0) for t in ptypes.types}
        type_id = torch.from_numpy(ptypes[...]).long()

        x = type_id_to_atomic_number(type_id, type_mapper)

        # Extract data from OVITO DataCollection object
        pos = atoms.particles.positions[...]
        cell_matrix = atoms.cell[...][:3, :3].T

        # Create Freud Box for PBC handling
        box = Box.from_matrix(cell_matrix)
        box.periodic = atoms.cell.pbc

        # Perform knn query
        nq = AABBQuery(box, pos)
        nlist = nq.query(pos, dict(num_neighbors=self.k, exclude_ii=True)).toNeighborList()

        # Build edge index and attributes
        q_idx = torch.from_numpy(nlist.query_point_indices.copy()).long()
        p_idx = torch.from_numpy(nlist.point_indices.copy()).long()

        edge_index = torch.stack([q_idx, p_idx])

        pos_t = torch.from_numpy(pos).float()
        diff_t = pos_t[p_idx] - pos_t[q_idx]
        wrapped_diff = box.wrap(diff_t.numpy())
        edge_attr = torch.from_numpy(wrapped_diff).float()

        return x, edge_index, edge_attr

    def _get_graph_data_ovito(self, atoms: DataCollection) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from ovito.data import NearestNeighborFinder
        
        # Map OVITO particle types to atomic numbers
        ptypes = atoms.particles_.particle_types_
        type_mapper = {t.id: atomic_numbers.get(t.name, 0) for t in ptypes.types}
        type_id = torch.from_numpy(ptypes[...]).long()

        x = type_id_to_atomic_number(type_id, type_mapper)

        finder = NearestNeighborFinder(self.k, atoms)
        indices, deltas = finder.find_all()

        # Build edge index
        # q_idx represents the central atoms (query points)
        # p_idx represents the neighbor atoms
        q_idx = torch.arange(atoms.particles.count).view(-1, 1).expand(-1, self.k).flatten()
        p_idx = torch.from_numpy(indices).flatten().long()

        edge_index = torch.stack([q_idx, p_idx])

        # edge attributes are the wrapped displacement vectors
        edge_attr = torch.from_numpy(deltas).flatten(0, 1).float()

        return x, edge_index, edge_attr

    def convert(self, ovito_data: DataCollection, backend: str = "ovito") -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            atoms_repr: An OVITO DataCollection.
            backend: The backend to use for KNN computation.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        assert ovito_data.cell is not None, "The input structure must have a cell defined."

        knn_method = {
            "freud": self._get_graph_data_freud,
            "ovito": self._get_graph_data_ovito,
        }

        x, edge_index, edge_attr = knn_method[backend](ovito_data)

        pyg_data = Data(
            num_nodes=ovito_data.particles.count,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return pyg_data


class RadialBesselBasis(nn.Module):
    r"""Radial Bessel basis, as proposed in Gasteiger et al (2022). Directional Message Passing for
    Molecular Graphs (arXiv:2003.03123).

    Args:
        num_radial: The number of radial basis functions.
        stop: The cutoff value for scaling the distance.
        trainable: Whether to train the frequencies :math:`n \pi`.
    """

    def __init__(
        self,
        num_radial: int = 8,
        stop: float = 6.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial
        self.trainable = trainable
        self.r_max = stop

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)
        self.freq.requires_grad_(self.trainable)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate Bessel Basis for input x.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Radial Bessel basis (shape [num_edges, 1, num_radial]).
        """
        inv_r_max = 1.0 / self.r_max
        prefactor = 2.0 * inv_r_max
        x_expanded = x.unsqueeze(-1)
        sin_arg = self.freq * x_expanded * inv_r_max
        return torch.sin(sin_arg).mul_(prefactor).div_(x_expanded)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_radial={self.num_radial}, stop={self.r_max})"


class PaiNNRadial(nn.Module):
    """Rotation invariant radial filter that processes the distance between neighbors.

    Args:
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance for the radial basis functions.
        hidden_channels: Dimensionality of the hidden scalar features.
    """

    def __init__(self, num_radial: int, hidden_channels: int, cutoff: float = 6.0) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels

        self.rbf_filter = nn.Sequential(
            RadialBesselBasis(num_radial, cutoff),
            Linear(num_radial, 3 * hidden_channels),  # we split into 3 parts later
        )

    def forward(self, dist_mag: Tensor) -> Tensor:
        """Forward pass for the radial filter.

        Args:
            dist_mag: Distance magnitudes (shape [num_edges, 1]).
        """
        return self.rbf_filter(dist_mag)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.rbf_filter)[1:-1]})"


class PaiNNMessage(nn.Module):
    """PaiNN message block. It processes scalar and vector features together, generating messages
    for both streams.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        dropout: Dropout rate for the internal MLP.
        scale_factor: Scaling factor for the message passing. To avoid exploding gradients, it is
            recommended to set this to 1 / num_neighbors.
    """

    def __init__(
        self, hidden_channels: int, dropout: float = 0.1, scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.scale_factor = scale_factor
        self.dropout = dropout

        self.phi = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, 3 * hidden_channels),
        )
        self.rms_norm = nn.RMSNorm(hidden_channels)

    def forward(
        self, s: Tensor, v: Tensor, edge_index: Tensor, rbf_filter: Tensor, edge_vector: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the message block. The variable names are chosen to reflect the
        original PaiNN paper.

        Args:
            s: Scalar features (shape [num_nodes, 1, hidden_channels]).
            v: Vector features (shape [num_nodes, 3, hidden_channels]).
            edge_index: Edge indices (shape [2, num_edges]).
            rbf_filter: Radial filter (shape [num_edges, 1, 3 * hidden_channels]).
            edge_vector: Distance unit vectors (shape [num_edges, 3]).
        """
        assert rbf_filter.shape[-1] == 3 * self.hidden_channels, (
            "Edge filter output dimension must be 3x hidden_channels"
        )

        i, j = edge_index

        s_norm = self.rms_norm(s)

        # since linear and gather are commutative, we can apply the linear layer first
        phi_s = self.phi(s_norm)
        filter = phi_s[j] * rbf_filter

        # split into scalar, vector, and gate
        m_s, m_vv, m_vs = torch.chunk(filter, chunks=3, dim=-1)

        # Scalar message
        ds = scatter(m_s, i, dim=0, dim_size=s.size(0), reduce="sum")

        # Vector message
        # gate = v[j] * m_vv + m_vs * edge_vector[..., None]
        gate = torch.einsum("edc, ec -> edc", v[j], m_vv.squeeze(1))
        gate.addcmul_(edge_vector.unsqueeze(-1), m_vs)

        dv = scatter(gate, i, dim=0, dim_size=v.size(0), reduce="sum")

        # Residual
        s = s + ds * self.scale_factor
        v = v + dv * self.scale_factor

        return s, v


class PaiNNUpdate(nn.Module):
    """PaiNN update block. It takes the output from the message block and updates the scalar and
    vector features.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        dropout: Dropout rate for the internal MLP.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.update_net = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, 3 * hidden_channels),
        )
        self.v_proj = Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.rms_norm = nn.RMSNorm(hidden_channels)

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the update block.

        Args:
            s: Scalar features (shape [num_nodes, 1, hidden_channels]).
            v: Vector features (shape [num_nodes, 3, hidden_channels]).
        """
        s_norm = self.rms_norm(s)

        u, w = torch.chunk(self.v_proj(v), chunks=2, dim=-1)
        w_norm = w.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()

        context = torch.cat([s_norm, w_norm], dim=-1)
        filter = self.update_net(context)

        a_ss, a_vv, a_sv = torch.chunk(filter, chunks=3, dim=-1)

        # scaling functions are used as nonlinearity
        dv = a_vv * u

        # ds = a_ss + a_sv * torch.sum(u * w, dim=1, keepdim=True)
        uw_dot = torch.einsum("ndc, ndc -> nc", u, w).unsqueeze(1)
        ds = torch.addcmul(a_ss, a_sv, uw_dot)

        # Residuals
        s = s + ds
        v = v + dv

        return s, v


class PaiNNHead(nn.Module):
    """PaiNN head block. It takes the scalar features and produces predictions.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        num_classes: Number of output classes.
        dropout: Dropout rate for the internal MLP.
    """

    def __init__(self, hidden_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_classes = out_channels

        self.mlp = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, s: Tensor) -> Tensor:
        """Forward pass for the head block. It only uses the scalar features for prediction.

        Args:
            s: Scalar features (shape [num_nodes, hidden_channels]).
        """
        return self.mlp(s)


class PaiNN(nn.Module):
    """PaiNN model adapted for classification tasks. It simply removes the final reduce to keep the
    atomic representation.

    Args:
        num_classes: Number of output classes.
        num_species: Number of unique atomic species (default 119 for all elements).
        num_radial: Number of radial basis functions (default 4).
        num_layers: Number of message-passing layers (default 3).
        hidden_channels: Dimensionality of the hidden scalar features (default 128).
        dropout: Dropout rate for the internal MLPs (default 0.1).
        scale_factor: Scaling factor for the message passing (default 1.0). To avoid exploding
            gradients, it is recommended to set this to 1 / num_neighbors.
    """

    def __init__(
        self,
        out_channels: int,
        num_species: int = 119,
        num_radial: int = 4,
        num_layers: int = 1,
        hidden_channels: int = 16,
        dropout: float = 0.1,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.scale_factor = scale_factor

        self.embedding = nn.Embedding(num_species, hidden_channels)
        self.rbf = PaiNNRadial(num_radial, hidden_channels, cutoff=6.0)

        # Separate message and update blocks for better modularity
        self.message_blocks = nn.ModuleList(
            [PaiNNMessage(hidden_channels, dropout, scale_factor) for _ in range(num_layers)]
        )
        self.update_blocks = nn.ModuleList(
            [PaiNNUpdate(hidden_channels, dropout) for _ in range(num_layers)]
        )
        self.head = PaiNNHead(hidden_channels, out_channels, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Forward pass for the PaiNN model.

        Args:
            x: Node features (shape [num_nodes, num_features]).
            edge_index: Edge indices (shape [2, num_edges]).
            edge_attr: Edge features (distance vectors) (shape [num_edges, 3]).
        """
        # dist_mag = torch.linalg.norm(edge_attr, dim=1, keepdim=True)
        dist_mag = edge_attr.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        edge_unit_vec = edge_attr / dist_mag

        s = self.embedding(x).unsqueeze(1)
        v = torch.zeros(s.size(0), 3, s.size(2), device=s.device)

        rbf_filter = self.rbf(dist_mag)
        for message, update in zip(self.message_blocks, self.update_blocks):
            s, v = message(s, v, edge_index, rbf_filter, edge_unit_vec)
            s, v = update(s, v)

        out = self.head(s.squeeze(1))

        return out


class AtomicStructureClassification(ModifierInterface):
    ckpt_file = FilePath(
        label="Checkpoint file",
        ovito_file_exists=True,
        ovito_file_filter=["PyTorch checkpoint files (*.ckpt)", "All files (*)"],
    )

    device = Enum("cuda", ["cuda", "cpu"], label="Device")
    should_compile = Bool(True, label="Model compilation")

    _exponent = Range(low=0, value=10, label="Batch size exponent (2^x)")
    batch_size = Property(observe="_exponent", label="Batch size")

    use_freud = Bool(False, label="Use Freud backend\n(recommended for\nlarge systems)")
    only_selected = Bool(False, label="Only selected")

    @observe("device, should_compile")
    def _validate_compilation_state(self, event):
        if self.should_compile:
            if self.device == "cpu":
                self.should_compile = False
                print("Compilation disabled: Only supported on CUDA.")
            elif torch.cuda.get_device_capability() < (7, 0):
                self.should_compile = False
                print("Compilation disabled: Requires Compute Capability 7.0+.")

    @cached_property
    def _get_batch_size(self) -> int:
        return int(2**self._exponent)

    def get_graph(self, data: DataCollection, num_neighbors: int) -> Data:
        backend = "freud" if self.use_freud else "ovito"
        graph = PeriodicKNN(k=num_neighbors).convert(data, backend=backend)
        return graph

    def load_model(self) -> None:
        #TODO find how to load the model with minimal dependencies. Options seems to be to export the use torchscript
        # or to import the model definition in this file, which is not ideal. For now we just fallback to a dummy model
        # to test the modifier workflow without needing to export a model.
        print("Model loading is not implemented. Using untrained model for testing.")
        try:
            model: torch.nn.Module = torch.load(self.ckpt_file, map_location=self.device, weights_only=False)
        except Exception:
            model = PaiNN(out_channels=10)

        if self.should_compile:
            model.compile(fullgraph=True, dynamic=True)

        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def inference(self, model, graph: Data) -> Generator[float, None, torch.Tensor]:
        """Run inference on a single DataCollection object and return the predicted class indices for
        each particle.

        Args:
            model: The trained PyTorch Lightning Module for prediction.
            data: The input DataCollection object containing the particle data to predict on.
        """
        num_layers: int = model.num_layers

        loader = NeighborLoader(
            graph,
            num_neighbors=[-1] * num_layers,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        graph_preds = []
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            with torch.autocast(device_type=self.device):
                all_logits: torch.Tensor = model(batch.x, batch.edge_index, batch.edge_attr)
                target_logits = all_logits[:batch.batch_size]
                out = torch.argmax(target_logits, dim=-1)
            graph_preds.append(out.to("cpu", non_blocking=True))
            yield (i / len(loader))

        torch.cuda.synchronize()
        predictions = torch.cat(graph_preds, dim=0)

        return predictions

    def modify(self, data: DataCollection, frame: int, **kwargs) -> Generator[float, None, None]:
        if self.only_selected:
            # Placeholder for logic to filter only selected particles
            pass

        # Don't run the modifier if no checkpoint file is provided
        if not self.ckpt_file:
            return
        
        if not hasattr(self, "model"):
            self.load_model()

        num_neighbors = 12 # placeholder, should be extracted from checkpoint
        graph = self.get_graph(data, num_neighbors=num_neighbors) #TODO cache the graph if the structure doesn't change
        results = yield from self.inference(self.model, graph)

        # Placeholder for logic to apply results to the data collection
        data.particles_.create_property("ASC Structure Type", dtype=int, data=results)
