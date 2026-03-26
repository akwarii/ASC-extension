#### Atomic Structure Classification ####
# Description of your Python-based modifier.

import json
import os
from collections.abc import Generator

import torch
from torch import Tensor
from torch.export.passes import move_to_device_pass
from ovito.data import DataCollection
from ovito.modifiers import ExpandSelectionModifier
from ovito.pipeline import ModifierInterface
from ovito.traits import FilePath
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from traits.api import Bool, Enum, Property, Range, cached_property, observe

#! Note to devs: Do not use assertions in this code as they will make OVITO crash when raised.
#! Instead, raise appropriate exceptions with informative error messages.

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
]  # fmt: off
atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}


class PeriodicKNN:
    """Utility class to convert an OVITO DataCollection to a PyG Data object
    using periodic KNN graph construction.

    Args:
        num_neighbors: The number of nearest neighbors to connect to each atom in the graph.
    """

    def __init__(self, num_neighbors: int = 20) -> None:
        if num_neighbors < 1:
            raise ValueError("The number of neighbors must be greater than 0.")

        self.num_neighbors = num_neighbors

    @staticmethod
    def _get_atomic_numbers(data: DataCollection) -> Tensor:
        """Convert a tensor of type ids to atomic numbers using a mapping.

        Args:
            data: The OVITO DataCollection object.

        Returns:
            A tensor of shape (num_atoms,) containing the atomic numbers of the atoms.
        """
        ptypes = data.particles_.particle_types_
        type_mapper = {t.id: atomic_numbers.get(t.name, 0) for t in ptypes.types}
        type_id = torch.from_numpy(ptypes[...]).long()

        max_type_id = int(type_id.max().item())
        mapping_tensor = torch.zeros(max_type_id + 1, dtype=torch.long)

        for t_id, z in type_mapper.items():
            mapping_tensor[t_id] = z

        return mapping_tensor[type_id]

    def _get_graph_data(
        self, atoms: DataCollection, selection: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        from ovito.data import NearestNeighborFinder

        x = self._get_atomic_numbers(atoms)[selection]

        finder = NearestNeighborFinder(self.num_neighbors, atoms)
        indices, deltas = finder.find_all(indices=selection.tolist())

        # q_idx represents the central atoms (query points)
        # p_idx represents the neighbor atoms
        q_idx = (
            torch.arange(selection.shape[0]).view(-1, 1).expand(-1, self.num_neighbors).flatten()
        )
        p_idx = torch.from_numpy(indices).flatten().long()

        edge_index = torch.stack([q_idx, p_idx])

        # edge attributes are the wrapped displacement vectors
        edge_attr = torch.from_numpy(deltas).flatten(0, 1).float()

        return x, edge_index, edge_attr

    def convert(
        self,
        ovito_data: DataCollection,
        selection: Tensor | None = None,
    ) -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            atoms_repr: An OVITO DataCollection.
            mask: A boolean tensor indicating which atoms to include in the graph.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        if selection is None:
            selection = torch.arange(ovito_data.particles.count)

        x, edge_index, edge_attr = self._get_graph_data(ovito_data, selection)

        pyg_data = Data(
            num_nodes=selection.numel(),
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return pyg_data


class ASCModifier(ModifierInterface):
    ckpt_file = FilePath(
        label="Model file",
        ovito_file_exists=True,
        ovito_file_filter=[
            "PyTorch model (*.pt2)",
            "All files (*)",
        ],
    )

    if torch.cuda.is_available():
        device = Enum("cuda", ["cuda", "cpu"], label="Device")
    elif torch.backends.mps.is_available():
        device = Enum("mps", ["mps", "cpu"], label="Device")
    else:
        device = "cpu"

    _is_compiled = False
    should_compile = Bool(True, label="Model compilation")

    # Workaround for the fact that Range traits don't support power of 2 step values.
    _exponent = Range(low=0, value=10, label="Batch size exponent (2^x)")
    batch_size = Property(observe="_exponent", label="Batch size")

    __cpu_count = os.cpu_count() or 0
    num_workers = Range(
        low=0,
        high=__cpu_count,
        value=__cpu_count,
        label="Data loading workers",
    )

    only_selected = Bool(False, label="Only selected")

    @observe("ckpt_file")
    def _on_ckpt_file_change(self, event) -> None:
        if not self.ckpt_file:
            del self._program
            del self.model
            del self.metadata
            return
        self.load_model()

    @observe("device")
    def _on_device_change(self, event) -> None:
        if hasattr(self, "model"):
            self._program = move_to_device_pass(self._program, self.device)
            self.model = self._program.module()

    # TODO currently, a compiled model is not replaced with an uncompiled one if the user unchecks the "Model compilation" checkbox.
    @observe("should_compile")
    def _on_compile_change(self, event) -> None:
        if hasattr(self, "model"):
            self.compile_model()

    @cached_property
    def _get_batch_size(self) -> int:
        return int(2**self._exponent)

    def _validate_metadata(self) -> None:
        required_keys = set(["num_neighbors", "num_layers"])
        actual_keys = set(self.metadata.keys())

        missing_keys = required_keys - actual_keys
        if missing_keys:
            raise ValueError(f"Metadata is missing required keys: {missing_keys}")

        strictly_positive_keys = ["num_neighbors", "num_layers"]
        for key in strictly_positive_keys:
            if self.metadata[key] <= 0:
                raise ValueError(
                    f"Invalid metadata: '{key}' must be strictly positive "
                    f"(got {self.metadata[key]})."
                )

    def _model_warmup(self, optimized_module, *args) -> None:
        example = (
            torch.randint(low=1, high=118, size=(10,), device=self.device),
            torch.randint(low=0, high=10, size=(2, 30), device=self.device, dtype=torch.long),
            torch.randn(30, 3, device=self.device),
        )

        with torch.no_grad():
            for _ in range(5):
                optimized_module(*example)

    def compile_model(self) -> None:
        if not self.should_compile:
            self._is_compiled = False
            return

        if self.device != "cuda" or torch.cuda.get_device_capability() < (7, 0):
            self._is_compiled = False
            return

        self.model.compile(fullgraph=True, dynamic=True)
        self._model_warmup(self.model)
        self._is_compiled = True

    def load_model(self) -> None:
        extra_files = {"metadata.json": ""}
        program = torch.export.load(self.ckpt_file, extra_files=extra_files)

        self.metadata = json.loads(extra_files["metadata.json"])
        self._validate_metadata()

        # Move the loaded program to the specified device before extracting the module
        self._program = move_to_device_pass(program, self.device)
        self.model = self._program.module()

        self.compile_model()

    @torch.inference_mode()
    def inference(self, model, loader) -> Generator[float, None, Tensor]:
        """Run inference on a single DataCollection object and return the predicted class indices for
        each particle.

        Args:
            model: The trained PyTorch Lightning Module for prediction.
            data: The input DataCollection object containing the particle data to predict on.
        """
        graph_preds = []
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)

            with torch.autocast(device_type=self.device):
                all_logits: Tensor = model(batch.x, batch.edge_index, batch.edge_attr)
                target_logits = all_logits[: batch.batch_size]
                out = torch.argmax(target_logits, dim=-1)

            graph_preds.append(out.to("cpu", non_blocking=True))
            yield (i / len(loader))

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        else:
            pass

        predictions = torch.cat(graph_preds, dim=0)

        return predictions

    # TODO add topk/confidence threshold options to return multiple predictions per particle
    # This option should add more properties:
    # - "ASC Confidence": the confidence score of the prediction (e.g. softmax probability)
    # - "ASC Top K Structure": the top K predicted structure types (e.g. as a list of integers)
    # - "ASC Structure Type": the predicted structure type (e.g. as an integer index)
    def modify(self, data: DataCollection, frame: int, **kwargs) -> Generator[float, None, None]:
        if not self.ckpt_file:
            return

        if data.cell is None:
            raise ValueError("The input structure must have a defined unit cell.")

        num_neighbors: int = self.metadata["num_neighbors"]
        num_layers: int = self.metadata["num_layers"]

        selection = torch.arange(data.particles.count)
        expanded_selection = torch.arange(data.particles.count)
        if self.only_selected:
            # No selection modifier is applied
            if data.particles.selection is None:
                return

            mask = torch.from_numpy(data.particles.selection[...]).bool()

            # There is a selection modifier, but no particles are selected
            if not mask.sum():
                return

            selection = torch.argwhere(mask).squeeze()

            # Expand the selection to include neighbors of selected particles
            # This avoid edge cases classifying isolated/surface atoms
            data.apply(
                ExpandSelectionModifier(
                    mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                    num_neighbors=num_neighbors,
                )
            )
            expanded_mask = torch.from_numpy(data.particles.selection[...]).bool()
            expanded_selection = torch.argwhere(expanded_mask).squeeze()

        # TODO cache the graph if the structure doesn't change
        knn = PeriodicKNN(num_neighbors=num_neighbors)
        graph = knn.convert(data, selection=expanded_selection)

        loader = NeighborLoader(
            graph,
            num_neighbors=[num_neighbors] * num_layers,
            input_nodes=selection,
            batch_size=min(self.batch_size, selection.numel()),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0 and self.device == "cuda",
            multiprocessing_context="fork" if self.num_workers > 0 else None,
        )

        # default to -1 for unpredicted particles
        results = torch.full((data.particles.count,), -1, dtype=torch.long)
        out = yield from self.inference(self.model, loader)
        results[selection] = out

        # Placeholder for logic to apply results to the data collection
        data.particles_.create_property("ASC Structure Type", dtype=int, data=results)
