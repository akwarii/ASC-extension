# Atomic Structure Classification

This is a *OVITO* wrapper for the [Atomic Structure Classification library](https://github.com/akwarii/Lightning-CEGANN2).

## Description

<!-- TODO description of the project -->

\[[Full description]\]

## Parameters

| GUI name                      | Python name     | Description                                                                                                                                                                                                                                                            | Default            |
| ----------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **Model file path**           | `model_path`    | Allows you to define a custom PyTorch model. The model will be loaded from the path entered. When set to `None`, the modifier will not perform any computation.                                                                                                        | `None`             |
| **Device**                    | `device`        | Allows you to select your computing device from: "cpu", "cuda", "mps". Only available devices will be shown. Please read the "Installation" section for additional information. Default to the available accelerator if available (i.e., `cuda` or `mps`), else `cpu`. | `cpu`/`mps`/`cuda` |
| **Batch size exponent (2^x)** | `_exponent`     | Allows you to define the batch size (number of atoms) used by the model at once. A larger number means more memory is used but will lead to faster results. We enforce the batch size to be a power of two in order to ensure optimal inference speed.                 | `10`               |
| **Batch size**                | `batch_size`    | Read-only parameters that allows the user to know its current batch size without computing `2^x`.                                                                                                                                                                      | `1024`             |
| **Data loading workers**      | `num_workers`   | Allows you to define the number of parallel processes used to load and transfer data from the CPU to the `device`. A higher number of workers implies a larger memory consumption. Set to `0` to disable multi-processing.                                             | `cpu count`        |
| **Only selected**             | `only_selected` | Apply the modifier only to the selected particles. Following the convention set by other modifiers, even atoms that are not selected will be used as neighbors.                                                                                                        | `False`            |

## Example

<!-- TODO add visual example -->

\[[Usage example]\]

## Installation

- OVITO Pro [integrated Python interpreter](https://docs.ovito.org/python/introduction/installation.html#ovito-pro-integrated-interpreter):

  ```
  ovitos -m pip install --user git+https://github.com/akwarii/ASC-extension.git
  ```

  The `--user` option is recommended and [installs the package in the user's site directory](https://pip.pypa.io/en/stable/user_guide/#user-installs).

- Other Python interpreters or Conda environments:

  ```
  pip install git+https://github.com/akwarii/ASC-extension.git
  ```

Due to ~~skill~~ technical issues with [pyg-lib](https://github.com/pyg-team/pyg-lib) installation, we are currently shipping the CPU version of [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io). The CUDA accelerated version will be available once this issue is solved or if installed manually.

## Technical information / dependencies

Tested on Linux (Debian 12) and MacOS:

- ovito==3.15.0
- torch==2.10.0
- torch-geometric==2.7.0

Tests were **not** performed on Windows.

## Contact

[Gaël Huynh](mailto:gael.huynh@gmail.com)

[Dylan Bissuel](mailto:dylan.bissuel@univ-lyon1.fr)
