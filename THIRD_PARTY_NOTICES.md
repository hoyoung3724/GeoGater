# Third-party notices

GeoGater is released under the MIT License (see `LICENSE`). Parts of the code are adapted
from the projects below, which keep their original licenses.

| Files | Source | License |
|---|---|---|
| `geogater/backbones/g3.py` (model definition) | [G3](https://github.com/Applied-Machine-Learning-Lab/G3) | Apache-2.0 ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)) |
| `geogater/backbones/georanker.py`, `pipeline/georanker_rewards.py` (reward model and prompts) | [GeoRanker](https://github.com/Applied-Machine-Learning-Lab/GeoRanker) | Apache-2.0 ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)) |
| `pipeline/g3_rag.py`, `pipeline/build_index.py` (RAG prompt, index construction) | [G3](https://github.com/Applied-Machine-Learning-Lab/G3) | Apache-2.0 ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)) |
| `geogater/backbones/rff/` | [random-fourier-features-pytorch](https://github.com/jmclong/random-fourier-features-pytorch) | MIT ([`geogater/backbones/rff/LICENSE`](geogater/backbones/rff/LICENSE)) |

The Apache-2.0 files were modified: training-only code was removed and the code was
reorganized for this repository.

Released checkpoints and datasets (G3 checkpoint, GeoRanker LoRA, MP16-Pro, IM2GPS3K, YFCC4K,
YFCC26K) are not redistributed here and remain subject to their own terms. `data_splits/`
contains only image ids, coordinates and scene labels of the query images used in our
experiments.
