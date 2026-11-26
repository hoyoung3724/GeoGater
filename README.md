# GeoGater: Meta-Feature Guided Adaptive Routing for Worldwide Image Geolocalization

**Hoyoung Choi, Sungwon Kim, Jiwan Kim, Chanyoung Park**
KAIST, Daejeon, South Korea

Official implementation of **GeoGater** (WSDM 2026). Paper: *coming soon*.

Generation-based geolocalization pipelines such as [G3](https://arxiv.org/abs/2405.14702) and
[GeoRanker](https://arxiv.org/abs/2505.13731) run **both** a zero-shot LVLM path and a
retrieval path for every query and then spend most of their compute on a downstream LVLM stage
(RAG generation for G3, reward scoring for GeoRanker). GeoGater is a lightweight add-on (2.2M
parameters) that predicts, per query, which path will localize the query better and invokes the
downstream LVLM stage **only for that path**. The backbone is not modified or retrained.

The router reads a small set of **meta-features** computed from the inference process:

| Path | Reliability proxy | Features |
|---|---|---|
| Zero-shot | verbalized confidence of N sampled LVLM answers | `conf_*`, `zs_align*` |
| Retrieval | spatial consistency of the top-K retrieved coordinates (DBSCAN, ε = 25 km) | `cluster_*`, `pairwise_*`, `nn_*` |
| Both | G3 image / text / GPS similarities | `img*`, `text*`, `gps*`, `cand_sum*` |

It is trained with BCE on the oracle path label plus a distance-aware loss that weights each
decision by the geographic error it would cause.

## Results

Recall@distance (%) with MP16-Pro as the reference database. Router trained on YFCC26K.

| Method | IM2GPS3K 1km | 25km | 200km | 750km | 2500km | YFCC4K 1km | 25km | 200km | 750km | 2500km |
|---|---|---|---|---|---|---|---|---|---|---|
| G3 | 16.35 | 41.34 | 56.29 | 71.97 | 85.22 | **23.79** | **37.03** | 49.75 | 66.64 | **81.42** |
| G3 + GeoGater | **17.07** | **43.14** | **57.65** | **73.46** | **86.03** | 22.99 | 36.91 | **50.20** | **67.09** | 81.11 |
| GeoRanker | 18.45 | 43.98 | 59.23 | 74.14 | 87.09 | **33.30** | 44.41 | 55.84 | 71.63 | 84.05 |
| GeoRanker + GeoGater | **18.97** | **45.33** | **59.95** | **75.14** | **87.77** | 33.22 | **45.20** | **56.58** | **72.38** | **84.09** |

Downstream LVLM calls per query:

| | IM2GPS3K | YFCC4K |
|---|---|---|
| G3 → + GeoGater | 20.00 → 13.74 (−31.3%) | 20.00 → 15.43 (−22.9%) |
| GeoRanker → + GeoGater | 15.00 → 8.82 (−41.2%) | 19.00 → 11.89 (−37.4%) |

## Installation

```bash
git clone https://github.com/hoyoung3724/GeoGater.git && cd GeoGater
conda create -n geogater python=3.10 -y && conda activate geogater
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Only needed for the GeoRanker reward step:
pip install flash-attn --no-build-isolation
```

Training the router (step 7) only needs `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`.

## Quick start: reproduce Table 1 from released features

`precomputed/` contains the meta-features and routing targets for every setting in Table 1
(about 15 MB), so the router can be trained and evaluated without any LVLM calls.
Each setting (5 seeds) takes about a minute on one GPU.

```bash
bash scripts/reproduce_table1.sh
# or a single setting:
python train.py --work_dir precomputed --backbone georanker --rank_k 12 --zs_k 3 --eval_dataset im2gps3k
```

Example output:

```
Method                    R@1     R@25    R@200    R@750   R@2500      Med     Mean
-----------------------------------------------------------------------------------
Zero-shot               16.22    43.84    59.09    75.31    87.52    73.53  1261.90
Retrieval               17.65    42.08    57.06    72.87    86.39    94.66  1367.33
GeoRanker               18.45    43.98    59.23    74.14    87.09    72.49  1302.13
Oracle                  22.99    50.55    66.87    81.61    92.53    22.86   807.72
GeoGater (ours)         18.97    45.33    59.95    75.14    87.77    60.64  1248.19
...
Downstream LVLM calls / query: backbone 15.00 -> GeoGater 8.82 (-41.17%)
```

Runs are deterministic: fixed seeds `{42, 101, 7, 99, 12}` and a fixed train/validation split
(`--split_seed 123`). The reported numbers are the mean over the 5 seeds.

| Setting | `--backbone` | `--rank_k` (K) | `--zs_k` (N) |
|---|---|---|---|
| G3, IM2GPS3K / YFCC4K | `g3` | 15 | 5 |
| GeoRanker, IM2GPS3K | `georanker` | 12 | 3 |
| GeoRanker, YFCC4K | `georanker` | 14 | 5 |

## Full pipeline

### Data

Place the data under `data/` (or pass `--data_root`):

```
data/
├── mp16/       mp16.csv, mp16-images.tar, tar_index.pkl     # MP16-Pro reference database
├── yfcc26k/    yfcc26k_places365.csv, yfcc26ktest/           # router training set
├── im2gps3k/   im2gps3k_places365.csv, im2gps3ktest/
└── yfcc4k/     yfcc4k_places365.csv,  yfcc4ktest/
```

* MP16-Pro (images, `tar_index.pkl`, metadata): [Jia-py/MP16-Pro](https://huggingface.co/datasets/Jia-py/MP16-Pro)
* IM2GPS3K images: [im2gps3ktest.zip](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip)
* YFCC4K images: [yfcc4k.zip](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip)
* YFCC26K images: the YFCC25600 split of [GeoEstimation](https://github.com/TIBHannover/GeoEstimation)

The exact query lists we used (IM2GPS3K: 2,997, YFCC4K: 4,364, YFCC26K: 21,802 images) are in
[`data_splits/`](data_splits). Copy each to `data/<dataset>/<dataset>_places365.csv`. Every
pipeline step aligns queries by row order, so use these files as-is.

### Checkpoints

```
checkpoints/
├── g3.pth               # official G3 retriever (both backbones)
└── georanker_lora/      # GeoRanker reward-model LoRA (adapter_config.json, adapter_model.safetensors)
```

* `g3.pth`: the official G3 checkpoint from [Jia-py/G3-checkpoint](https://huggingface.co/Jia-py/G3-checkpoint).
  The same repository hosts a prebuilt MP16-Pro FAISS index (`g3.index`). To skip index building,
  place it at `outputs/<backbone>/index/mp16.index`.
* `georanker_lora/`: the LoRA weights released with GeoRanker; the base model is
  `Qwen/Qwen2-VL-7B-Instruct`.

### Steps

Every step is a module under `pipeline/` and writes into `--work_dir` (default `outputs/`).
The full command sequence is in [`scripts/run_g3.sh`](scripts/run_g3.sh) and
[`scripts/run_georanker.sh`](scripts/run_georanker.sh).

| # | Command | Output |
|---|---|---|
| 1 | `python -m pipeline.build_index --backbone {g3,georanker} --g3_ckpt ...` | FAISS index; top-100 and inverse top-100 DB rows per query |
| 2 | `python -m pipeline.zero_shot --dataset ...` | N = 5 GPT-4o samples of (lat, lon, confidence) per query |
| 3 | `python -m pipeline.build_candidates --backbone ... --dataset ...` | retrieval candidates H_Ret + zero-shot coordinates H_ZS |
| 4a | `python -m pipeline.g3_rag --dataset ...` | G3 RAG generations (k = 5, 10, 15) |
| 4b | `python -m pipeline.georanker_rewards --dataset ...` | GeoRanker reward for every candidate |
| 5 | `python -m pipeline.extract_features --backbone ... --rank_k K --zs_k N --g3_ckpt ...` | meta-features |
| 6 | `python -m pipeline.build_targets --backbone ... --rank_k K --zs_k N` | per-path selected-candidate errors and oracle labels |
| 7 | `python train.py --backbone ... --rank_k K --zs_k N` | trained router and evaluation |

Steps 2 and 4a call the OpenAI API (`export OPENAI_API_KEY=...`); both resume from partial
output. Step 4 is the backbone's own downstream stage: it is run on all queries **once**, to
build the routing labels. At deployment, GeoGater runs it only for the path it selects.

The work directory looks like this:

```
outputs/
├── zero_shot/<dataset>.csv
├── g3/{index, candidates, rag, features, checkpoints}/
└── georanker/{index, candidates, rewards, features, checkpoints}/
```

### Router hyperparameters

Defaults in `train.py` are the paper settings (Supp. A): 3-layer MLP (hidden 1024,
LayerNorm, ReLU, dropout 0.3), AdamW (lr 1e-4, weight decay 1e-4), batch 256, up to 100
epochs with early stopping (patience 10) on validation geometric-mean error, loss
`L = 5 · L_dist + L_BCE` (`--loss ltce ce --loss_w 5 1`), temperature τ = 1.

## Repository structure

```
GeoGater/
├── geogater/
│   ├── router.py            # GeoGater MLP + distance-aware routing loss
│   ├── meta_features.py     # meta-feature definitions (T(S), cluster consensus, compactness)
│   ├── paths.py             # work-directory layout
│   ├── utils.py             # dataset registry, geodesic helpers, metrics
│   └── backbones/
│       ├── g3.py            # G3 retriever + checkpoint loader
│       ├── georanker.py     # GeoRanker reward model (inference)
│       └── rff/             # random Fourier features for the G3 location encoder
├── pipeline/                # steps 1-6
├── train.py                 # step 7
├── precomputed/             # released meta-features + targets for Table 1
├── data_splits/             # exact query lists (row order matches precomputed/)
├── scripts/
└── licenses/                # third-party license texts
```

## Acknowledgements

This code builds on [G3](https://github.com/Applied-Machine-Learning-Lab/G3),
[GeoRanker](https://github.com/Applied-Machine-Learning-Lab/GeoRanker) and
[random-fourier-features-pytorch](https://github.com/jmclong/random-fourier-features-pytorch).
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the adapted files and their licenses.

## License

MIT (see [`LICENSE`](LICENSE)), except for the third-party code listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

Coming soon.

## Contact

For questions or bug reports, please open a [GitHub issue](https://github.com/hoyoung3724/GeoGater/issues)
or contact Hoyoung Choi (chy3724@kaist.ac.kr).
