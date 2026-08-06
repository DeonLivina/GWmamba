# gw-cmamba

Gravitational-wave transient classification (background / glitch / signal[/ blip]) from
strain + auxiliary witness channels, using Mamba-family state-space models. This repo
curates five model lineages of increasing sophistication, sharing one data pipeline.

## Structure

```
data_pipeline/     Real-data pipeline: fetch Omicron triggers, extract/whiten background
                    and glitch windows from raw strain, inject synthetic signals/blips,
                    leakage & coincidence sanity checks, visualization.
                      data_pipeline/viz/   Q-scan and timeseries plotting

simulation/         Synthetic data generation: waveform injection, witness-channel
                    synthesis/coupling, built on top of data_pipeline's real background.

common/             Shared code used by multiple model lineages:
                      special_loader.py   4-class (+blip) loader, 9 witness channels,
                                           tracks one specially-flagged candidate event.
                                           Used by models/cmamba only.
                      compact_loader.py   3-class loader (no blip class), different
                                           sampling strategy. Used by baseline/before_model/
                                           after_model/ssl_model.
                      losses.py           SupervisedSimCLRLoss (supervised contrastive loss)

models/
  baseline/          2 conv down-layers -> plain mamba_ssm.Mamba stack -> mean pool ->
                      linear classifier. Cross-entropy only. The floor every other
                      lineage here is compared against.
  before_model/       Two branches sharing funnel encoders: a SupCon "bypass" head
                      (strain/witness pooled straight to an 8D embedding, skipping Mamba)
                      and a "fusion" path (shared-weight Mamba over all 1+N streams ->
                      classifier). CE + 2x SupCon.
  after_model/        Evolution of before_model (originally "aux_model" -- renamed to
                      pair with before_model): both the SupCon and classifier heads now
                      read the SAME shared-trunk Mamba output (independently pooled),
                      adds an auxiliary strain-only CE head, and supports bidirectional
                      Mamba. CE + aux CE + 2x SupCon.
  ssl_model/          Self-supervised: LeJEPA-style latent next-token prediction over
                      fused strain+witness context, regularized with Sketched Isotropic
                      Gaussian Regularization (SIGReg), pretrained then fine-tuned.
                        pretrain.py   Stage 1: self-supervised backbone pretraining
                        finetune.py   Stage 2 (freeze backbone, train classifier head)
                                      + Stage 3 (eval + 8D corner-pairplot viz).
                                      Imports the backbone from pretrain.py.
  cmamba/             Channel-correlation-enhanced Mamba (Zeng et al., arXiv:2406.05316),
                      ported module-for-module from the official repo, applied to the
                      strain+witness setting via GDD-MLP cross-channel mixing and a
                      within-sample channel mixup.
                        cmamba_model.py            faithful hand-rolled selective scan
                        cmamba_mambassm_model.py   same architecture, mamba_ssm mixer
                        pscan.py                   parallel-scan op used by cmamba_model.py
                        cmamba_train.py             training script
                        cmamba_ablation_train.py    channel-mixing ablation (plain Mamba
                                                     vs. +GDD-MLP+mixup, witness zeroed
                                                     vs. real, shared train/val/test split)

data/
  configs/            YAML injection configs per detector (simulation/)
  raw_triggers/        H1_triggers.zip, L1_triggers.zip -- see "Data" below
```

Each `models/<lineage>/*.py` is run as a standalone script (`python models/cmamba/cmamba_train.py`)
from anywhere; a small `sys.path` shim at the top of each entry point adds `common/` so the shared
loaders/losses resolve without installing this as a package.

### Not included

Other experimental lineages exist in the author's private working copy (`dual_model`, an earlier
"V3" ablation pipeline, and a separate `uni_test/` tree with JEPA/lag-mixer/unified-trunk variants)
but were intentionally left out of this repo to keep it focused.

## Setup

```
pip install -r requirements.txt
```

`mamba-ssm` needs a CUDA-capable GPU to build/run (used by `baseline`, `before_model`, `after_model`,
`ssl_model`, and `models/cmamba/cmamba_mambassm_model.py`). `models/cmamba/cmamba_model.py` (the
faithful hand-rolled port) has no such dependency and runs on CPU, just slower.

## Data

**Shipped in this repo:**
- `data/configs/*.yaml` -- injection configs
- `data/raw_triggers/{H1,L1}_triggers.zip` -- Omicron trigger CSVs, filtered down to exactly the
  10 witness channels `common/special_loader.py` uses (`H1_WITNESS_CHANNELS`/`L1_WITNESS_CHANNELS`),
  zipped. This is enough to inspect/reuse those specific channels, but **not** enough to regenerate
  `full_data/` from scratch (see caveat below).

**Not shipped** (too large for git -- `full_data/H1`/`full_data/L1`, the whitened HDF5 output of
`data_pipeline/`, run ~2.1GB with individual files up to 416MB):
- Run `data_pipeline/get_triggers.py` to fetch the full raw Omicron trigger set for each detector,
  `data_pipeline/bg_triggers.py` + `data_pipeline/extract_background.py` /
  `data_pipeline/extract_glitches.py` to build background/glitch windows, and
  `data_pipeline/inject_signal.py` / `data_pipeline/inject_gaussians.py` (using `simulation/`) to
  build the signal/blip classes. `data_pipeline/ligo_loader.py` documents the expected
  `full_data/<DETECTOR>/` output layout that `common/special_loader.py` and `common/compact_loader.py`
  read from.

**Caveat:** the zipped trigger files here only contain the 5+5 witness-channel CSVs per detector --
they exclude each detector's `GDS-CALIB_STRAIN` (strain, not witness) trigger file, which
`data_pipeline/bg_triggers.py` and `data_pipeline/strain_witness_coincidence.py` need to fully
regenerate background windows from scratch. Re-run `data_pipeline/get_triggers.py` for the complete
raw set if you need a from-scratch rebuild.

**Known gap:** `data_pipeline/str_down.py` downloads raw L1 strain only; there is no working H1
equivalent (an `aux_h1_down.py`/`aux_l1_down.py` pair existed in the original working tree but both
turned out to be saved HTML error pages rather than working scripts, and were dropped here).

## Running a lineage

Each lineage's `train.py` (or `cmamba_train.py` / `pretrain.py`) is self-contained:

```
python models/baseline/train.py
python models/before_model/train.py
python models/after_model/train.py
python models/ssl_model/pretrain.py      # stage 1
python models/ssl_model/finetune.py      # stages 1+2+3 end-to-end
python models/cmamba/cmamba_train.py
python models/cmamba/cmamba_ablation_train.py   # channel-mixing ablation
```

`before_model`, `after_model`, and `cmamba` also have an `ablation.py` (or `cmamba_ablation_train.py`)
driver that trains multiple variants (with/without witness, uni/bidirectional, with/without
channel-mixing) on identical data splits and reports a side-by-side accuracy comparison.

Checkpoints, wandb logs, and generated plots are written to the current working directory and are
gitignored -- rerun the relevant script to regenerate them.
