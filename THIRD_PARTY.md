# Third-party sources

## Sparse Delta Memory

The experiments implement the Sparse Delta Memory semantics described by
Cabannes et al. The implementation is checked against an independent serial
form of the published semantics.

- Paper: <https://arxiv.org/abs/2607.07386>
- Official repository: <https://github.com/facebookresearch/sparse-delta-memory>
- Pinned revision: `183e7df809131b80ad4393741029d0f20fc3640b`
- Official repository license: CC BY-NC 4.0
- Paper license: CC BY 4.0

The code in this repository was written independently against the published
semantics. No official source code, model weights, or generated artifacts are
redistributed.

## WikiText-103

The reproduction script downloads the public WikiText-103 raw-v1 Parquet
conversion from `Salesforce/wikitext` at revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`, then verifies every shard before
local preprocessing. Dataset files are not redistributed. The dataset card
lists CC BY-SA and GFDL licensing; users are responsible for those terms when
downloading the data.
