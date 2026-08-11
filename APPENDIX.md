# Experimental details

## Model and data

The occupancy run used layout `BBBBBBBA`: seven faithful SDM layers followed
by one dense-attention layer. Width was 128 with four transformer heads, one
SDM memory head, `N=256`, top-16 reads, top-4 writes, and expansion-four MLPs.
Training used seed 0, BF16 activations, 4,000 optimizer steps, batch 8, sequence
length 2,048, AdamW, and 65.536 million GPT-2 tokens sampled from deterministic
WikiText-103 streams. Validation and test each used 128 fixed sequences.

The model had 14,654,556 active parameters plus 229,376 learned initial-memory
parameters. Validation NLL was 5.44253; test NLL was 5.45160. Exact prefix
causality had maximum difference zero.

Occupancy is the exact number of distinct hard write addresses selected by a
single request, summed across the seven SDM banks. `M₀` is model state shared
across requests. The occupancy table counts FP32 private value rows; a dense
int32 logical-to-physical map and allocator metadata are additional runtime
state.

At 2,048 tokens, the dense SDM value table is 896 KiB per request and the mean
copy-on-write values are 330.5 KiB. The logical map is 7 KiB. Allocator
metadata, dense KV, training activations, and peak device allocation are
separate quantities.

## Allocator benchmark

The synthetic random-route CUDA screen used the same seven-layer, width-128,
`N=256`, read-16, write-4 shape. Dense-table and copy-on-write execution matched
exactly in FP32 mutable state, and prefix causality was exactly zero. Each case
used five warm-up iterations and 20 measured iterations, repeated as five full
traces on one NVIDIA A40. The table reports the median trace.

The selected policy adds enough rows for four further decode steps whenever
capacity runs out. “Logical rows not materialized” counts addresses that never
received a private value. “Allocated serving-state reduction” compares the
actual slab capacity, logical map, allocator metadata, allocated dense KV, and
lifecycle state with the fully materialized dense-table equivalent. It is not
peak GPU allocation or model-weight memory; shared `M₀` is excluded from both
sides.

Prefill runs the dense native recurrence and then packs its exact terminal
working set. The retained decode representation consists of a layer-wide FP32
value slab, one dense int32 logical-to-physical map per live bank, a shared
free-row stack, template IDs, counters, and pooled headroom. Reads from an
untouched address resolve directly to shared learned `M₀`; only the first write
creates private value state.

Growing allocates a larger contiguous value tensor, copies the existing slab,
and adds the new row IDs to the free stack. The four-step aggregate quantum
amortizes that operation without assigning fixed row quotas to requests. At
batch 64 it copied 157,696 width-128 FP32 rows over the complete trace and still
measured 6.70% slower than dense-table decode. Release normally returns rows to
the existing pool; trimming repacks live rows only after sustained excess
capacity.

The fused narrow CUDA path is used after host-side accounting proves enough
capacity for the next interval. It resolves first touches, applies the exact
gated-delta write, and performs the sparse read in one kernel launch. Validation
retains a slower diagnostic path with explicit allocation resolution and
overflow checks.

## Source boundary

The native controller, trainer, deterministic data preparation, allocator, and
serving benchmark required to rerun these two experiments are included. The
implementation was written independently against the pinned SDM paper and
released semantics; no upstream source code or weights are redistributed. See
[THIRD_PARTY.md](THIRD_PARTY.md) for licenses.
