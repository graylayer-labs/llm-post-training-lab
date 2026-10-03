# Where the memory goes, and what changes at 70B

Part of the Part 4 write-up, issue
[#4](https://github.com/graylayer-labs/llm-post-training-lab/issues/4).
This page covers two of its questions: where the memory goes during
training on the laptop, and how the same recipe would change at 70B across
several nodes.

Two kinds of number appear here. **Measured** figures come from saved files
in `outputs/`, and each one names its file. **Estimates** are back-of-envelope
arithmetic, shown inline so a reader can check them. Section 3 is reasoning
only. Nothing in it was run.

**A note on units.** The measured figures were computed as
bytes / 2^30 (`accelerator_memory_gb` in
`src/post_training/train/common.py`, which reads
`torch.mps.driver_allocated_memory()`). They are therefore GiB, although the
run files and [sft-results.md](sft-results.md) label them GB. Section 1 and 2
estimates use the same divisor, so the columns compare like with like.
Section 3 uses decimal GB and TB (1 TB = 1,000 GB), as hardware vendors do.

## 1. Where the memory goes on the laptop (SFT)

Setting: `Qwen/Qwen2.5-0.5B` in bf16 on MPS, LoRA r=16 on all seven
projections plus two trainable embedding rows, `nll` loss, micro-batch 2,
`max_length` 512 (`configs/sft.yaml`). 494,032,768 parameters in total,
8,800,000 trainable (`outputs/sft/summary.json`, `params_total` and
`params_trainable`). Vocabulary 151,936 (the model's `config.json`; written
as "152k" elsewhere in this repo).

### Measured figures and their sources

| Figure | Value | Source |
|---|---|---|
| Peak MPS driver memory, clean SFT re-run, sampled once per optimiser step and at evaluation | 8.10 GB | `outputs/sft/summary.json`, `peak_memory_gb`; [sft-results.md](sft-results.md), Runs table |
| MPS memory at the end of the same run | 6.0 GB | `outputs/sft/summary.json`, `device.mps_alloc_gb_at_end` |
| Memory probe, LoRA r=16, `nll`, 8 steps at batch 2 (two runs) | 4.56 / 4.70 GB | [sft-results.md](sft-results.md), "LoRA vs full fine-tune memory"; raw in `outputs/memory_probe/results.jsonl` |
| Memory probe, LoRA r=16, `chunked_nll`, no token rows | 3.75 GB | same |
| Memory probe, LoRA r=64 | 5.28 GB | same |
| Memory probe, full fine-tune, bf16 weights, grads and AdamW state | 6.59 GB | same |

The probe trains on 16 short rows, so its sequences are shorter than the
512-token cap. Its peaks are not the peak of the full run. The full run's
peak also covers evaluation and the longest batches. Both numbers are
driver memory, which includes the allocator's cached blocks, not just live
tensors.

### Breakdown

| Component | Measured | Estimate | How the estimate was made |
|---|---|---|---|
| Base weights, bf16 | not isolated | 0.92 GB (estimate) | 494,032,768 params × 2 bytes = 988,065,536 bytes; / 2^30 = 0.92 |
| LoRA params + grads + AdamW state | r=64 minus r=16 (mean of 4.56 and 4.70): 5.28 − 4.63 = 0.65 GB for 26.4M extra params | 0.13 GB (estimate) | PEFT keeps adapter weights in fp32 by default when the base is bf16. 8,800,000 × (4 param + 4 grad + 8 two moments) = 16 bytes × 8.8M = 140.8 MB = 0.13 GB. If the adapter were bf16 the figure would be 0.07 GB. |
| Activations saved for backward | not isolated | 1.1 to 1.5 GB (estimate) | See below |
| Full-vocab logits for `nll` | 0.81 to 0.95 GB (`nll` 4.56 / 4.70 minus `chunked_nll` 3.75) | 0.58 to 1.45 GB at the full 512 tokens (estimate) | 2 × 512 × 151,936 = 155,582,464 elements. bf16: × 2 bytes = 0.29 GB. The loss upcasts to fp32: × 4 bytes = 0.58 GB. If the bf16 logits, the fp32 copy and the fp32 gradient of the logits are all live at once: 0.29 + 0.58 + 0.58 = 1.45 GB. |
| Allocator cache and driver overhead | 8.10 − (0.92 + 0.13 + 1.3 + 1.45) ≈ 4.3 GB (derived from the measured peak and the estimates above) | not estimated from first principles | Remainder. See "Where measured and estimated disagree". |

**Activations, worked.** From the model's `config.json`: 24 layers, hidden
896, intermediate 4,864, 14 query heads and 2 key/value heads of width 64.
Per token and per layer, the tensors that autograd keeps for backward in
bf16 are roughly: two norm inputs (2 × 896), the query (896), key and value
(2 × 128), the attention output (896), the `o_proj` input (896), the MLP
input (896), and four intermediate-width tensors for gate, up, their product
and the `down_proj` input (4 × 4,864). That is 25,088 elements, about
50 KB. Over 2 × 512 = 1,024 tokens and 24 layers: 25,088 × 2 bytes × 1,024 ×
24 = 1,233 MB = 1.15 GB. If the attention backend also saves the score
matrices (eager attention does; SDPA may not): 14 heads × 512 × 512 × 2
bytes × 2 sequences × 24 layers = 0.33 GB more. So 1.1 to 1.5 GB. The
estimate ignores dropout masks and the LoRA branches' own small
intermediates. The base weights are frozen, but the activations must still
be kept, because the gradient has to flow back through every frozen layer to
reach the LoRA weights in the layers below. LoRA saves optimiser state, not
activation memory.

**Full fine-tune, for comparison.** The probe's full fine-tune keeps
weights, grads and AdamW state all in bf16: 494,032,768 × (2 grad + 4 two
moments) = 2.76 GB more than the frozen model (estimate). Measured: 6.59 −
4.63 = 1.96 GB more than the r=16 mean. It trains 56 times the parameters
for 1.4 times the memory, because at this size the logits and activations
are the larger share. A standard mixed-precision full fine-tune with fp32
master weights and fp32 moments would be 494M × 12 bytes = 5.5 GB of
optimiser state alone (estimate), which is why that path was not taken on a
24 GB laptop.

### Where measured and estimated disagree

- **The peak is roughly double the sum of the parts.** The estimates add to
  about 3.8 GB at 512 tokens. The run peaked at 8.10 GB. Three reasons,
  from the evidence in [sft-results.md](sft-results.md):
  1. *The measurement is driver memory, not live tensors.* It is sampled
     just before `torch.mps.empty_cache()`, so it includes every block the
     caching allocator still holds from that step. The out-of-memory table
     in sft-results.md shows how large that gap can get: 2.84 GiB of
     tensors against 27.34 GiB held by the driver, before the per-step cache
     release was added.
  2. *Logit buffers change shape every batch.* Each batch has a different
     sequence length, so the allocator cannot reuse the previous step's
     blocks and holds both for a while. This is the same mechanism that
     caused the original crash.
  3. *Sampling once per step misses spikes and catches others.* The sample
     sits after the optimiser step, when gradients have been released but
     the cache has not. A spike inside a step is missed; a sample during
     evaluation of a long batch is included. The true peak may be higher
     than 8.10 GB, and the composition of what was sampled is not known.
- **The logits cost less than the upper estimate.** Measured 0.81 to 0.95
  GB against 0.58 to 1.45 GB at 512 tokens. The probe's rows are shorter
  than 512 tokens, so the measured figure is for a smaller T. It sits where
  the estimate predicts for sequences of a few hundred tokens with an fp32
  copy and its gradient live together.
- **Full fine-tune state costs less than estimated.** Measured 1.96 GB
  against 2.76 GB. The sample is taken after the optimiser step, by which
  point the Trainer has most likely released the gradients (`zero_grad`),
  so the 0.92 GB of bf16 gradients is no longer live. 2.76 − 0.92 = 1.84 GB
  is close to the measured 1.96. Not verified by a separate measurement.
- **LoRA r=64 costs more than its state alone.** Measured 0.65 GB for 26.4M
  extra parameters against 0.39 GB of fp32 state (26,394,624 × 16 bytes).
  The rest is the rank-64 intermediates saved in every adapted projection,
  which the state estimate does not count.
- **Two runs of the same setting differ by 0.14 GB** (4.56 and 4.70). Read
  differences of that size as noise.

## 2. DPO memory

The DPO run is described in `configs/dpo.yaml` and
`src/post_training/train/dpo.py`. Figures below are estimates only. **The
measured DPO peak is reported in docs/dpo-results.md.**

What DPO adds, conceptually:

- **Two sequences per pair instead of one.** Each pair has a chosen and a
  rejected answer. Both go through the policy, forward and backward, so the
  activations and logits of Section 1 are counted twice per pair. At
  `batch_size: 2` pairs, `max_length: 768`, that is four sequences of up to
  768 tokens per micro-step against two of 512 for SFT: 3 times the tokens.
  Logits (estimate): 4 × 768 × 151,936 = 466,747,392 elements; × 4 bytes
  fp32 = 1.74 GB for one fp32 copy, 4.35 GB if bf16, fp32 copy and fp32
  gradient are live together. Activations (estimate): 3 × the SFT figure,
  3.4 to 4.5 GB. TRL concatenates chosen and rejected into one forward
  pass, which changes the shape but not the total.
- **A reference model.** The DPO loss compares the policy's log-probability
  ratio with a frozen reference's. In general that is a second copy of the
  model, run forward on both sequences of every pair: another 0.92 GB of
  bf16 weights (494,032,768 × 2 bytes, estimate) and a second set of forward
  activations, which need not be saved for backward.

In this project the reference costs no extra weights. The SFT adapter is
merged into the base weights, a fresh LoRA is trained on top, and TRL takes
the reference log-probs with that LoRA disabled
(`reference: sft` in `configs/dpo.yaml`; the reasoning is in
[decisions.md](decisions.md), "Use the SFT model as DPO's reference by
merging its adapter"). Switching the adapter off gives exactly the merged
SFT model, so there is no second copy of the weights and no second
optimiser. The reference forward runs under `no_grad`, so its activations
are freed layer by layer. The cost is compute, not memory: per batch of
pairs, a reference forward and a policy forward over both the chosen and
rejected sequences, plus one backward, against one forward and one backward
over a single sequence for SFT.

Why that matters here: the laptop has 24 GB of unified memory shared with
everything else. A second 0.92 GB copy would fit, but the pattern is the
point. The adapter-off trick is what makes DPO on a LoRA policy cost about
the same memory as SFT on longer sequences, and it is the same trick that
removes the reference copy at 70B (Section 3).

Rough DPO estimate, summing the parts: 0.92 weights + 0.13 LoRA state +
3.4 to 4.5 activations + 1.7 to 4.4 logits = 6 to 10 GB of live tensors
before allocator overhead (estimate). Compare that with the measured peak in
docs/dpo-results.md, remembering the allocator gap seen in Section 1.

## 3. The same recipe at 70B across multiple nodes

**This section is reasoning, not measurement.** Nothing here was run. The
project's non-goals exclude multi-GPU work. The arithmetic uses round
numbers and states its assumptions. Where a shape is needed, it uses a
Llama-2-70B-like layout: 80 layers, hidden 8,192, intermediate 28,672, 64
query heads and 8 key/value heads of width 128. Hardware: 8 nodes × 8 GPUs
with 80 GB each, NVLink inside a node, InfiniBand between nodes.

### 3.1 Full fine-tuning with mixed-precision AdamW: the state alone

Mixed-precision AdamW holds, per parameter: bf16 weights (2 bytes), bf16
gradients (2), fp32 master weights (4), and two fp32 moments (4 + 4). That is
16 bytes per parameter.

70 × 10^9 × 16 bytes = 1,120 × 10^9 bytes = 1.12 TB.

Of that, the optimiser state (master weights and two moments) is 12 bytes
per parameter = 840 GB. Weights are 140 GB and gradients 140 GB. No GPU holds
1.12 TB, so the state has to be split.

### 3.2 ZeRO stages and FSDP

ZeRO (DeepSpeed) and FSDP (PyTorch) shard the training state across the
data-parallel ranks instead of replicating it. The stages differ in what
they shard. Per-GPU figures for 64 GPUs, state only, no activations:

| Stage | Sharded | Replicated | Per-GPU state, 70B, 64 GPUs (estimate) | Fits in 80 GB? |
|---|---|---|---|---|
| None (plain DDP) | nothing | everything | 1,120 GB | No |
| ZeRO-1 | optimiser state (12 B/param) | weights, grads | 140 + 140 + 840 / 64 = 293 GB | No |
| ZeRO-2 | optimiser state, grads | weights | 140 + (140 + 840) / 64 = 155 GB | No |
| ZeRO-3 / FSDP full shard | optimiser state, grads, weights | nothing | 1,120 / 64 = 17.5 GB | Yes |

Only ZeRO-3 or FSDP full sharding gets a 70B full fine-tune under 80 GB, and
only when sharded across all 64 GPUs. Sharding within one node of 8 gives
1,120 / 8 = 140 GB per GPU, which does not fit. That is the key constraint
below: full fine-tuning at 70B must shard across nodes, over the slower
interconnect.

FSDP's sharding unit is a layer (or a group of layers). At each layer's
forward, every rank all-gathers that layer's full bf16 weights (70B / 80
layers × 2 bytes = 1.75 GB per layer, estimate), uses them, and frees them.
So one or two layers' worth of gathered weights is live at any time, on top
of the 17.5 GB shard.

### 3.3 Activations and activation checkpointing

Per token and per layer, using the same tally as Section 1 with the 70B
shapes: two norm inputs (2 × 8,192), query (8,192), key and value (2 ×
1,024), attention output (8,192), `o_proj` input (8,192), MLP input (8,192),
four intermediate-width tensors (4 × 28,672). That is 165,888 elements, 332
KB in bf16, assuming flash attention so no score matrices are saved. Over
80 layers: 26.5 MB per token.

For one sequence of 4,096 tokens: 26.5 MB × 4,096 = 109 GB (estimate). That
alone exceeds a GPU, so activations have to be cut, and two tools do it:

- **Tensor parallelism (TP) within a node** splits each layer's weights and
  most of its activations across the 8 GPUs of a node. With TP=8 the 109 GB
  per sequence becomes 13.6 GB per GPU (estimate), before checkpointing.
- **Activation checkpointing** saves only each layer's input (8,192 × 2 bytes
  = 16 KB per token per layer) and recomputes the rest during backward. For
  4,096 tokens and 80 layers: 16 KB × 4,096 × 80 = 5.4 GB per sequence
  (estimate), plus one layer's full activations as a transient during
  recompute: 332 KB × 4,096 = 1.4 GB, divided by TP. The cost is about one
  extra forward pass of compute, roughly a third more per step.

With TP=8 and full checkpointing, a micro-batch of two 4,096-token sequences
costs about 2 × 5.4 / 8 + 1.4 / 8 ≈ 1.5 GB per GPU if the saved inputs are
also sharded (as in Megatron's sequence parallelism), or about 11 GB if they
are replicated across the TP group. Plan for the larger figure.

The logits appear at 70B too. With a 128k vocabulary and 4,096 tokens:
4,096 × 131,072 × 4 bytes = 2.1 GB per sequence in fp32 (estimate), which
TP splits along the vocabulary to 0.27 GB per GPU. Chunked or fused losses
avoid holding them at all. The laptop's `nll` problem is the same problem,
smaller.

### 3.4 Communication

What moves, per training step, under ZeRO-3 / FSDP full shard across 64
GPUs:

- **All-gather of weights, forward.** Each GPU must receive every weight it
  does not hold: 140 GB × 63 / 64 ≈ 138 GB in bf16.
- **All-gather of weights, backward.** The gathered weights were freed after
  the forward to save memory, so they are gathered again: another ≈ 138 GB.
  (FSDP can keep them if memory allows, trading memory for bandwidth.)
- **Reduce-scatter of gradients.** Each GPU sends its share of every other
  rank's gradient shard: ≈ 138 GB in bf16.

Total about 3 × 140 GB ≈ 420 GB per GPU per step (estimate). How long that
takes depends on the link:

- NVLink inside a node: about 450 GB/s per direction per GPU on current
  hardware (assumption). 420 GB takes about 1 s.
- InfiniBand between nodes: 400 Gb/s = 50 GB/s per GPU NIC (assumption). 420
  GB takes about 8 s, if all of it crosses nodes.

Compute per step, for a global batch of 1 million tokens: 6 × 70 × 10^9 ×
10^6 = 4.2 × 10^17 FLOP. At 400 TFLOP/s sustained per GPU (about 40% of
peak, assumption) over 64 GPUs: 4.2 × 10^17 / (64 × 4 × 10^14) ≈ 16 s.
So ZeRO-3 across InfiniBand spends about half as long on communication as
on compute, and much of it can overlap with compute (gather layer n+1 while
computing layer n). With a smaller batch per step the ratio gets worse,
because communication volume is fixed per step while compute scales with
tokens. This is why ZeRO-3 across nodes wants large batches and why
hybrid schemes are used instead.

**Why tensor parallelism stays inside a node.** TP inserts two all-reduces
of activations per layer in the forward and two in the backward, on the
critical path, with nothing to overlap them against. Each one moves the
layer's activations: 4,096 tokens × 8,192 × 2 bytes = 67 MB per sequence
(estimate). Over 80 layers, four per layer: 320 all-reduces of 67 MB, about
21 GB per sequence per step, in small latency-bound pieces. That is fine on
NVLink and painful over InfiniBand. So TP groups are the 8 GPUs of a node,
and the cross-node dimension is data parallel (FSDP across the 8 nodes) or
pipeline parallel.

**Hybrid layout for full fine-tuning.** TP=8 inside each node cuts each
GPU's share of the model to 70B / 8 = 8.75B parameters. FSDP across the 8
nodes then shards those 8.75B × 16 bytes = 140 GB eight ways: 17.5 GB per
GPU, the same total as flat ZeRO-3, but the all-gather group is 8 ranks
across nodes instead of 64, and the gathered volume per GPU drops to 8.75B
× 2 bytes × 7 / 8 ≈ 15 GB per pass, about 46 GB per step (estimate), rather
than 420.

### 3.5 LoRA at 70B

The base is frozen, so it has no gradients and no optimiser state: the 840
GB of optimiser state and 140 GB of gradients disappear. What remains is the
140 GB of bf16 weights, which still do not fit one 80 GB GPU. Two ways out:

- **Shard the frozen weights within a node.** FSDP full shard over 8 GPUs:
  140 / 8 = 17.5 GB per GPU. The forward and backward still all-gather each
  layer's weights, but over NVLink and with no gradient reduce-scatter for
  the base. Nodes can then be plain data-parallel replicas.
- **Quantise them (QLoRA).** 4-bit weights: 70 × 10^9 × 0.5 bytes = 35 GB,
  plus quantisation constants, about 37 GB. The whole base fits on one GPU
  with room for activations. QLoRA is viable on CUDA; it was ruled out on
  the laptop because bitsandbytes is CUDA-only ([decisions.md](decisions.md),
  "Use bf16 LoRA on Apple MPS"). It costs dequantisation compute on every
  forward.

The LoRA state is small. With r=16 on all seven projections: per layer,
r × (in + out) summed over q (8,192 + 8,192), k (8,192 + 1,024), v (8,192 +
1,024), o (8,192 + 8,192), gate (8,192 + 28,672), up (8,192 + 28,672), down
(28,672 + 8,192) = 16 × 161,792 = 2,588,672 parameters; × 80 layers = 207M
parameters, 0.3% of the model. In fp32 with AdamW: 207M × 16 bytes = 3.3 GB
(estimate). That can be replicated on every GPU without sharding.

**How LoRA changes the communication.** The gradient reduce-scatter shrinks
from 140 GB to the LoRA gradients: 207M × 2 bytes = 0.4 GB per step. With
QLoRA and the full base on every GPU there is no weight all-gather at all,
only a 0.4 GB all-reduce of adapter gradients across replicas, which any
interconnect handles. With a sharded bf16 base the all-gather of weights
remains, but stays inside the node. In short, LoRA turns a
communication-bound problem into a compute-bound one and makes simple data
parallelism across nodes viable again.

The question "does LoRA still make sense at 70B" has two halves. For memory
and communication, clearly yes. For quality, a full fine-tune on enough data
is still the upper bound, and LoRA's rank and target set become tuning
choices. This project cannot answer the second half; it trained one rank on
one model.

### 3.6 DPO at 70B

DPO needs reference log-probabilities for every chosen and rejected
sequence. Three ways to get them:

1. **A separate reference model.** A full second copy: 140 GB of bf16
   weights, sharded like the policy (17.5 GB per GPU over 8, 2.2 GB per GPU
   over 64), plus a full forward pass on both sequences of every pair, with
   its own all-gathers. This is unavoidable when the policy is a full
   fine-tune, because then the policy's weights drift from the reference
   and there is no cheap way back.
2. **The adapter-off trick.** When the policy is a LoRA on top of the
   reference, disable the adapter and run the same weights. No second copy.
   This is what the laptop run does (Section 2) and it works the same at
   70B.
3. **Precompute the reference log-probs once.** The reference is frozen, so
   its log-prob for each sequence never changes during training. Run one
   forward pass over the whole pair set before training, store one number
   per sequence, and train with the policy alone. Compute: one inference
   pass over the data, about a third of one training epoch's FLOPs, done
   once instead of every epoch. It can run on a quantised copy or on
   inference hardware. TRL exposes this as `precompute_ref_log_probs`.

Precomputing is the usual answer at scale because it removes both the
memory and the per-step compute of the reference, and because preference
datasets are small compared with pretraining data, so the stored numbers
are tiny. Its costs are bookkeeping (the stored log-probs must match the
tokenisation and padding used in training) and one constraint: any method
that changes the reference during training, such as periodic reference
updates, cannot use it. The adapter-off trick is simpler when the policy is
LoRA, at the price of an extra forward per pair each step.

### 3.7 What I would choose, and the per-GPU figure

**SFT at 70B: LoRA + FSDP full shard within a node, data parallel across
nodes.** Per GPU, estimate: 17.5 GB (bf16 base shard) + 1.75 GB (one layer
gathered) + 3.3 GB (LoRA params, grads and AdamW, replicated) + 11 GB
(activations with checkpointing, two 4,096-token sequences, inputs
replicated) + about 2 GB (logits and loss buffers, chunked) ≈ 36 GB, under
half of 80 GB. Why: no optimiser state for the base, no gradient
reduce-scatter for the base, the weight all-gather stays on NVLink, and the
cross-node traffic is a 0.4 GB adapter all-reduce per step. It is also the
same recipe as the laptop run, scaled, so results and bugs transfer. If the
nodes have fewer or smaller GPUs, QLoRA with the quantised base on every
GPU and plain DDP is the fallback.

**Full fine-tuning at 70B: TP=8 inside nodes, FSDP (ZeRO-3) across the 8
nodes, full activation checkpointing.** Per GPU, estimate: 17.5 GB (state
shard: 70B × 16 bytes / 64) + 2 to 4 GB (one or two layers gathered for the
current TP shard) + about 11 GB (activations, as above) + about 1 GB
(logits, vocabulary-sharded) + several GB of allocator slack ≈ 35 to 45 GB.
Why: it is the only stage that fits at all (Section 3.2), TP inside the node
keeps the latency-bound collectives on NVLink, and sharding across only 8
nodes rather than 64 flat ranks keeps the gathered volume per GPU near 46
GB per step rather than 420. The remaining headroom goes to a larger
micro-batch, which raises compute per step and hides the cross-node
all-gathers behind it.

**DPO at 70B:** the same layout as SFT with LoRA, plus either the
adapter-off reference (simple) or precomputed reference log-probs (cheaper
per step). The chosen-and-rejected pair doubles the activation term per
pair, so halve the pairs per micro-batch or raise checkpointing.

All of these are estimates with stated assumptions. The honest check is to
run one step and read the allocator, which is exactly what the laptop
runs do.

## 4. What would differ beyond memory

- **Data pipeline.** The laptop loads 2,000 rows, filters and de-duplicates
  them in memory, and tokenises on the fly. At 70B the data is millions of
  rows, pre-tokenised and packed into fixed-length sequences so no step
  wastes compute on padding, sharded across data-parallel ranks, and
  streamed. De-duplication becomes a job in itself: the 7,587 duplicate
  rows that `make_splits` dropped here (`outputs/sft/summary.json`,
  `data_stats.duplicates_dropped`) were found by exact prompt match on a
  few thousand rows; at scale that means hashing and near-duplicate
  detection, and leakage between training and held-out sets is harder to
  rule out. Packing also changes the loss masking: prompt masking per row
  has to survive concatenation.
- **Evaluation cost.** Here, the eval loss over 200 held-out rows took about
  34 s (`outputs/sft/summary.json`, `eval_runtime`), and generation of
  answers runs in a separate pass. At 70B, generating a few hundred answers
  is cheap next to training, but a serious evaluation set is thousands of
  prompts across several systems and seeds, and decoding is memory-bound
  and slow. That calls for an inference server with continuous batching and
  a KV cache, not the training loop's `generate`. A model judge, if used,
  adds a second large model's inference per answer; this project dropped
  the judge to keep cost inside the existing subscription
  ([decisions.md](decisions.md), "Evaluate without a model judge"), and at
  scale the same question becomes a budget line.
- **Why the rubric and harness design carry over.** None of the evaluation
  design depends on model size. The harness scores every stage on the same
  held-out prompts, which is what makes "SFT versus DPO" a comparison rather
  than two numbers. The rule-based rubric checks what matters for the task
  (stopping, repetition, format, figures not grounded in the prompt), is
  deterministic, and costs nothing to run, so it scales to any number of
  answers. Every run records its commit, tree state, config and library
  versions (`outputs/*/summary.json`, `provenance`), and every quoted figure
  names its saved file. Those habits are the ones that get lost first on a
  cluster, where a run is a job id and the config was edited in a shell.
  Making the run config a file and the result a saved artefact is the part
  of this project that would change least at 70B.
