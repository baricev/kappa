**Integrating MaxText MoE Strategies into a Third-Party JAX Training + Inference Codebase**

This file is the **canonical** MoE integration guide: MaxText-style phases (below), inference/decode/sharding caveats, and the **`kappa` roadmap** at the end. A short alias pointer lives at [`QWEN3_MOE_IMPLEMENTATION_PLAN.md`](QWEN3_MOE_IMPLEMENTATION_PLAN.md).

The recommended MaxText approach—starting with the **simple dropping mode** (fixed-capacity padding + explicit token dropping) for rapid prototyping and numerical correctness checks, then upgrading to the **dropless ragged backend** (Tokamax preferred) for maximum efficiency—transfers directly to a JAX codebase that supports **both training and inference**. The Qwen/Qwen3-30B-A3B-Instruct-2507 model (128 experts, top-8 routing) remains fully compatible, and all modes preserve the clean FSDP → EP transition (with EP communication isolated to high-bandwidth ICI on TPUs or NCCL on GPUs).

Your existing MoE layer (router → dispatch → expert FFN → combine) already contains the core building blocks. The changes remain localized and non-disruptive; you will vendor a small number of helpers from MaxText while adding training-specific elements: the load-balancing auxiliary loss and (for sparse modes) efficient backward-pass handling via `use_custom_sort_vjp`.

### Phase 1: Implement Simple Dropping Mode (Fixed Padding) – Ideal First Step for Training + Inference
This mode uses only standard JAX/XLA operations and provides **full automatic differentiation** with no custom VJP required. It is the safest starting point for verifying correctness in both forward (inference) and backward (training) passes.

1. **Add configuration parameters** (mirror MaxText defaults):
   ```python
   capacity_factor: float = 1.25          # > 0 enables dropping
   num_experts: int = 128
   num_experts_per_tok: int = 8
   load_balance_loss_weight: float = 0.01 # critical for training only
   ```

2. **Extend your MoE forward pass** (adapted from MaxText’s dense dropping path):
   - Compute top-k gating (your existing router).
   - Calculate per-expert `capacity = round(capacity_factor * (batch_size * seq_len / num_experts))`.
   - Build fixed buffer `[num_experts, capacity, hidden_dim]`, gather tokens (padding with zeros), run dense expert FFN on the full buffer, mask padding, and combine. Excess tokens are dropped (residual-only path).

   Pseudocode (insert/replace in your `MoE` module):
   ```python
   # After top-k routing
   bs, seq_len, _ = hidden.shape
   capacity = round(capacity_factor * (bs * seq_len / num_experts))
   # dispatch_indices, expert_mask = _build_fixed_capacity_dispatch(...)  # copy MaxText helper
   dispatched = jnp.zeros((num_experts, capacity, hidden_dim), dtype=hidden.dtype)
   # gather + pad
   expert_out = expert_ffn(dispatched)                     # dense matmul
   # mask padding, combine with gate weights
   combined = combine_with_mask(expert_out, expert_mask, top_k_weights)
   ```

3. **Add training-only auxiliary loss** (computed once per MoE layer, after routing):
   ```python
   def load_balance_loss(router_probs, dispatch_counts):
       # router_probs: averaged gate probabilities across tokens
       # dispatch_counts: histogram of tokens routed per expert
       variance_term = jnp.var(router_probs, axis=-1).mean()
       imbalance_term = jnp.mean((dispatch_counts / dispatch_counts.sum()) ** 2)
       return load_balance_loss_weight * (variance_term + imbalance_term)

   # In training forward pass
   aux_loss = load_balance_loss(router_probs, expert_counts)
   total_loss = main_loss + aux_loss
   ```

   This prevents expert collapse during training; the loss is **zero** during pure inference.

4. **Validation**: Run identical inputs through your original basic implementation vs. this dropping mode (both forward and backward). Expect numerical equivalence within ~1e-5 (bfloat16). Gradients for dropped tokens are naturally zero.

This phase requires **zero new dependencies** and guarantees static shapes for easy XLA compilation and sharding.

### Phase 2: Upgrade to Dropless Ragged Backend (Tokamax Preferred) – Production Recommendation for Training + Inference
Once Phase 1 passes all numerical and gradient checks, switch to the **Tokamax Ragged Dot** kernel. This is MaxText’s recommended path for both training and inference because it:
- Eliminates all padding/dropping overhead,
- Delivers the highest MFU,
- Provides **full forward + backward support** (all 18 Pallas tile configurations for weights and activations),
- Works seamlessly with your existing FSDP/EP sharding.

1. **Install Tokamax** (standalone, no full MaxText needed). In `kappa` we standardize on **git install** (matches OpenXLA head):
   ```bash
   uv pip install git+https://github.com/openxla/tokamax.git
   # alternatives: pip install -U tokamax  (if published)  or  pip install git+https://github.com/openxla/tokamax.git
   ```

2. **Vendor minimal MaxText helpers** (~100–150 lines total):
   - Clone https://github.com/AI-Hypercomputer/maxtext.
   - Copy from `src/maxtext/layers/moe.py`:
     - `_sort_activations` / dispatch logic,
     - `load_balance_loss` (already shown above),
     - Sharding utilities (`PartitionSpec` for expert axis).
   - Adapt these into your NNX/Flax-style module (the `RoutedMoE` class is deliberately lightweight).

3. **Replace expert computation** (set `sparse_matmul=True`, `use_tokamax_gmm=True`, `capacity_factor=-1`):
   ```python
   # Sort activations once (shared between fwd and bwd)
   sorted_activations, expert_offsets, token_counts = _sort_activations(
       hidden, dispatch_indices, use_custom_sort_vjp=True
   )

   # Tokamax kernel (full autodiff support)
   import tokamax
   expert_out = tokamax.ragged_dot(
       sorted_activations,          # [total_active_tokens, hidden_dim]
       expert_weights,              # [num_experts, hidden_dim, mlp_dim]
       expert_offsets,
       token_counts,
       implementation="auto"
   )

   # Combine step (unchanged)
   combined = combine_expert_outputs(expert_out, top_k_weights)
   ```

4. **Enable efficient backward pass** (training-critical):
   ```python
   use_custom_sort_vjp: bool = True   # replaces inefficient jax.numpy.take scatter-add
   ```
   This custom VJP is automatically registered when you call the sort helper with the flag; gradients flow correctly through the ragged kernel with minimal overhead.

5. **Sharding and EP isolation** (unchanged from Phase 1):
   - Keep FSDP on non-MoE layers.
   - Apply expert-dimension sharding (`shard_exp_on_fsdp` or `expert_shard_attention_option`) exactly as in MaxText.
   - EP collectives remain confined to the MoE layer.

### Additional Training-Only Considerations
- **Load-balancing loss**: Always enabled during training (set `load_balance_loss_weight > 0`). It is computed from router probabilities and dispatch counts in **both** modes.
- **Tile-size tuning** (Tokamax only): For optimal TPU/GPU MFU, expose the 18 tile parameters (`wi_tile_*`, `wo_tile_*`) and run Tokamax’s built-in tuner on representative batch sizes.
- **Numerical stability**: Set `float32_weight_sum=True` if you observe convergence issues with lower-precision (FP8/BF16) during training.
- **Gradient checkpointing**: Works transparently with Tokamax (no special handling needed).

### Testing and Rollout Strategy
1. **Correctness**: Dropping mode ↔ Tokamax mode must match on forward and full backward passes (use `jax.checkpoint` if memory is tight).
2. **Performance**: Measure tokens/second, MFU, and HBM on training batches. Expect 1.5–2× uplift in the Tokamax path.
3. **Edge cases**: Test high-imbalance routing, varying batch/sequence lengths, and FP8 (Tokamax supports it natively).
4. **Serving export**: For inference-only export (StableHLO), Tokamax works with `tokamax.DISABLE_JAX_EXPORT_CHECKS`.

In summary, the integration path is identical to the inference-only case but with three training-specific additions: the auxiliary loss, `use_custom_sort_vjp=True`, and full tile-configuration support. The phased rollout keeps your codebase modular, maintainable, and production-ready for both training and inference. For the authoritative reference, consult the MaxText MoE configuration documentation and the Tokamax library directly.

This approach gives you a clean, scalable MoE stack that matches MaxText’s production recommendations while staying tightly integrated with your existing architecture.



----


### Why the Phased Approach is Spot On

**Phase 1 (Fixed Padding) is a necessary safety net.** Jumping straight into custom Pallas kernels and ragged tensors is a recipe for silent gradient errors or unexplainable XLA panics. Implementing the dense, fixed-capacity mode first establishes a ground-truth baseline. As is often the case when debugging complex ragged implementations, isolating whether a bug is a fundamental networking error (like a misaligned `all_to_all` in your Expert Parallelism) or a custom kernel issue is nearly impossible without this dense baseline to diff against.

**Phase 2 (Tokamax) is the correct endgame.**
Relying on OpenXLA's Tokamax library offloads the burden of maintaining highly optimized grouped matrix multiplications. Given that token dropping heavily degrades model quality—and padding destroys Model FLOPs Utilization (MFU)—the ragged dot implementation is the only viable path for production-grade MoE.

### The Missing Pieces: Where the Proposal Needs Adjustment

While the proposal is excellent for standard training and prefill phases, it glosses over a few critical realities when this stack is used for autoregressive inference, particularly over massive context windows.

**1. The Autoregressive Decoding Bottleneck**
The proposal heavily emphasizes the forward and backward passes (which process massive, contiguous blocks of tokens). However, during the token-by-token decoding phase of generation, your batch size effectively drops to 1 per sequence.
* **The Issue:** Megablox/Tokamax kernels are optimized for large-scale grouped matrix multiplications. In the decoding phase, you are routing single tokens to distributed experts.
* **The Fix:** Your stack needs logic to cleanly bypass the heavy Tokamax ragged machinery during standard autoregressive decoding. For step-by-step decoding, a simple dense gather/scatter over the ICI network is often faster than setting up the ragged metadata and invoking a Pallas kernel for a handful of tokens.

**2. Stripping the Training Overhead for Inference**
The proposal suggests that during inference, the `load_balance_loss` simply evaluates to zero.
* **The Issue:** Even if the loss is zero, computing the variance and tracking the expert histograms `(dispatch_counts)` injects unnecessary FLOPs and state tracking into your compiled graph. When your custom Python scheduler is pushing continuous scripts or long chapters through the pipeline, this overhead accumulates.
* **The Fix:** Do not rely on runtime zeroes. Use static compilation flags (e.g., standard Python `if training:` blocks evaluated before the `@jax.jit`) to completely strip the auxiliary loss calculations, histogram tracking, and custom VJPs from the AST before XLA compiles the inference step.

**3. The `shard_map` Boundary**
The proposal correctly notes that EP collectives must remain confined to the MoE layer. However, it doesn't explicitly mention *how* to enforce this safely. As you implement this, ensure that the Phase 2 Tokamax kernel and its associated `argsort` and network transfers are strictly encapsulated within a `shard_map` block. This prevents the XLA compiler from accidentally fusing the ragged MoE logic with your surrounding FSDP dense layers or your custom paged KV cache attention kernels.

Overall, this is a highly robust architectural plan. It correctly identifies the necessity of isolating the MoE complexity while providing a clear path from a verifiable baseline to peak performance.

---

## Quantization on TPU (MaxText-aligned, including MoE)

In the **MaxText** reference stack on TPUs, the usual approach is **uniform low precision across the whole model**—attention projections, non-MoE FFN blocks, **and** MoE expert matmuls—rather than a standing policy of “INT8 (or INT8 weight-only) on dense layers only, experts stay BF16 until later.” Expert ragged paths were extended so quantization applies there too.

### Backends (MaxText)

- **Qwix** (recommended, relatively non-intrusive): quantized training/inference by intercepting `nn.Dense` / `dot_general`-style ops. Supports variants such as `"int8"`, `"fp8"`, `"fp8_full"`, etc. Configuration is often **global** (e.g. `use_qwix_quantization=true` plus a `quantization=...` flag), with optional **fine-grained `QtRule` regexes** (e.g. `decoder/.*layers.*`) to include or exclude module paths. Unless a rule excludes them, **the same rules apply to attention, MLP, and expert FFNs**.
- **AQT** (fallback / legacy): supports `"int8"`, `"int8w"` (weights-only), `"int4w"`, and mixed-precision modes.

### MoE expert matmuls (ragged kernels)

Low precision is expected to flow into **expert** matmuls, not stop at the router:

- **Megablox / `jax.lax.ragged_dot`**: supports **INT8, FP8, and BF16** on the forward path (multiple tile configurations).
- **Tokamax `ragged_dot`**: optimized heavily for **FP8** (many tile configurations); broader dtype coverage evolves with the library—treat the Tokamax / MaxText release you pin as source of truth.

### Aggressive FP8 recipe (production-oriented on modern TPUs)

MaxText documents an **FP8-heavy** recipe (sometimes described in terms like **w8a8g8** semantics) aimed at large MoE on TPUs with strong FP8 support (e.g. Ironwood-class / v7x): **E4M3FN** for weights and activations with **static per-axis scaling** in a fixed nominal range, **E5M2** for gradients with dynamic scaling, rounding choices for reproducibility, and quantization applied across attention, MLP, **Megablox / grouped matmul paths**, and relevant **weight all-gathers**. **Splash** attention is often **excluded** from FP8 for VPU / quality reasons—that exclusion is an attention-kernel choice, not MoE-specific.

### INT8 vs FP8 in practice

- **INT8** (Qwix `int8` or AQT): conservative baseline; good for validating numerics and training stability.
- **FP8** (`fp8` / `fp8_full`): more aggressive; on hardware with native FP8 MXUs it can materially improve compute density and MFU vs BF16; often preferred at scale for MoE once stability is confirmed.

### Contrast with **kappa** (`jax-functional`) today

This repo is **functional JAX** (einsum / `lax` MoE), **not** MaxText’s Flax/NNX + Qwix interception. There is **no** in-tree `use_qwix_quantization` switch. To mirror MaxText’s **full-model** INT8/FP8 story here you would either: adopt a **MaxText-class training/inference wrapper** around the same parameterization, or **reimplement** quantized contracts (scales, `dot_general` / packed weights, MoE ragged path dtype policy) on the existing `kappa/qwen3/*` forward. The **roadmap** items for Tokamax + EP still apply; quantization is an **additional** layer once those paths are stable.

For authoritative flags, `QtRule` examples, and tile tuning, use **MaxText’s quantization guide** and **MoE configuration** docs in the pinned MaxText revision.

---

## Kappa (`jax-functional`) roadmap: Qwen3 MoE

This section maps the ideas above onto **this repo**. Phases **A–F** are kappa-specific naming; they align with **Phase 1 (fixed capacity)** and **Phase 2 (Tokamax)** in the preceding sections, plus decode, `shard_map`, and training stripping called out in *The Missing Pieces*.

### Goals (current consensus)

- **Inference parity vs Hugging Face is not a requirement.** Golden tests against HF logits or hidden states are out of scope unless priorities change.
- **Internal consistency** still matters: diff one implementation against another (e.g. fixed-capacity vs dropless ragged, or prefill vs decode path) to catch routing and kernel bugs.
- **Training** (loss, optimizer, data) remains **later**; design MoE APIs so training can attach **without** polluting inference graphs.
- **Reference architecture**: phased **fixed-capacity JAX** → **dropless ragged** (**Tokamax preferred** on TPU for scale), with **decode-aware** dispatch and **`shard_map`-style** isolation when expert parallelism lands.

### Current baseline (`kappa`) — updated

- **`kappa/qwen3/ffn_moe.py`**: router + top-k; **`gather_einsum`** (historical `take` + einsum), **`fixed_capacity`**, **`ragged_jax`** (`jax.lax.ragged_dot`), **`ragged_tokamax`** (calls `tokamax.ragged_dot` when available, else JAX).
- **`Qwen3Config`**: `moe_impl`, `moe_capacity_factor`, `moe_ragged_decode_token_threshold`; **`scripts/verify_qwen3_moe_impls.py`** and **`scripts/infer_qwen3.py --moe-impl`**.
- **Install Tokamax** (optional, for `ragged_tokamax`): `uv pip install git+https://github.com/openxla/tokamax.git` (see `AGENTS.md`).
- **Still missing**: EP / `shard_map` MoE boundary, training (aux loss, sort VJP, inference-only `jit`), Tokamax **API/tile tuning** validation on TPU, richer tests (`ragged_tokamax` vs `ragged_jax` when installed).

### Phase A — Configuration and modes — **done**

- `moe_impl`, `moe_capacity_factor`, `moe_ragged_decode_token_threshold` on `Qwen3Config`; default `gather_einsum`.

### Phase B — Fixed-capacity (dropping) path — **done (inference forward)**

- Implemented in `ffn_moe.py`; compare to `gather_einsum` via `verify_qwen3_moe_impls.py` (high-capacity case) or lower capacity to exercise drops.
- **Training (later)**: optional **load-balancing auxiliary loss** as in *Phase 1* above.

### Phase C — Dropless ragged expert compute — **done (first slice)**

- Sort-by-expert + three `ragged_dot` SwiGLU matmuls + weighted combine; **`ragged_tokamax`** tries `tokamax.ragged_dot`.
- **Remaining**: confirm **OpenXLA Tokamax** call signature / kwargs against your installed revision; **tile tuning** (`wi_tile_*` / `wo_tile_*`) when MFU matters; optional **`verify` script** branch asserting `ragged_tokamax` ≈ `ragged_jax` when Tokamax is present.

### Phase D — Prefill vs decode dispatch — **partially done**

- **Done**: `total_tokens <= moe_ragged_decode_token_threshold` forces **`gather_einsum`** inside `moe_swiglu_ffn` for ragged modes (decode-friendly).
- **Optional later**: explicit prefill/decode hooks in `block.py` if you want different thresholds or logging without recomputing `b*t` inside the FFN.

### Phase E — Sharding and expert parallelism — **not started**

- Encapsulate **sort + ragged_dot + EP collectives** inside a **`shard_map` boundary** (*The `shard_map` Boundary* above).
- Add **PartitionSpec** for expert axes; keep collectives **localized** to the MoE layer.

### Phase F — Training (when prioritized) — **not started**

- **`use_custom_sort_vjp=True`** on the dispatch path; **strip training from inference graphs** with **Python-level `if training:`** or separate `jit` targets so aux loss, histograms, and training-only VJPs are **absent** from compiled inference (*Stripping the Training Overhead* above).

### What’s next (backlog)

1. **Tokamax on device**: install with `uv pip install git+https://github.com/openxla/tokamax.git`, run `ragged_tokamax` on **TPU** (and MPS if supported), adjust `_ragged_dot` in `ffn_moe.py` if the library API differs from `jax.lax.ragged_dot`.
2. **Tests**: extend `verify_qwen3_moe_impls.py` to compare `ragged_tokamax` vs `ragged_jax` when Tokamax imports; optional `pytest` wiring.
3. **Phase E** when scaling off single device: **`shard_map`**, expert-axis **PartitionSpec**, EP collectives (vendored patterns from MaxText, copied into `kappa`).
4. **Phase F** when training MoE: load-balance loss, custom sort VJP, separate inference graph build.
5. **Other Qwen inference gaps** (outside this MoE subsection): ~~chunked long prefill~~ (done: `forward_prefill_chunk` + `generate(..., prefill_chunk_size=)`); paged KV, etc.—see Gemma3 / separate notes.
6. **Quantization (later)**: MaxText-style **full-model** INT8/FP8 (Qwix/AQT) including **expert** ragged matmuls—see **Quantization on TPU (MaxText-aligned, including MoE)** above; kappa has no Qwix interception today, so this is either a wrapper around MaxText-style modules or explicit quantized ops in `kappa/qwen3/*`.

### References (external)

- MaxText: https://github.com/AI-Hypercomputer/maxtext — `src/maxtext/layers/moe.py` for patterns to **copy** into `kappa`.
- Tokamax: https://github.com/openxla/tokamax

