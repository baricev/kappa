# Porting Gemma-JAX to Apple Silicon via jax-mps

## Background

Apple discontinued its `jax-metal` plugin (~2024), which previously provided JAX support on Apple Silicon GPUs. A new community library, `jax-mps`, targets the Metal Performance Shaders (MPS) backend. This document records the changes required to run a functional Gemma-3 JAX implementation on `jax-mps` (JAX >= 0.5, Python 3.13).

The MPS backend is experimental. Many StableHLO operations that work on CPU, GPU (CUDA), and TPU either fail outright or produce silently incorrect results on MPS. The fixes below address three categories of incompatibility discovered during the port.

---

## Fix 1: `EmitPythonCallback` not supported on MPS

### Symptom

```
ValueError: `EmitPythonCallback` not supported on mps backend.
```

Raised during `jax.lax.scan` inside `paxml_generate_chunked_scan_queue`.

### Root cause

`jax.experimental.io_callback()` was used inside a JIT-compiled function to stream generated token chunks from device to host asynchronously via a `queue.Queue`. During XLA lowering, `io_callback` emits an `EmitPythonCallback` HLO custom-call. The MPS backend does not implement this custom-call.

### Fix (`examples/multi_turn_chat.py`)

Removed `jax.experimental.io_callback()` from inside the JIT boundary. The token chunks were already being returned from the function (`return final_carry, tokens_chunk`), so the device→host transfer (`jax.device_get`) and queue enqueue were moved to the Python driver loop (`run_full_generation_with_chunked_callback`), which runs outside of JIT.

This is the same pattern used by `chat_cli.py`, which never had this issue.

Additionally, `chunk_id` was removed from `static_argnames` since it was only consumed by the callback. Previously, every unique `chunk_id` value triggered a full XLA recompilation — removing it from static args is also a compilation-cache improvement.

```python
# Before (inside JIT):
jax.experimental.io_callback(tap_chunk, None, tokens_chunk, ordered=True)
return final_carry, tokens_chunk

# After (inside JIT):
return final_carry, tokens_chunk

# After (in Python driver, outside JIT):
chunk_host = jax.device_get(chunk_device)
_CHUNK_QUEUE.put({"chunk_id": chunk_count, "chunk": chunk_host})
```

---

## Fix 2: `stablehlo.scatter` reshape failure in `_update_ragged`

### Symptom

```
[MPS ERROR] Exception dispatching stablehlo.scatter:
  [reshape] Cannot reshape array of size 4096 into shape (4,1,1,1,1).
```

The 4096 is `cache_length` (the S dimension of the KV cache). The shape `(4,1,1,1,1)` has `batch_size=4` as its leading dimension. The error originates inside `update_cache_layer` → `_update_ragged`.

### Root cause

`_update_ragged` used `jax.vmap` over a function containing `jax.lax.dynamic_update_slice`:

```python
def update_one(cache_k, ..., pos):
    idx = pos % max_cache_len
    cache_k = lax.dynamic_update_slice(cache_k, new_k, (idx, 0, 0))
    ...

jax.vmap(update_one)(key_cache_layer, ..., write_pos_B)
```

When the start indices differ per batch element (each element writes to a different cache position), JAX cannot lower the vmapped `dynamic_update_slice` to a single HLO `DynamicUpdateSlice` op. Instead, it lowers to `stablehlo.scatter` with batched index computation. The MPS backend's scatter implementation fails when attempting to reshape the scatter indices into the operand's rank.

Note: the `_update_dense` path (used during prefill) also uses scatter (`.at[batch_indices, cache_indices].set(..., mode="drop")`), but that scatter pattern — advanced integer indexing — lowers to a different HLO scatter configuration that MPS handles correctly. The vmap-of-`dynamic_update_slice` pattern is specifically what fails.

### Fix (`gemma_jax/core/cache.py`, both `_update_ragged` definitions)

Replaced the `jax.vmap(dynamic_update_slice)` pattern with `jnp.where` and a one-hot positional mask. This is purely element-wise (comparison + select) and generates no scatter HLO:

```python
B, S, K, H = key_cache_layer.shape
idx = write_pos_B % S                                       # (B,)
mask = jnp.arange(S)[None, :] == idx[:, None]               # (B, S) one-hot
mask_kv = mask[:, :, None, None]                             # (B, S, 1, 1)

updated_k = jnp.where(mask_kv, key_proj[:, None, :, :], key_cache_layer)
updated_v = jnp.where(mask_kv, value_proj[:, None, :, :], val_cache_layer)
```

The broadcast semantics: `key_proj[:, None, :, :]` is `(B, 1, K, H)`, which broadcasts against the `(B, S, 1, 1)` mask and the `(B, S, K, H)` cache layer. Only the single position where `mask` is True gets the new value; all other positions retain the old cache contents.

Trade-off: `jnp.where` touches every element in the cache layer (O(B·S·K·H)) vs. `dynamic_update_slice` which only writes O(B·K·H). For generation (single-token update, S=4096), this is ~4096× more memory traffic per layer per step. In practice the overhead is acceptable on Apple Silicon's unified memory architecture, and it eliminates the scatter entirely.

---

## Fix 3: int8 KV cache quantization produces silent wrong results on MPS

### Symptom

The model runs to completion with no errors but generates only PAD tokens (token ID 0), which decode to empty strings:

```
[Consumer thread] chunk 0, text='['', '', '', '']'
```

All batch elements produce identical empty output — a systematic failure, not a numerical precision issue.

### Root cause

The KV cache used int8 quantization for memory savings:

- **Storage**: `key`/`value` arrays stored as `int8` with per-head `bfloat16` scale factors
- **Write path**: `_quantize_to_int8()` converts bfloat16 projections to int8 via `(x / scale).round().astype(jnp.int8)`
- **Read path**: `lookup_layer()` dequantizes via `k_q.astype(jnp.bfloat16) * k_s[..., None]`

The MPS backend does not reliably support int8 tensor operations. Specifically, `astype(jnp.int8)` and/or int8 arithmetic (the `jnp.where` select between int8 tensors, int8→bfloat16 promotion) silently produce incorrect values (likely all-zeros). This means:

1. During prefill, `_update_dense` writes int8 zeros (or garbage) to the cache
2. During generation, `_update_ragged` writes int8 zeros to the cache
3. `lookup_layer` dequantizes zeros → bfloat16 zeros
4. Attention over all-zero keys produces uniform scores → softmax gives uniform weights → weighted sum of zero values = zero
5. After 34 transformer layers, residual connections with zero attention outputs still leave some signal from FFN, but the final logits are degenerate
6. `argmax` over near-uniform logits returns index 0 = PAD token

The fact that ALL batch elements produce identical empty output (rather than random garbage) is consistent with the cache being filled with a constant (zeros), making the model's behavior deterministic and degenerate regardless of input.

### Fix (`gemma_jax/core/cache.py`)

Three targeted changes disable int8 quantization while preserving the existing interface (scale arrays remain as all-ones no-ops):

**a) `init_cache`**: Allocate key/value as `bfloat16` instead of `int8`:

```python
# Before:
key=jnp.zeros(cache_shape, dtype=jnp.int8),
value=jnp.zeros(cache_shape, dtype=jnp.int8),

# After:
key=jnp.zeros(cache_shape, dtype=dtype),   # dtype=bfloat16
value=jnp.zeros(cache_shape, dtype=dtype),
```

**b) `_quantize_to_int8`**: No-op pass-through returning the input unchanged:

```python
def _quantize_to_int8(x: Array) -> tuple[Array, Array]:
    scale = jnp.ones(x.shape[:-1], dtype=x.dtype)
    return x, scale  # no quantization, unit scale
```

**c) `lookup_layer`**: Skip the int8→bfloat16 cast and scale multiplication:

```python
# Before:
k = k_q.astype(jnp.bfloat16) * k_s[..., None]
v = v_q.astype(jnp.bfloat16) * v_s[..., None]

# After:
k = self.key[layer]    # already bfloat16
v = self.value[layer]
```

**Memory impact**: The 4B model's KV cache at bfloat16 is ~2.3 GB vs ~1.15 GB at int8. Apple Silicon devices typically have 16–32 GB of unified memory, so this is acceptable. The scale arrays (`key_scale`, `value_scale`) persist as all-ones bfloat16 tensors (~9 MB total) — negligible overhead, and removing them would require interface changes throughout the codebase.

---

## Summary of MPS backend limitations encountered

| Operation | HLO lowering | MPS support | Workaround |
|---|---|---|---|
| `jax.experimental.io_callback()` | `EmitPythonCallback` custom-call | Not implemented | Move callback to host-side Python |
| `jax.vmap(lax.dynamic_update_slice)` with per-element indices | `stablehlo.scatter` (batched) | Crashes (reshape error) | Replace with `jnp.where` + one-hot mask |
| `astype(jnp.int8)` / int8 tensor ops | Various int8 HLO ops | Silently wrong results | Use bfloat16 throughout |
| `.at[idx].set(val, mode="drop")` (advanced indexing) | `stablehlo.scatter` (standard) | Works | — |
| `lax.dynamic_update_slice` (non-batched) | `DynamicUpdateSlice` | Works | — |
| `jax.lax.scan` | `stablehlo.while` | Works | — |
| bfloat16 arithmetic | Standard f16/f32 ops | Works | — |

---

## Files modified

- `examples/multi_turn_chat.py` — removed `io_callback`, moved chunk enqueue to host
- `gemma_jax/core/cache.py` — scatter-free `_update_ragged`, disabled int8 quantization
