# Advanced KV Cache Strategies in JAX: From Ring Buffers to PagedAttention

**User:**
> question: does the maxtext/ JetStream cache implemenation use a ring buffer? i suspect that it does not?

**Model:**
You are correct—MaxText and JetStream do **not** use ring buffer caches by default.

Instead, they use **dense, statically-sized pre-allocated arrays** and update them continuously using `jax.lax.dynamic_update_slice` (or Pallas/Triton equivalents).

They track the current position (the length of the sequence) and simply insert new tokens at that index. If the sequence length exceeds the statically provisioned maximum context length (e.g., `max_target_length`), they either truncate, fail to compile, or raise an out-of-bounds error during generation, rather than silently wrapping around and destroying early context.

This aligns with Google's production approach: provision massive static caches (e.g. 32k, 128k, or 1M+ tokens on TPUs via PagedAttention/PagedKV) and guarantee deterministic attention across the entire context window, treating the hardware limits as a hard boundary rather than falling back to destructive ring wrapping.

***

**User:**
> what advanced cache strategies do MaxText and JetStream implement?

**Model:**
Both MaxText and JetStream implement highly advanced caching strategies to maximize throughput and memory efficiency in production TPU environments.

Here are the core advanced strategies they use instead of simple ring caches:

### 1. PagedAttention (Paged KV Cache)
Instead of allocating one massive, contiguous memory block for a sequence (which leads to fragmentation and wasted memory when sequences don't reach maximum length), MaxText uses **Paged KV Caching**.
- Memory is broken into small, fixed-size blocks ("pages").
- A central block manager dynamically allocates pages to sequences as they grow during decoding.
- This allows for >95% memory utilization and allows the server to batch significantly more sequences simultaneously without out-of-memory errors.

### 2. Radix/Prefix Caching
When multiple users send prompts that share a common prefix (e.g., a massive system prompt, a few-shot example template, or a shared document), JetStream implements **Prefix Caching**.
- The KV cache for the shared prefix is computed once.
- Any subsequent request with that exact prefix "borrows" the pre-computed KV cache states via a Radix Tree lookup.
- This eliminates redundant prefill computations entirely, drastically lowering Time-to-First-Token (TTFT) for overlapping prompts.

### 3. Asymmetric Prefill vs. Decode Caches
MaxText structurally separates the KV Cache layouts used during the "prefill" phase vs the "autoregressive decode" phase.
- **Prefill Cache:** Optimized for massive chunked matrix multiplications along the sequence dimension.
- **Autoregressive Cache:** Optimized for low-latency vector-matrix lookups (reading all previous keys/values to compute attention for a single new token).
- During inference, JetStream explicitly manages the transition between these two representations.

### 4. Continuous Batching (In-Flight Batching)
Because they aren't bound by static sequence lengths or rigid ring boundaries, these engines use iteration-level batching. When sequence A finishes generating, its slots/pages are immediately freed and sequence C is slotted in to begin prefilling on the very next step, without waiting for sequence B to finish.

### 5. Multi-Axis Sharding
The caches are designed to be optimally partitioned across the TPU mesh using JAX's `NamedSharding`. The sequence, head, and batch dimensions are distributed across multiple TPUs (Tensor Parallelism and Sequence Parallelism) to keep the massive KV cache distributed in HBM (High Bandwidth Memory) rather than isolated on a single device.

