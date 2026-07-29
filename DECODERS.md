# Decoder architectures

TSP-D decoding is autoregressive: at each step the policy picks a **truck** next node, then a **drone** next node, under action masks from the environment. All variants share the same Kool `AttentionEncoder` over coordinates and the same `StepDecoder` API; only the step-wise scoring network differs.

Architecture name = `{decoder}_{dynamics}` (e.g. `attention_model_on`).

---

## Shared pipeline

```
coords [B, N, 2]
    → AttentionEncoder
    → node_embeddings [B, N, H], graph_embedding [B, H]

each truck/drone decision:
    prev_embed [B, H]          # embedding of last chosen node (or depot)
    dynamic [B, N, 1]          # travel-time features from Env (optional)
    avail_actions [B, N]       # 1 = feasible
        → optional DynamicEncoder (Linear 1→H) on dynamic
        → StepDecoder.step(...) → logits [B, N]
        → mask + Categorical / argmax → action, log π
```

`Policy` (`src/models/policy.py`) owns the encoder, optional dynamic `DynamicEncoder`, and one `StepDecoder`. Decoders implement:

| Method | Role |
| --- | --- |
| `reset(encoder_output, batch_size)` | Episode state (LSTM hidden, AM fixed keys, …) |
| `step(..., prev_embed, dynamic_hidden, state, avail_actions)` | Unmasked logits `[B, N]` + new state |

Masking is applied in `Policy.forward`, not inside every decoder (AM uses the mask inside the glimpse; final logits are still masked by the policy).

### Dynamics on vs off

Travel-time features from the env are encoded with a pointwise `DynamicEncoder` (`Linear(1→H)`) → `[B, H, N]` when `dynamics=on`. How that tensor enters the decoder:

| Decoder | `dynamics=on` | `dynamics=off` |
| --- | --- | --- |
| `tspd_lstm` | Added into additive pointer energy via `project_d` | Pointer uses static + query only |
| `attention_model` | Mean-pooled over nodes and concatenated into step context | Step context = previous embedding only |
| `lstm_pointer` | Mean-pooled and concatenated into `LSTMCell` input | Cell input = previous embedding only |

---

## 1. `tspd_lstm` — paper LSTM + additive pointer

**File:** `src/models/decoder/tspd_lstm.py`  
**Pointer:** `src/models/layers/pointer.py` (`PointerAttention`)

This is the decoder from the TSP-D RL paper: an LSTM consumes the previous node embedding; its hidden state is the query for an additive pointer over static node encodings.

**State:** LSTM `(h, c)`, zero-initialized each episode.

**Step:**

1. `prev_embed` → `nn.LSTM` → query `q` from the top-layer hidden state.
2. Static keys: `node_embeddings` reshaped to `[B, H, N]`.
3. Additive energy: \(u_i = v^\top \tanh(W_\text{ref}\, e_i + W_q\, q \[+ W_d\, d_i\])\).
4. Optional `use_tanh` clips logits with \(C\tanh(u)\).

**Why it exists:** baseline matching the published TSP-D architecture (recurrent decode + paper-style dynamic fusion inside the pointer).

---

## 2. `attention_model` — Kool Attention Model (AM)

**File:** `src/models/decoder/attention_model.py`

Port of the [Kool et al.](https://arxiv.org/abs/1803.08475) decoder to one TSP-D decision (no separate graph-level “first node” special case beyond depot embeddings provided by the policy).

**State (fixed for the episode after `reset`):**

- `fixed_context` — projection of graph embedding  
- `glimpse_keys`, `glimpse_values`, `logit_keys` — split of a linear projection of all node embeddings  

**Step:**

1. **Query:** `fixed_context + step_context_proj(prev_embed [‖ mean(dynamic)])`.
2. **Multi-head glimpse:** masked attention of the query over glimpse keys/values (infeasible nodes set to −∞).
3. **Pointer:** scaled dot-product of `final_query_proj(glimpse)` with `logit_keys`, then \(C\tanh(\cdot)\).

There is **no LSTM**. Recurrence is only through `prev_embed` (and optional dynamic context). The glimpse is a single attention layer, not a full transformer block.

**Why it exists:** compare a strong non-recurrent TSP decoder against the paper LSTM under the same encoder and TSP-D env.

---

## 3. `lstm_pointer` — Vinyals LSTM pointer

**File:** `src/models/decoder/lstm_pointer.py`  
**Pointer:** `src/models/layers/additive_pointer_attention.py`

Closer to the original Pointer Network (Vinyals et al.): an `LSTMCell` updates hidden/cell from the previous embedding (and optional pooled dynamics); an additive pointer scores nodes from the new hidden state.

**State:** `hidden` and `cell`, both initialized to the graph embedding (not zeros).

**Step:**

1. `LSTMCell(prev_embed [‖ pooled dynamic], (h, c))` → new `(h, c)`.
2. Logits: \(u_i = v^\top \tanh(W_e\, e_i + W_d\, h)\).

Unlike `tspd_lstm`, dynamics enter the **cell input**, not the pointer energy; the pointer itself is static-only additive attention.

**Why it exists:** classic pointer-network recurrent decode as a third axis in the ablation matrix.

---

## Comparison (quick)

| | `tspd_lstm` | `attention_model` | `lstm_pointer` |
| --- | --- | --- | --- |
| Recurrence | `nn.LSTM` | none (context only) | `LSTMCell` |
| Attention | additive pointer (+ optional dyn in energy) | MHA glimpse + scaled DP pointer | additive pointer (static) |
| Init state | zeros | projected keys + graph context | graph embedding as `(h, c)` |
| Dynamics hook | inside pointer | in step-context concat | in cell input concat |
| Typical refs | Bogyrbayeva et al. (TSP-D) | Kool AM | Vinyals Ptr-Net |

All six combinations (`decoder` × `dynamics`) are selected via Hydra:

```bash
uv run python -m src.experiments.run \
  decoder=attention_model dynamics=on \
  ...
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full training loop and repo layout.
