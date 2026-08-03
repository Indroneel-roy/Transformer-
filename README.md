# Transformer From Scratch

A from-scratch PyTorch implementation of the Transformer architecture introduced in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — built module by module, from self-attention up to a full encoder-decoder model.

This repository is meant as both a **learning resource** and a **reference implementation**: each core component is developed and explained in its own notebook before being assembled into a complete, working transformer.

---

## Repository Structure

| File | Description |
|---|---|
| [`1_Self_attention_in_transformer.ipynb`](./1_Self_attention_in_transformer.ipynb) | Scaled dot-product self-attention from first principles |
| [`2_Multi_head_attention.ipynb`](./2_Multi_head_attention.ipynb) | Extending single-head attention to multi-head attention |
| [`3_Positional_encoding.ipynb`](./3_Positional_encoding.ipynb) | Sinusoidal positional encodings for sequence order |
| [`4_Layer_Normalization.ipynb`](./4_Layer_Normalization.ipynb) | Layer normalization and its role in stabilizing training |
| [`5_Encoder.ipynb`](./5_Encoder.ipynb) | Assembling the encoder block (attention + feed-forward + residuals) |
| [`6_Decoder.ipynb`](./6_Decoder.ipynb) | Assembling the decoder block (masked self-attention + cross-attention) |
| [`Full_transformer.py`](./Full_transformer.py) | Complete encoder-decoder Transformer model, ready to import and train |
| [`Note/`](./Note) | Supplementary notes on the architecture |

Each notebook builds on the concepts from the previous one, so they're best read in numerical order if you're following along as a tutorial.

---

## Model Description

This implementation follows the encoder-decoder Transformer architecture of Vaswani et al. (2017). Below is a formal description of each component, matching the notation used in the original paper.

### 1. Input Representation

Given an input sequence of tokens, each token is mapped to a dense vector via a learned embedding matrix, scaled by √d_model:

```
E(x) = Embedding(x) · √d_model
```

Since the model contains no recurrence or convolution, positional information is injected via **sinusoidal positional encodings**, added directly to the embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

where `pos` is the position in the sequence and `i` indexes the embedding dimension. The final input to the encoder/decoder stack is:

```
Z0 = E(x) + PE(x)
```

### 2. Scaled Dot-Product Attention

The core attention mechanism maps a set of queries `Q`, keys `K`, and values `V` to a weighted output:

```
Attention(Q, K, V) = softmax( QKᵀ / √dk ) · V
```

The scaling factor `1/√dk` prevents the dot products from growing too large in magnitude, which would push the softmax into regions with extremely small gradients.

### 3. Multi-Head Attention

Rather than performing a single attention function, queries, keys, and values are linearly projected `h` times into `dk`, `dk`, and `dv`-dimensional subspaces, attention is applied in parallel, and the results are concatenated and projected again:

```
headi = Attention(Q·WQi, K·WKi, V·WVi)
MultiHead(Q, K, V) = Concat(head1, ..., headh) · WO
```

This allows the model to jointly attend to information from different representation subspaces at different positions.

### 4. Position-wise Feed-Forward Network

Each encoder/decoder layer contains a fully connected feed-forward network, applied identically to each position:

```
FFN(x) = max(0, x·W1 + b1) · W2 + b2
```

### 5. Residual Connections and Layer Normalization

Every sub-layer (attention or feed-forward) is wrapped in a residual connection followed by layer normalization:

```
LayerNorm(x + Sublayer(x))
```

This stabilizes training and allows gradients to flow more easily through deep stacks.

### 6. Encoder

Each of the `N` encoder layers consists of two sub-layers:

1. Multi-head self-attention over the input sequence
2. Position-wise feed-forward network

```
Zenc = EncoderLayer(Zenc)   (applied N times)
```

### 7. Decoder

Each of the `N` decoder layers consists of three sub-layers:

1. **Masked** multi-head self-attention (prevents positions from attending to future tokens)
2. Multi-head cross-attention over the encoder output (`Q` from decoder, `K`/`V` from encoder)
3. Position-wise feed-forward network

The causal mask ensures the prediction for position `i` depends only on known outputs at positions `< i`, preserving the auto-regressive property.

### 8. Output Layer

The decoder output is projected to the target vocabulary size and passed through a softmax to produce next-token probabilities:

```
P(y) = softmax(Zdec · Wout)
```

### Model Hyperparameters

| Symbol | Description | Typical (base) value |
|---|---|---|
| `d_model` | Embedding / hidden dimension | 512 |
| `h` | Number of attention heads | 8 |
| `dk`, `dv` | Dimension per head (`d_model / h`) | 64 |
| `d_ff` | Feed-forward inner dimension | 2048 |
| `N` | Number of encoder/decoder layers | 6 |
| `P_drop` | Dropout rate | 0.1 |

*(Update the table above with the actual values used in `Full_transformer.py` if they differ from the paper's base configuration.)*

---




## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP

---

## License

This project is licensed under the [MIT License](./LICENSE).

## Author

**Indroneel Roy**
