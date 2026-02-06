# rust-mnist-mlp

MNIST digit classifier using a multilayer perceptron (MLP) and backpropagation. Implemented in Rust with sigmoid activation and mini-batch stochastic gradient descent (SGD).

---

## Quick start

**Prerequisites:** Rust toolchain (e.g. `rustup`), MNIST data in the `data/` folder.

1. **Get MNIST data** (if not already present):
   - Place the four IDX files in `data/`: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.
   - Or download and extract from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist) (or a mirror) into `data/`.

2. **Run the program:**
   ```bash
   cargo run
   ```
   The program loads MNIST, trains the network (784 → 30 → 10) for 30 epochs, then runs inference on a custom image `seven.png`. Put a 28×28-friendly digit image at `seven.png` (or change the path in `main.rs`).

3. **Optional:** To try your own digit image, use the same name in code or call `data_loader::create_from_img("your_image.png")` to preprocess it to MNIST-style 28×28 grayscale.

---

## Project structure and workflow

| Step | What happens | Where in code |
|------|----------------|----------------|
| **Data** | Load MNIST, normalize pixels to [0,1], one-hot encode labels | `data_loader::MnistDataSet::new()`, `vectorized_result()` |
| **Custom image** | Load and preprocess a hand-drawn digit to 28×28 | `data_loader::create_from_img()`, `prepare_mnist_image()` |
| **Build network** | Create MLP with given layer sizes and learning rate | `Network::new(&[784, 30, 10], learning_rate)` |
| **Training** | Shuffle data, split into mini-batches, backprop + gradient step per batch | `Network::sgd()`, which calls `update_mini_batch()` |
| **Single batch step** | For each sample: forward → deltas → gradients; then average and update weights | `update_mini_batch()` → `backprop()` → `forward()`, `compute_deltas()`, `compute_gradients()` |
| **Inference** | Forward pass only; predicted digit = argmax of output | `Network::predict()`, `Network::prediction_to_digit()` |
| **Evaluation** | Count correct predictions on test set | `Network::evaluate()` |

**Loss:** MSE per output, $C = \frac{1}{2} \sum (y - t)^2$.  
**Update rule:** $w \leftarrow w - (\eta/|\text{batch}|) \sum \nabla w$, $b \leftarrow b - (\eta/|\text{batch}|) \sum \nabla b$.

---

## Notation

- **L** — number of weight layers in the network.
- **$w^{(l)}_{jk}$** — weight from neuron **k** in layer **l−1** to neuron **j** in layer **l** (in code: `weights[j][k]` for that layer).
- **$b^{(l)}_{j}$** — bias of neuron **j** in layer **l**.
- **$z^{(l)}_{j}$** — pre-activation (weighted sum) of neuron **j** in layer **l**.
- **$a^{(l)}_{j}$** — activation (output) of neuron **j** in layer **l**; **$a^{(0)}$** is the network input.
- **$\sigma(z)$** — Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **$\sigma'(z)$** — derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
- **$t$** (or **$y$** in target) — target output (one-hot for training).

---

## Stage 1: Forward pass

For each layer **l** from 1 to L we compute pre-activation and activation; these are cached for the backward pass.

**Weighted sum (pre-activation):**
$$
z^{(l)}_{j} = \sum_{k} w^{(l)}_{jk} \cdot a^{(l-1)}_{k} + b^{(l)}_{j}
$$

**Activation:**
$$
a^{(l)}_{j} = \sigma(z^{(l)}_{j})
$$

**In code:** `Network::forward()` (used inside `backprop()`). It returns all $a$ (including input) and all $z$ per layer. Single-layer forward is `Layer::calculate_output()`; full inference without cache is `Network::predict()`.

---

## Stage 2: Backward pass (deltas)

We compute the error signal $\delta$ for each layer, from output back to input.

### Output layer ($\delta^{(L)}$)

$$
\delta^{(L)}_{j} = (a^{(L)}_{j} - t_{j}) \cdot \sigma'(z^{(L)}_{j})
$$

$t_j$ is the target (one-hot) for output $j$. This is the derivative of the MSE loss with respect to $z^{(L)}_j$ via the chain rule.

### Hidden layers ($\delta^{(l)}$)

Error is propagated backward using the weights of the *next* layer:

$$
\delta^{(l)}_{j} = \left( \sum_{k} w^{(l+1)}_{kj} \cdot \delta^{(l+1)}_{k} \right) \cdot \sigma'(z^{(l)}_{j})
$$

So we combine the deltas of layer $l+1$, weighted by the connections back to neuron $j$, then multiply by $\sigma'(z^{(l)}_j)$.

**In code:** `Network::compute_deltas()`. It takes the cached `all_layers_outputs` and `all_weighted_sums` from the forward pass, plus the target, and returns a `Vec<Vec<f32>>` of deltas (one `Vec<f32>` per layer).

---

## Stage 3: Gradients

Using the cached $a$ and the computed $\delta$, we get the gradients of the cost with respect to weights and biases.

**Bias gradient:**
$$
\frac{\partial C}{\partial b^{(l)}_{j}} = \delta^{(l)}_{j}
$$

**Weight gradient:**
$$
\frac{\partial C}{\partial w^{(l)}_{jk}} = a^{(l-1)}_{k} \cdot \delta^{(l)}_{j}
$$

**In code:** `Network::compute_gradients()`. It takes `layers_deltas` and `all_layers_outputs` and returns weight gradients and bias gradients (same shapes as the network parameters).

---

## Full training step (per sample and per batch)

1. **Forward:** Compute and store all $z$ and $a$ → `forward()`.
2. **Backward:** Compute $\delta$ for every layer → `compute_deltas()`.
3. **Gradients:** Compute $\partial C/\partial w$ and $\partial C/\partial b$ → `compute_gradients()`.
4. **Batch:** For a mini-batch, sum the gradients over all samples, then update:
   - $w \leftarrow w - (\eta/n) \sum \nabla w$, $b \leftarrow b - (\eta/n) \sum \nabla b$,
   where $n$ is the batch size → done inside `update_mini_batch()` after calling `backprop()` for each sample and accumulating.

**In code:** `backprop()` orchestrates steps 1–3. `update_mini_batch()` loops over the batch, calls `backprop()` for each (input, target), accumulates gradients, then applies the update with step size $\eta/n$.

---

## Summary

| Theory | Code |
|--------|------|
| Forward: $z^l = W^l a^{l-1} + b^l$, $a^l = \sigma(z^l)$ | `forward()`, `Layer::calculate_output()` |
| Output delta: $\delta^L = (a^L - t) \odot \sigma'(z^L)$ | `compute_deltas()` (output layer) |
| Hidden delta: $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$ | `compute_deltas()` (hidden loop) |
| $\partial C/\partial b = \delta$, $\partial C/\partial w = \delta \cdot a^{\mathrm{in}}$ | `compute_gradients()` |
| Mini-batch SGD update | `update_mini_batch()`, `sgd()` |

---

## Implementation note

Weight indexing in code: `weights[neuron_j][input_k]` corresponds to $w^{(l)}_{jk}$ (connection from input index $k$ to neuron index $j$). This matches the formula $z_j = \sum_k w_{jk} a_k + b_j$.
