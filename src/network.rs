use rand::{seq::SliceRandom, Rng};

/// A single layer in a neural network.
/// `weights[neuron][input]` = connection strength, `biases[neuron]` = offset value.
pub struct Layer {
    /// Weight matrix: `weights[neuron][input]`
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: one per neuron
    pub biases: Vec<f32>,
}

/// Sigmoid: σ(z) = 1 / (1 + e^(-z)). Squashes values to (0, 1).
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// σ'(z) = σ(z)(1 − σ(z)). Used in backprop to chain the gradient.
fn sigmoid_derivative(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Sum of (input[i] * weights[i]) for one neuron's pre-activation.
fn dot_product(input: &[f32], weights: &[f32]) -> f32 {
    input.iter().zip(weights).map(|(x, w)| x * w).sum()
}

impl Layer {
    /// Creates a new layer with random weights and biases in range [-1.0, 1.0).
    pub fn new(input_size: usize, num_neurons: usize) -> Self {
        let mut rng = rand::rng();

        let weights = (0..num_neurons)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        let biases = (0..num_neurons)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        Self { weights, biases }
    }

    /// Forward pass: for each neuron, z = w·x + b, then output = σ(z).
    pub fn calculate_output(&self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.biases.len());
        let num_neurons = self.weights.len();

        for i in 0..num_neurons {
            let z = dot_product(&inputs, &self.weights[i]) + self.biases[i];
            outputs.push(sigmoid(z))
        }

        outputs
    }
}

/// Multilayer perceptron: sequential layers from input to output.
pub struct Network {
    layers: Vec<Layer>,
    /// Step size for gradient descent updates
    learning_rate: f32,
}

impl Network {
    /// Creates a network with architecture defined by `sizes`.
    /// Example: `&[784, 128, 10]` creates 784→128→10 (2 weight layers).
    pub fn new(sizes: &[usize], learning_rate: f32) -> Self {
        let mut layers = Vec::new();

        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }

        Self {
            layers,
            learning_rate,
        }
    }

    /// Forward pass through all layers. Returns final layer activations (e.g. 10 class scores).
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut current_output = input.to_vec();

        for layer in &self.layers {
            current_output = layer.calculate_output(&current_output);
        }

        current_output
    }

    /// Backpropagation: compute gradients of loss w.r.t. all weights and biases.
    /// Returns (weight_gradients, bias_gradients) for each layer. Uses MSE: (y - target)².
    pub fn backprop(
        &self,
        network_input: &[f32],
        target: &[f32],
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        // --- FORWARD PASS: save all z (pre-activation) and a (activation) for backprop ---
        let mut all_layers_outputs = vec![network_input.to_vec()];
        let mut all_weighted_sums = vec![];

        let mut current_layer_input = network_input.to_vec();

        for layer in &self.layers {
            let mut weighted_sum = Vec::new();
            let mut layer_outputs = Vec::new();

            for i in 0..layer.weights.len() {
                let z = dot_product(&current_layer_input, &layer.weights[i]) + layer.biases[i];
                weighted_sum.push(z);
                layer_outputs.push(sigmoid(z));
            }

            all_weighted_sums.push(weighted_sum);
            all_layers_outputs.push(layer_outputs.clone());

            current_layer_input = layer_outputs;
        }

        // --- BACKWARD PASS: compute δ (delta) per layer ---
        let mut layers_deltas: Vec<Vec<f32>> = Vec::new();
        let last_layer_weighted_sums = all_weighted_sums.last().expect("Weighted sums exist");
        let last_layer_outputs = all_layers_outputs.last().expect("Output exist");

        // Output layer: δ = (y - target) * σ'(z)  [derivative of MSE + chain rule]
        let mut current_layer_deltas: Vec<f32> = last_layer_outputs
            .iter()
            .zip(target.iter())
            .zip(last_layer_weighted_sums.iter())
            .map(|((&y, &target), &z)| (y - target) * sigmoid_derivative(z))
            .collect();

        layers_deltas.push(current_layer_deltas.clone());

        // Hidden layers: δ_l = (W_{l+1}^T · δ_{l+1}) * σ'(z_l). Walk from last-1 down to 0.
        for l in (0..self.layers.len() - 1).rev() {
            let mut next_delta: Vec<f32> = Vec::new();
            let current_layer_weighted_sums = &all_weighted_sums[l];
            let layer_to_right = &self.layers[l + 1];

            for i in 0..self.layers[l].weights.len() {
                // Backprop error from layer l+1: sum over next layer's neurons j
                let mut error_signal = 0.0;
                for j in 0..layer_to_right.weights.len() {
                    error_signal += layer_to_right.weights[j][i] * current_layer_deltas[j];
                }
                next_delta.push(error_signal * sigmoid_derivative(current_layer_weighted_sums[i]));
            }

            current_layer_deltas = next_delta.clone();
            layers_deltas.push(next_delta);
        }

        layers_deltas.reverse();

        // --- GRADIENTS: dC/dw = δ * a_in,  dC/db = δ ---
        let mut weight_gradients: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut bias_gradients = Vec::new();

        for i in 0..self.layers.len() {
            let layer_deltas = &layers_deltas[i];
            let layer_inputs = &all_layers_outputs[i]; // activations into this layer

            bias_gradients.push(layer_deltas.clone());

            let mut layer_weight_gradients = Vec::new();
            for &delta in layer_deltas {
                let mut neuron_weight_gradients = Vec::new();
                for &activation in layer_inputs {
                    neuron_weight_gradients.push(delta * activation);
                }
                layer_weight_gradients.push(neuron_weight_gradients);
            }

            weight_gradients.push(layer_weight_gradients);
        }

        (weight_gradients, bias_gradients)
    }

    pub fn update_mini_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
        let mut total_grad_w: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut total_grad_b: Vec<Vec<f32>> = Vec::new();

        for layer in &self.layers {
            let zero_weights: Vec<Vec<f32>> =
                vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            total_grad_w.push(zero_weights);

            let zero_biases: Vec<f32> = vec![0.0; layer.biases.len()];
            total_grad_b.push(zero_biases);
        }

        for (input, target) in batch {
            let (delta_grad_w, delta_grad_b) = self.backprop(input, target);
            for l in 0..self.layers.len() {
                total_grad_b[l]
                    .iter_mut()
                    .zip(&delta_grad_b[l])
                    .for_each(|(total_b, delta_b)| {
                        *total_b += delta_b;
                    });

                total_grad_w[l].iter_mut().zip(&delta_grad_w[l]).for_each(
                    |(total_grad_neuron_w, delta_grad_neuron_w)| {
                        total_grad_neuron_w
                            .iter_mut()
                            .zip(delta_grad_neuron_w)
                            .for_each(|(total_w, delta_w)| {
                                *total_w += delta_w;
                            });
                    },
                );
            }
        }

        // Update weights based of mini batch results
        let step = self.learning_rate / batch.len() as f32;
        for l in 0..self.layers.len() {
            self.layers[l]
                .biases
                .iter_mut()
                .zip(&total_grad_b[l])
                .for_each(|(bias, grad_bias)| {
                    *bias -= step * grad_bias;
                });

            self.layers[l]
                .weights
                .iter_mut()
                .zip(&total_grad_w[l])
                .for_each(|(neuron_weights, grad_weights)| {
                    neuron_weights
                        .iter_mut()
                        .zip(grad_weights)
                        .for_each(|(w, gw)| {
                            *w -= step * gw;
                        });
                });
        }
    }

    /// Stohastic Gradient Descent
    pub fn sgd(
        &mut self,
        mut training_data: Vec<(Vec<f32>, Vec<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        test_data: Option<&[(Vec<f32>, u8)]>,
    ) {
        for i in 0..epochs {
            training_data.shuffle(&mut rand::rng());

            for batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(batch);
            }

            if let Some(data) = test_data {
                let success = self.evaluate(data);
                println!("Epoch {}: {} / {}", i, success, data.len());
            } else {
                println!("Epoch {} finished", i);
            }
        }
    }

    pub fn evaluate(&self, test_data: &[(Vec<f32>, u8)]) -> usize {
        let mut test_results = 0;

        for (input, target) in test_data {
            let output = self.predict(input);

            // Find neuron index with the bigges signal
            let (predicted, _) =
                output
                    .iter()
                    .enumerate()
                    .fold((0, 0.0), |(max_idx, max_val), (idx, &val)| {
                        if val > max_val {
                            (idx, val)
                        } else {
                            (max_idx, max_val)
                        }
                    });

            if predicted == *target as usize {
                test_results += 1;
            }
        }
        test_results
    }
}
