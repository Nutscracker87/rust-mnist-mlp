use rand::Rng;

/// A single layer in a neural network.
/// `weights[neuron][input]` = connection strength, `biases[neuron]` = offset value.
pub struct Layer {
    /// Weight matrix: `weights[neuron][input]`
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: one per neuron
    pub biases: Vec<f32>,
}

/// Sigmoid activation function, that implement a standart logigistic function: 1 / (1 + e^(-z))
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_derivative(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Computes the dot product of input and weights
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

    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut current_output = input.to_vec();

        for layer in &self.layers {
            current_output = layer.calculate_output(&current_output);
        }

        current_output
    }

    pub fn backprop(
        &self,
        network_input: &[f32],
        target: &[f32],
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        // --- FORWARD PASS ---
        // Store all intermediate values for the backward pass
        let mut all_layers_outputs = vec![network_input.to_vec()];
        let mut all_weighted_sums = vec![];

        let mut current_layer_input = network_input.to_vec();

        for layer in &self.layers {
            let mut weighted_sum = Vec::new();
            let mut layer_outputs = Vec::new();

            // layer.weights.len(): num of neurons in the layer
            for i in 0..layer.weights.len() {
                // z = w * x + b
                let z = dot_product(&current_layer_input, &layer.weights[i]) + layer.biases[i];
                weighted_sum.push(z);
                // a = sigmoid(z)
                layer_outputs.push(sigmoid(z));
            }

            all_weighted_sums.push(weighted_sum);
            all_layers_outputs.push(layer_outputs.clone());

            current_layer_input = layer_outputs;
        }

        // --- BACKWARD PASS ---
        // 1. Calculate the error (delta) for the output layer
        let mut layers_deltas: Vec<Vec<f32>> = Vec::new();
        let last_layer_weighted_sums = all_weighted_sums.last().expect("Weighted sums exist");
        let last_layer_outputs = all_layers_outputs.last().expect("Output exist");

        let mut current_delta: Vec<f32> = last_layer_outputs
            .iter()
            .zip(target.iter())
            .zip(last_layer_weighted_sums.iter())
            .map(|((&y, &target), &z)| (y - target) * sigmoid_derivative(z))
            .collect();

        layers_deltas.push(current_delta.clone());

        // loop over layers starting from last - 1 to the first
        for l in (0..self.layers.len() - 1).rev() {
            let mut next_delta: Vec<f32> = Vec::new();
            let current_layer_weighted_sums = &all_weighted_sums[l];

            let layer_to_right = &self.layers[l + 1];

            // loop over layer[l] neurons
            for i in 0..self.layers[l].weights.len() {
                let mut error_signal = 0.0;
                // loop over layer_to_rights neurons
                for j in 0..layer_to_right.weights.len() {
                    error_signal += layer_to_right.weights[j][i] * current_delta[j];
                }

                next_delta.push(error_signal * sigmoid_derivative(current_layer_weighted_sums[i]));
            }

            current_delta = next_delta.clone();
            layers_deltas.push(next_delta);
        }

        layers_deltas.reverse();

        // CALCULATE Gradients for each neuron
        let mut weight_gradients: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut bias_gradients = Vec::new();

        for i in 0..self.layers.len() {
            let layer_deltas = &layers_deltas[i];
            // This is a_in for this layer. (we started with layer 0 input training set, so for each layer
            // layer[i] is an inputs and layer[i+1] is layer output)
            let layer_input = &all_layers_outputs[i];

            // 1. Bias gradient: dC/db = delta
            bias_gradients.push(layer_deltas.clone());

            // 2. Weight gradient: dC/dw = delta * a_in
            let mut layer_weight_gradients = Vec::new();
            for &delta in layer_deltas {
                let mut neuron_weight_gradients = Vec::new();
                for &activation in layer_input {
                    neuron_weight_gradients.push(delta * activation);
                }

                layer_weight_gradients.push(neuron_weight_gradients);
            }

            weight_gradients.push(layer_weight_gradients);
        }

        (weight_gradients, bias_gradients)
    }
}
