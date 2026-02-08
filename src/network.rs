//! Feedforward neural network with sigmoid activations, trained by backpropagation
//! and mini-batch stochastic gradient descent (SGD).
//!
//! **Loss:** MSE per output, C = ½ Σ (y − t)².  
//! **Update:** w ← w − (η/|batch|) Σ ∇w,  b ← b − (η/|batch|) Σ ∇b.

use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

/// A single fully-connected layer: output = σ(W·input + b).
/// - `weights`: matrix of size (num_neurons × input_size); weights[neuron][input] is the connection strength.
/// - `biases`: one per neuron; shifts the activation threshold.
pub struct Layer {
    /// Weight matrix: `weights[neuron][input]`
    pub weights: Array2<f32>,
    /// Bias vector: one per neuron
    pub biases: Array1<f32>,
}

/// Sigmoid activation: σ(z) = 1 / (1 + e^(-z)). Maps any real number to (0, 1).
/// Used so outputs are bounded and differentiable everywhere.
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Derivative of sigmoid: σ'(z) = σ(z)(1 − σ(z)). Needed in backprop to chain the gradient through the layer.
fn sigmoid_derivative(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

impl Layer {
    /// Creates a new layer with random weights and biases in [-1.0, 1.0).
    /// Random init breaks symmetry so neurons can learn different features.
    pub fn new(input_size: usize, num_neurons: usize) -> Self {
        let mut rng = rand::rng();

        let weights: Array2<f32> =
            Array2::from_shape_fn((num_neurons, input_size), |_| rng.random_range(-1.0..1.0));

        let biases: Array1<f32> =
            Array1::from_shape_fn(num_neurons, |_| rng.random_range(-1.0..1.0));

        Self { weights, biases }
    }

    /// Forward pass: z = W·input + b, then output = σ(z) for each neuron.
    pub fn calculate_output(&self, inputs: ArrayView1<f32>) -> Array1<f32> {
        let z: Array1<f32> = self.weights.dot(&inputs) + &self.biases;
        let outputs = z.mapv(sigmoid);
        outputs
    }
}

/// Multilayer perceptron: a stack of fully-connected layers, input → layer1 → … → output.
pub struct Network {
    layers: Vec<Layer>,
    /// Learning rate η: step size for gradient descent (w ← w − η·∇C).
    learning_rate: f32,
}

impl Network {
    /// Builds a network from layer sizes. E.g. [784, 30, 10] → input 784, hidden 30, output 10 (two weight matrices).
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

    /// Forward pass: feed input through each layer in order. Returns the last layer’s activations (e.g. 10 class scores).
    pub fn predict(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut current_output = input.to_owned();
        for layer in &self.layers {
            current_output = layer.calculate_output(current_output.view());
        }
        current_output
    }

    /// Backpropagation: compute gradients of the cost C w.r.t. every weight and bias for one (input, target) pair.
    /// Steps: forward (cache activations a and weighted sums z) → compute_deltas δ → compute_gradients from δ and a.
    pub fn backprop(
        &self,
        network_input: ArrayView1<f32>,
        target: ArrayView1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let (all_layers_outputs, all_weighted_sums) = self.forward(network_input);
        let layers_deltas = self.compute_deltas(&all_layers_outputs, &all_weighted_sums, target);
        let (weight_gradients, bias_gradients) =
            self.compute_gradients(layers_deltas, all_layers_outputs);
        (weight_gradients, bias_gradients)
    }

    /// Forward pass that saves every layer’s output (a) and pre-sigmoid sum (z) for backprop.
    /// For each layer: z = W·a_prev + b, then a = σ(z). Returns (list of a including input, list of z).
    fn forward(&self, network_input: ArrayView1<f32>) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let mut all_layers_outputs: Vec<Array1<f32>> = vec![network_input.to_owned()];
        let mut all_weighted_sums = vec![];

        for layer in &self.layers {
            let current_input = all_layers_outputs.last().unwrap().view();
            let z = layer.weights.dot(&current_input) + &layer.biases;
            let a = z.mapv(sigmoid);
            all_weighted_sums.push(z);
            all_layers_outputs.push(a);
        }

        (all_layers_outputs, all_weighted_sums)
    }

    /// Backward pass: compute the error signal δ for each layer (needed to get gradients).
    /// Output layer (MSE): δ^L = (a^L − t) ⊙ σ'(z^L). Hidden: δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l). (⊙ = element-wise.)
    fn compute_deltas(
        &self,
        all_layers_outputs: &Vec<Array1<f32>>,
        all_weighted_sums: &Vec<Array1<f32>>,
        target: ArrayView1<f32>,
    ) -> Vec<Array1<f32>> {
        let mut layers_deltas: Vec<Array1<f32>> = Vec::new();
        let last_layer_weighted_sums = all_weighted_sums.last().expect("Weighted sums exist");
        let last_layer_outputs = all_layers_outputs.last().expect("Output exist");

        // Output layer: δ = (prediction − target) ⊙ σ'(z); this is ∂C/∂z for the last layer.
        let mut current_layer_deltas =
            (last_layer_outputs - &target) * last_layer_weighted_sums.mapv(sigmoid_derivative);
        layers_deltas.push(current_layer_deltas.clone());

        // Hidden layers, from last hidden back to first: δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l).
        for l in (0..self.layers.len() - 1).rev() {
            let layer_to_right = &self.layers[l + 1];
            current_layer_deltas = layer_to_right.weights.t().dot(&current_layer_deltas)
                * all_weighted_sums[l].mapv(sigmoid_derivative);
            layers_deltas.push(current_layer_deltas.clone());
        }

        layers_deltas.reverse();
        layers_deltas
    }

    /// From deltas δ and cached activations a, compute ∂C/∂W and ∂C/∂b for each layer.
    /// ∂C/∂b = δ;  ∂C/∂W_{j,k} = δ_j · a^{l−1}_k (outer product of δ and previous layer’s output).
    fn compute_gradients(
        &self,
        layers_deltas: Vec<Array1<f32>>,
        all_layers_outputs: Vec<Array1<f32>>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut weight_gradients: Vec<Array2<f32>> = Vec::new();

        for i in 0..self.layers.len() {
            // Reshape δ to (N×1) and a^{l−1} to (1×M) so δ * a gives (N×M) = ∂C/∂W for this layer.
            let layer_deltas_matrix = &layers_deltas[i].view().insert_axis(Axis(1));
            let layer_inputs = &all_layers_outputs[i].view().insert_axis(Axis(0));
            let layer_weight_gradients = layer_deltas_matrix * layer_inputs;
            weight_gradients.push(layer_weight_gradients);
        }

        // Bias gradient is just the delta for that layer.
        let bias_gradients: Vec<Array1<f32>> = layers_deltas;
        (weight_gradients, bias_gradients)
    }

    /// One mini-batch step: compute gradients for each sample (in parallel), average them, then update weights and biases.
    /// Update: w ← w − (η/n)·Σ∇w,  b ← b − (η/n)·Σ∇b, where n = batch size.
    pub fn update_mini_batch(&mut self, batch: &[(Array1<f32>, Array1<f32>)]) {
        // Backprop for each (input, target) in the batch (parallel over samples).
        let delta_grads: Vec<(Vec<Array2<f32>>, Vec<Array1<f32>>)> = batch
            .par_iter()
            .map(|(input, target)| self.backprop(input.view(), target.view()))
            .collect();

        // Sum gradients across the batch (start from first sample, add the rest).
        let mut total_grad_w = delta_grads[0].0.clone();
        let mut total_grad_b = delta_grads[0].1.clone();
        for i in 1..delta_grads.len() {
            for l in 0..self.layers.len() {
                total_grad_w[l] += &delta_grads[i].0[l];
                total_grad_b[l] += &delta_grads[i].1[l];
            }
        }

        // Gradient descent: subtract (learning_rate / n) * average gradient from each weight and bias.
        let step = self.learning_rate / batch.len() as f32;
        for l in 0..self.layers.len() {
            self.layers[l].weights.scaled_add(-step, &total_grad_w[l]);
            self.layers[l].biases.scaled_add(-step, &total_grad_b[l]);
        }
    }

    /// Stochastic gradient descent: train for `epochs` passes over the data.
    /// Each epoch: shuffle training data, split into mini-batches, run update_mini_batch on each.
    /// If `test_data` is provided, report correct count after every epoch.
    pub fn sgd(
        &mut self,
        mut training_data: Vec<(Array1<f32>, Array1<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        test_data: Option<&[(Array1<f32>, u8)]>,
    ) {
        let total_batches = (training_data.len() + mini_batch_size - 1) / mini_batch_size;
        println!(
            "SGD: {} epochs, {} batches per epoch",
            epochs, total_batches
        );
        let total_start = Instant::now();
        for i in 0..epochs {
            let epoch_start = Instant::now();
            training_data.shuffle(&mut rand::rng());
            println!("Epoch {}/{} ...", i + 1, epochs);
            for batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(batch);
            }

            if let Some(data) = test_data {
                let success = self.evaluate(data);
                let epoch_secs = epoch_start.elapsed().as_secs_f64();
                println!(
                    "  Epoch {}/{}: {} / {} correct ({:.2}s)",
                    i + 1,
                    epochs,
                    success,
                    data.len(),
                    epoch_secs
                );
            } else {
                let epoch_secs = epoch_start.elapsed().as_secs_f64();
                println!("Epoch {} finished ({:.2}s)", i, epoch_secs);
            }
        }
        let total_secs = total_start.elapsed().as_secs_f64();
        println!("Total training time: {:.2}s", total_secs);
    }

    /// Returns the number of test samples classified correctly. Prediction = argmax over the 10 outputs.
    pub fn evaluate(&self, test_data: &[(Array1<f32>, u8)]) -> usize {
        let mut test_results = 0;
        for (input, target) in test_data {
            let output = self.predict(input.view());
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

    /// Converts the network’s output (10 activations) to a digit 0–9: index of the maximum value.
    pub fn prediction_to_digit(prediction: ArrayView1<f32>) -> usize {
        prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }
}
