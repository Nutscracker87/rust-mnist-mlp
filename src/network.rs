//! Feedforward neural network with sigmoid activations, trained by backpropagation
//! and mini-batch stochastic gradient descent (SGD).
//!
//! **Loss:** MSE per output, C = ½ Σ (y − t)².  
//! **Update:** w ← w − (η/|batch|) Σ ∇w,  b ← b − (η/|batch|) Σ ∇b.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

/// A single layer in a neural network.
/// `weights[neuron][input]` = connection strength, `biases[neuron]` = offset value.
pub struct Layer {
    /// Weight matrix: `weights[neuron][input]`
    pub weights: Array2<f32>,
    /// Bias vector: one per neuron
    pub biases: Array1<f32>,
}

/// Sigmoid: σ(z) = 1 / (1 + e^(-z)). Squashes values to (0, 1).
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// σ'(z) = σ(z)(1 − σ(z)). Used in backprop to chain the gradient.
fn sigmoid_derivative(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

impl Layer {
    /// Creates a new layer with random weights and biases in range [-1.0, 1.0).
    pub fn new(input_size: usize, num_neurons: usize) -> Self {
        let mut rng = rand::rng();

        let weights: Array2<f32> =
            Array2::from_shape_fn((num_neurons, input_size), |_| rng.random_range(-1.0..1.0));

        let biases: Array1<f32> =
            Array1::from_shape_fn(num_neurons, |_| rng.random_range(-1.0..1.0));

        Self { weights, biases }
    }

    /// Forward pass: for each neuron, z = w·x + b, then output = σ(z).
    pub fn calculate_output(&self, inputs: ArrayView1<f32>) -> Array1<f32> {
        let z: Array1<f32> = self.weights.dot(&inputs) + &self.biases;

        let outputs = z.mapv(sigmoid);

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
    pub fn predict(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut current_output = input.to_owned();

        for layer in &self.layers {
            current_output = layer.calculate_output(current_output.view());
        }

        current_output
    }

    /// Backpropagation: ∇C w.r.t. all weights and biases.
    /// Pipeline: forward (cache z, a) → compute_deltas (δ per layer) → compute_gradients (∂C/∂w, ∂C/∂b).
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

    /// Forward pass with cache for backprop.
    /// For each layer l:  z^l = W^l a^{l−1} + b^l,  a^l = σ(z^l).
    /// Returns (all a from input through last layer, all z per layer).
    fn forward(&self, network_input: ArrayView1<f32>) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let mut all_layers_outputs: Vec<Array1<f32>> = vec![network_input.to_owned()];

        let mut all_weighted_sums = vec![];

        for layer in &self.layers {
            // get last output for previous layer to use as input for the current (next) layer
            let current_input = all_layers_outputs.last().unwrap().view();
            let z = layer.weights.dot(&current_input) + &layer.biases;
            let a = z.mapv(sigmoid);

            all_weighted_sums.push(z);
            all_layers_outputs.push(a);
        }

        (all_layers_outputs, all_weighted_sums)
    }

    /// Backward pass: compute error signal δ for each layer.
    /// Output layer (MSE + chain rule):  δ^L = (a^L − t) ⊙ σ'(z^L).
    /// Hidden layers:  δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l).  (⊙ = element-wise)
    fn compute_deltas(
        &self,
        all_layers_outputs: &Vec<Array1<f32>>,
        all_weighted_sums: &Vec<Array1<f32>>,
        target: ArrayView1<f32>,
    ) -> Vec<Array1<f32>> {
        let mut layers_deltas: Vec<Array1<f32>> = Vec::new();
        let last_layer_weighted_sums = all_weighted_sums.last().expect("Weighted sums exist");
        let last_layer_outputs = all_layers_outputs.last().expect("Output exist");

        // Output layer: δ^L = (y − t) · σ'(z)
        let mut current_layer_deltas =
            (last_layer_outputs - &target) * last_layer_weighted_sums.mapv(sigmoid_derivative);
        layers_deltas.push(current_layer_deltas.clone());

        // Hidden: δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l),  l from L−1 down to 0
        for l in (0..self.layers.len() - 1).rev() {
            let layer_to_right = &self.layers[l + 1];
            current_layer_deltas = layer_to_right.weights.t().dot(&current_layer_deltas)
                * all_weighted_sums[l].mapv(sigmoid_derivative);

            // Add copy to the history
            layers_deltas.push(current_layer_deltas.clone());
        }

        layers_deltas.reverse();

        layers_deltas
    }

    /// Gradient of cost w.r.t. weights and biases from deltas and cached activations.
    /// ∂C/∂b^l = δ^l,  ∂C/∂W^l_{j,k} = δ^l_j · a^{l−1}_k.
    fn compute_gradients(
        &self,
        layers_deltas: Vec<Array1<f32>>,
        all_layers_outputs: Vec<Array1<f32>>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut weight_gradients: Vec<Array2<f32>> = Vec::new();

        for i in 0..self.layers.len() {
            // let layer_deltas = &layers_deltas[i];
            // let layer_inputs = &all_layers_outputs[i]; // a^{l−1}

            let layer_deltas_matrix = &layers_deltas[i].view().insert_axis(Axis(1)); // (N x 1) - "column"
            let layer_inputs = &all_layers_outputs[i].view().insert_axis(Axis(0)); // (1 x M) - "row"

            // ∂C/∂w_{j,k} = δ_j · a^{in}_k
            let layer_weight_gradients = layer_deltas_matrix * layer_inputs;

            weight_gradients.push(layer_weight_gradients);
            //weight_gradients.push(layer_weight_gradients);
        }

        let bias_gradients: Vec<Array1<f32>> = layers_deltas;

        (weight_gradients, bias_gradients)
    }

    /// Mini-batch gradient step: sum ∇ over batch, then  w ← w − (η/n)Σ∇w,  b ← b − (η/n)Σ∇b.
    pub fn update_mini_batch(&mut self, batch: &[(Array1<f32>, Array1<f32>)]) {
        let delta_grads: Vec<(Vec<Array2<f32>>, Vec<Array1<f32>>)> = batch
            .par_iter()
            .map(|(input, target)| self.backprop(input.view(), target.view()))
            .collect();

        // Create zero matrixes for totals
        let mut total_grad_w = delta_grads[0].0.clone();
        let mut total_grad_b = delta_grads[0].1.clone();

        // Sum deltas
        for i in 1..delta_grads.len() {
            for l in 0..self.layers.len() {
                total_grad_w[l] += &delta_grads[i].0[l];
                total_grad_b[l] += &delta_grads[i].1[l];
            }
        }

        // 3. Update weights using graients
        let step = self.learning_rate / batch.len() as f32;
        for l in 0..self.layers.len() {
            // Використовуємо scaled_add для максимальної швидкості
            self.layers[l].weights.scaled_add(-step, &total_grad_w[l]);
            self.layers[l].biases.scaled_add(-step, &total_grad_b[l]);
        }
    }

    /// Stochastic gradient descent: train for `epochs`, each epoch shuffle data,
    /// split into mini-batches, and call `update_mini_batch` on each. Optionally
    /// evaluate on `test_data` after each epoch (input + label as u8).
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
        for i in 0..epochs {
            training_data.shuffle(&mut rand::rng());
            println!("Epoch {}/{} ...", i + 1, epochs);
            for batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(batch);
            }

            if let Some(data) = test_data {
                let success = self.evaluate(data);
                println!(
                    "  Epoch {}/{}: {} / {} correct",
                    i + 1,
                    epochs,
                    success,
                    data.len()
                );
            } else {
                println!("Epoch {} finished", i);
            }
        }
    }

    /// Returns how many test samples were classified correctly. Prediction = argmax of output.
    pub fn evaluate(&self, test_data: &[(Array1<f32>, u8)]) -> usize {
        let mut test_results = 0;

        for (input, target) in test_data {
            let output = self.predict(input.view());

            // Predicted class = index of the output neuron with highest activation
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

    pub fn prediction_to_digit(prediction: ArrayView1<f32>) -> usize {
        prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    // need to implement
    // will return pixels positions + center maybe
    // pub fn get_averaging_weights_distribution_of_pixels(&self)
    // {

    // }

    // pub fn centering_training_sample_inside_pixels_heat_signature {}
    // pub fn scaling_training_sample_to_fit_pixels_heat_signature {}
}
