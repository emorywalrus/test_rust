use rand::Rng;
use std::f32::consts::E;
fn dot(vec1: Vec<f32>, vec2: Vec<f32>) -> f32 {
    if vec1.len() != vec2.len() {
        println!("vectors not same length error");
        return 0.0;
    } else {
        let mut sum: f32 = 0.0;
        let mut i = 0;
        while i < vec1.len() {
            sum += vec1[i] * vec2[i];
            i += 1;
        }
        return sum;
    }
}

fn multiply(vec: Vec<f32>, num: f32) -> Vec<f32> {
    let mut new_vec = Vec::with_capacity(vec.len());
    let mut i = 0;
    while i < new_vec.len() {
        new_vec.push(vec[i] * num);
        i += 1;
    }
    return new_vec;
}

fn add(vec1: Vec<f32>, vec2: Vec<f32>) -> Vec<f32> {
    if vec1.len() != vec2.len() {
        println!("vectors not same length error");
        return vec![0.0];
    } else {
        let mut new_vec: Vec<f32> = Vec::with_capacity(vec1.len());
        let mut i = 0;
        while i < vec1.len() {
            new_vec.push(vec1[i] + vec2[i]);
            i += 1;
        }
        return new_vec;
    }
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / {1.0 + f32::powf(E, x * -1.0)};
}

#[derive(Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32
}

impl Neuron {
    fn fire_neuron(&self, activations: Vec<f32>) -> f32 {
        return sigmoid(dot(activations, self.weights.clone()) + self.bias);
    }
    fn to_string(&self) -> String {
        let mut ret_str: String = String::new();
        let mut weight_str: String = String::new();
        let mut i = 0;
        weight_str.push_str("[");
        while i < self.weights.len() {
            weight_str.push_str(&self.weights[i].to_string());
            if i < self.weights.len() - 1 {
                weight_str.push_str(", ");
            }
            i += 1;
        }
        weight_str.push_str("]");
        ret_str.push_str("[");
        ret_str.push_str(&weight_str);
        ret_str.push_str(", ");
        ret_str.push_str(&self.bias.to_string());
        ret_str.push_str("]");
        return ret_str;
    }
    
    fn randomly_step_neuron(&self, step_size: f32) -> Neuron {
        let mut rng = rand::thread_rng();
        let bias_change: f32 = rng.gen_range(-1000..=1000) as f32 / 1000.0 * step_size;
        let new_bias = self.bias + bias_change;
        let mut weight_change: Vec<f32> = Vec::with_capacity(self.weights.len());
        let mut i = 0;
        while i < self.weights.len() {
            weight_change.push(rng.gen_range(-1000..=1000) as f32 / 1000.0  * step_size);
            i += 1;
        }
        let mut new_weights: Vec<f32> = add(self.weights.clone(), weight_change);
        return Neuron {
            weights: new_weights,
            bias: new_bias
        };
    }
}

fn blank_neuron(prev_length: usize) -> Neuron {
    let mut weights: Vec<f32> = Vec::with_capacity(prev_length);
    let bias: f32 = 0.0;
    let mut i = 0;
    while i < prev_length {
        weights.push(0.0);
        i += 1;
    }
    return Neuron {
        weights: weights,
        bias: bias
    };
}

struct Layer {
    neuron_vec: Vec<Neuron>
}

impl Layer {
    fn fire_layer(&self, activations: Vec<f32>) -> Vec<f32> {
        let mut ret_activations: Vec<f32> = Vec::new();
        let mut i = 0;
        while i < self.neuron_vec.len() {
            ret_activations.push(self.neuron_vec[i].fire_neuron(activations.clone()));
            i += 1;
        }
        return ret_activations;
    }

    fn to_string(&self) -> String {
        let mut ret_str = String::from("[");
        let mut i = 0;
        while i < self.neuron_vec.len() {
            ret_str.push_str(&self.neuron_vec[i].to_string());
            if i < self.neuron_vec.len() - 1 {
                ret_str.push_str(",  ");
            }
            i += 1;
        }
        return ret_str;
    }

    fn randomly_step_layer(&self, step_size: f32) -> Layer{
        let mut i = 0;
        let mut new_neuron_vec: Vec<Neuron> = Vec::with_capacity(self.neuron_vec.len());
        while i < self.neuron_vec.len() {
            new_neuron_vec.push(self.neuron_vec[i].randomly_step_neuron(step_size));
            i += 1;
        }
        return Layer {neuron_vec: new_neuron_vec};
    }
}

fn blank_layer(prev_size: usize, size: usize) -> Layer {
    let blank_neuron: Neuron = blank_neuron(prev_size);
    let mut blank_neuron_vec: Vec<Neuron> = Vec::with_capacity(size);
    let mut i = 0;
    while i < size {
        blank_neuron_vec.push(blank_neuron.clone());
        i += 1;
    }
    return Layer{neuron_vec: blank_neuron_vec};
}

struct Net {
    layer_vec: Vec<Layer>,
    structure: Vec<usize>
}

impl Net {
    fn fire_net(&self, mut activations: Vec<f32>) -> Vec<f32> {
        let mut i: usize = 0;
        while i < self.layer_vec.len() {
            activations = self.layer_vec[i].fire_layer(activations);
            i += 1;
        }
        return activations;
    }
    fn to_string(&self)-> String {
        let mut ret_str: String = String::from("[");
        let mut i = 0;
        while i < self.layer_vec.len() {
            ret_str.push_str(&self.layer_vec[i].to_string());
            if i < self.layer_vec.len() - 1 {
                ret_str.push_str(", : ");
            }
            i += 1;
        }
        ret_str.push_str("]");
        return ret_str;
    }

    fn randomly_step_net(&self, step_size: f32) -> Net {
        let mut new_layer_vec: Vec<Layer> = Vec::new();
        let mut i = 0;
        while i < self.layer_vec.len() {
            new_layer_vec.push(self.layer_vec[i].randomly_step_layer(step_size));
            i += 1;
        }
        return Net {
            layer_vec: new_layer_vec,
            structure: self.structure.clone()
        };
    }
}

fn blank_net(structure: Vec<usize>) -> Net {
    let mut layer_vec: Vec<Layer> = Vec::new();
    let mut i: usize = 1;
    let mut prev_size = structure[0];
    let mut size: usize;
    while i < structure.len() {
        size = structure[i];
        layer_vec.push(blank_layer(prev_size, size));
        prev_size = size;
        i += 1;
    }
    return Net {layer_vec: layer_vec, structure: structure};
}



fn main() {

}