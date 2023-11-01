import numpy as np

# Load data from data.txt
def read_file(filename):
    try:
        box = np.loadtxt(filename, dtype=int)
        return box
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
        
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1 - x)

# Split the data into 10% cross-validation segments (90% train and 10% test)
def Cross_Validation(data, num_segments, test_ratio=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = len(data)
    test_size = int(num_samples * test_ratio)
    init = np.random.permutation(num_samples)
    test_indices = init[:test_size]
    train_indices = init[test_size:]

    test_data = data[test_indices]
    train_data = data[train_indices]

    return test_data, train_data

# Calculate the min and max values
def getmaxmin(data):
    max = np.max(data)
    min = np.min(data)
    return max, min

# Normalize 
def normalize(data):
    min = np.min(data)
    max = np.max(data)
    normalize = (data - min) / (max - min)
    return normalize

# Normalize input
def normalize_data_set(data):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    normalize = (data - min) / (max - min)

    Input = normalize[:, :8]
    output_data = normalize[:, 8]
    return Input, output_data


# Inverse normalization from the 0-1 range to the original data range
def inv_normalize(normalize, original_data):
    min = np.min(original_data)
    max = np.max(original_data)
    inv_normalize_data = (normalize * (max - min)) + min
    return inv_normalize_data

# Forward propagation
def forward_propagation(Input, hidden1, hidden2, hidden_Out, back_pOutput):
    hidden = sigmoid(np.dot(hidden1, Input) + hidden2)
    output = sigmoid(np.dot(hidden_Out, hidden) + back_pOutput)
    return hidden, output

# Update weights and biases 1
def update_input_hidden_layer_weights(w_input_to_hidden, b_hidden, v_w_input_hidden, v_b_hidden, hidden_gradient, Input, learning_rate, momentum_rate):
    v_w_input_hidden = (momentum_rate * v_w_input_hidden) + (learning_rate * np.dot(hidden_gradient, Input.T) / Input.shape[1])
    w_input_to_hidden += v_w_input_hidden
    v_b_hidden = (momentum_rate * v_b_hidden) + (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True))
    b_hidden += v_b_hidden


# Update weights and biases 2
def update_hidden_output_layer_weights(w_hidden_to_output, b_output, v_w_hidden_output, v_b_output, output_gradient, hidden, learning_rate, momentum_rate):
    v_w_hidden_output = (momentum_rate * v_w_hidden_output) + (learning_rate * np.dot(output_gradient, hidden) / hidden.shape[1])
    w_hidden_to_output += v_w_hidden_output
    v_b_output = (momentum_rate * v_b_output) + (learning_rate * np.mean(output_gradient, axis=1, keepdims=True))
    b_output += v_b_output

