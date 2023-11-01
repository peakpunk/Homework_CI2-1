import numpy as np

def read_cross_file(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for i in range(0, len(lines), 3):
            label = lines[i].strip()
            x, y = map(float, lines[i + 1].split())
            label_data = (label, (x, y),)
            label_data += tuple(map(int, lines[i + 2].split()))
            data.append(label_data)
    
    return data

def split_data_into_segments(data, num_segments):
    np.random.shuffle(data)  # สับเปลี่ยนข้อมูลที่ส่งผ่านมา
    segment_size = len(data) // num_segments

    test_data_segments = []
    train_data_segments = []

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size

        # สร้างชุดข้อมูล test และ train
        test = data[start:end]
        train = np.concatenate((data[:start], data[end:]), axis=0)

        test_data_segments.append(test)
        train_data_segments.append(train)

    return test_data_segments, train_data_segments


def transform_data(input_data):
    transformed_data = [(row[0], (row[1][0], row[1][1]), row[2], row[3]) for row in input_data]
    return transformed_data


def prepare_data(data):
    inputdata = np.array([item[1] for item in data])
    outputdata = np.array([item[2:] for item in data])
    return inputdata, outputdata

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(input_data, w_input_to_hidden, b_hidden, w_hidden_to_output, b_output):
    # คำนวณ hidden layer
    hidden_input = np.dot(w_input_to_hidden, input_data) + b_hidden
    hidden_output = sigmoid(hidden_input)

    # คำนวณ output layer
    output_input = np.dot(w_hidden_to_output, hidden_output) + b_output
    output = sigmoid(output_input)

    return hidden_output, output

def update_input_hidden_layer_weights(input_data, hidden_gradient, learning_rate, momentum_rate, w_input_to_hidden, b_hidden, v_w_input_hidden, v_b_hidden):
    batch_size = len(input_data)

    # คำนวณการเปลี่ยนแปลงของ weights
    delta_w_input_to_hidden = (learning_rate * np.dot(hidden_gradient, input_data.T)) / batch_size
    delta_b_hidden = (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True)) / batch_size

    # อัพเดท weights ด้วย momentum
    v_w_input_hidden = (momentum_rate * v_w_input_hidden) - delta_w_input_to_hidden
    v_b_hidden = (momentum_rate * v_b_hidden) - delta_b_hidden

    # อัพเดท weights และ bias ใน hidden layer
    w_input_to_hidden += v_w_input_hidden
    b_hidden += v_b_hidden
    
def update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate, w_hidden_to_output, b_output, v_w_hidden_output, v_b_output):
    batch_size = len(hidden)

    # คำนวณการเปลี่ยนแปลงของ weights
    delta_w_hidden_to_output = (learning_rate * np.dot(output_gradient, hidden.T)) / batch_size
    delta_b_output = (learning_rate * np.mean(output_gradient, axis=1, keepdims=True)) / batch_size

    # อัพเดท weights ด้วย momentum
    v_w_hidden_output = (momentum_rate * v_w_hidden_output) - delta_w_hidden_to_output
    v_b_output = (momentum_rate * v_b_output) - delta_b_output

    # อัพเดท weights และ bias ใน output layer
    w_hidden_to_output += v_w_hidden_output
    b_output += v_b_output
    
def train_custom_neural_network(inputdata, outputdata, Target_Epochs, Mean_Squared_Error, learning_rate, momentum_rate):
    for epoch in range(Target_Epochs):
        hidden, output = forward_propagation(inputdata)

        output_error = outputdata - output.T
        output_gradient = output_error.T * sigmoid_derivative(output)
        update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate)

        hidden_error = np.dot(w_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden)
        update_input_hidden_layer_weights(inputdata, hidden_gradient, learning_rate, momentum_rate)

        error = np.mean(output_error**2, axis=0)

        if np.all(error <= Mean_Squared_Error):
            print(f"Training stopped at epoch {epoch+1} because Mean Squared Error is below the threshold.")
            break

def calculate_accuracy(confusion_matrix):
    TP, TN, FP, FN = confusion_matrix
    total_predictions = TP + TN + FP + FN
    if total_predictions == 0:
        return 0.0
    accuracy_percentage = ((TP + TN) / total_predictions) * 100
    return accuracy_percentage
     
filename = 'cross.txt'
cross_data = read_cross_file(filename)

input_size = 2
hidden_size = 8 # สามารถกำหนดเองได้
output_size = 2
        
# ปรับ learning_rates และ momentum_rates ตามที่ต้องการ    
learning_rates = [0.2,0.5]
momentum_rates = [0.2,0.2]

K_segments = 10

print(f"Hidden node = {hidden_size}")   
for i in range(K_segments):
    # Initialize weights and biases with random values
    w_input_to_hidden = np.random.randn(hidden_size, input_size)
    v_w_input_hidden = np.random.randn(hidden_size, input_size)
    w_hidden_to_output = np.random.randn(output_size, hidden_size)
    v_w_hidden_output = np.random.randn(output_size, hidden_size)
    b_hidden = np.random.randn(hidden_size, 1)
    v_b_hidden = np.random.randn(hidden_size, 1)
    b_output = np.random.randn(output_size, 1)
    v_b_output = np.random.randn(output_size, 1)
    
    for lr in learning_rates:
        for momentum in momentum_rates: 
            print(f"Segment = {i+1}, Training with learning rate = {lr} and momentum = {momentum}")
            
            test, train = split_data_into_segments(cross_data, K_segments)
            
            transformed_train_data = transform_data(train[i])
            transformed_test_data = transform_data(test[i])
            
            input_train, output_train = prepare_data(transformed_train_data)
            
            train_custom_neural_network(input_train, output_train, 100, 0.00001, lr, momentum)
                
            input_test, output_test = prepare_data(transformed_test_data)
            Actual = output_test
            x, Predict = forward_propagation(input_test)
            Predict = np.transpose(Predict)
            threshold = 0.5
            predicted = (Predict[:, 1] > threshold).astype(int)
            
            confusion_matrix = np.zeros((2, 2), dtype=int)
            for i in range(2):
                for j in range(2):
                    confusion_matrix[i, j] = np.sum((Actual[:, i] == 1) & (predicted == j))
            
            TP = confusion_matrix[1, 1]
            TN = confusion_matrix[0, 0]
            FP = confusion_matrix[0, 1]
            FN = confusion_matrix[1, 0]
            Accuracy = calculate_accuracy((TP, TN, FP, FN))
            
            print("Confusion Matrix:")
            print(confusion_matrix)
            print("True Positive (TP):", TP)
            print("True Negative (TN):", TN)
            print("False Positive (FP):", FP)
            print("False Negative (FN):", FN)
            print(f"Accuracy = {Accuracy}")

    
  