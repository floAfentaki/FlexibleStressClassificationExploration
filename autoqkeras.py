import joblib
import numpy as np
from qkeras import QDense, QActivation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Activation  # Standard Keras ReLU
from sklearn.metrics import accuracy_score
from qkeras.quantizers import quantized_bits, quantized_relu
from convfxp import ConvFxp
import sys

# Read input arguments
dataset_name = sys.argv[1]
float_path = sys.argv[2]   # Path to the dataset files
number = sys.argv[3]       # Number of features to use
mfile = sys.argv[4]        # Path to the saved sklearn model (joblib file)

# 1. Load the sklearn trained MLP model from joblib
mlp_model = joblib.load(mfile)

# 2. Load the dataset
X_train = np.load(float_path + "/X_train_30.npy")
X_test = np.load(float_path + "/X_test_30.npy")
y_train = np.load(float_path + "/y_train.npy")
y_test = np.load(float_path + "/y_test.npy")

# Restrict to the specified number of features
X_train = X_train[:, :int(number.split('_')[0])]
X_test = X_test[:, :int(number.split('_')[0])]

# Convert data to fixed-point using ConvFxp
inpfxp = ConvFxp(0, 0, 4)
X_train_fxp = np.array([[inpfxp.to_fixed(w) for w in row] for row in X_train])
X_test_fxp = np.array([[inpfxp.to_fixed(w) for w in row] for row in X_test])

# 3. Extract weights and biases from the sklearn MLP model
weights = mlp_model.coefs_   # List of weight matrices
biases = mlp_model.intercepts_  # List of bias vectors

# Get the architecture info from sklearn model
layer_sizes = mlp_model.hidden_layer_sizes  # Tuple of hidden layer sizes
output_size = y_train.shape[1]  # Number of output classes

# Define the search space for quantization bits, integer bits, and activation types
quantization_bit_options = [4, 6, 8]  # Test 4-bit, 6-bit, and 8-bit quantization
integer_bit_options = [0, 1, 2, 3, 4]       # Test 0, 1, and 2 integer bits
relu_quantization_options = [  # Test different quantized_relu configurations
    (4, 0),  # 4 bits, 0 integer bits for ReLU
    (4, 1),  # 4 bits, 0 integer bits for ReLU
    (6, 1),  # 6 bits, 0 integer bits for ReLU
    (6, 2),  # 6 bits, 0 integer bits for ReLU
    (8, 1),  # 8 bits, 0 integer bits for ReLU
    (8, 2),  # 8 bits, 1 integer bit for ReLU
    (8, 3),  # 8 bits, 1 integer bit for ReLU
    (8, 4),  # 8 bits, 1 integer bit for ReLU
]
activation_types = ['quantized_relu', 'relu']  # Include quantized_relu, QReLU, and standard ReLU

best_accuracy = 0
best_params = None

stdoutput=sys.stdout
report_file = f"{float_path}/q_{number}.txt"
f=open(f"{report_file}", 'w')
stdoutput=f

# 4. Grid search loop through quantization bit options, integer bit options, relu options, and activation types
for quant_bits in quantization_bit_options:
    for int_bits in integer_bit_options:
        for activation_type in activation_types:
            # If activation is 'relu', skip relu quantization exploration
            if activation_type == 'relu':
                print(f"Testing quantization_bits={quant_bits}, integer_bits={int_bits}, activation_type={activation_type}")

                # Define the QKeras model architecture based on sklearn model
                quantized_mlp = Sequential()

                input_shape = (X_train.shape[1],)  # Input shape

                # Add the first quantized dense layer (input to first hidden layer)
                quantized_mlp.add(QDense(layer_sizes[0], input_shape=input_shape, 
                                         kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                         bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))

                # Apply standard ReLU activation
                quantized_mlp.add(Activation('relu'))

                # Add additional hidden layers
                for i in range(1, len(layer_sizes)):
                    quantized_mlp.add(QDense(layer_sizes[i], 
                                             kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                             bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))
                    quantized_mlp.add(Activation('relu'))  # Apply standard ReLU activation to hidden layers

                # Add the output layer (no activation because from_logits=True)
                quantized_mlp.add(QDense(output_size, 
                                         kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                         bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))

                # Transfer the weights from the sklearn MLP model to the QKeras model
                quantized_mlp.layers[0].set_weights([weights[0], biases[0]])
                for i in range(1, len(layer_sizes)):
                    quantized_mlp.layers[2 * i].set_weights([weights[i], biases[i]])

                # Compile the QKeras model
                quantized_mlp.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

                # Train the model
                quantized_mlp.fit(X_train_fxp, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=0)

                # Evaluate the model
                score = quantized_mlp.evaluate(X_test_fxp, y_test, verbose=0)
                accuracy = score[1]
                print(f"Accuracy: {accuracy*100:.2f}%")

                # Update best parameters if current combination is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (quant_bits, int_bits, None, None, activation_type)

            else:
                for relu_bits, relu_int in relu_quantization_options:
                    print(f"Testing quantization_bits={quant_bits}, integer_bits={int_bits}, relu_bits={relu_bits}, relu_int={relu_int}, activation_type={activation_type}")

                    # Define the QKeras model architecture based on sklearn model
                    quantized_mlp = Sequential()

                    input_shape = (X_train.shape[1],)  # Input shape

                    # Add the first quantized dense layer (input to first hidden layer)
                    quantized_mlp.add(QDense(layer_sizes[0], input_shape=input_shape, 
                                             kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                             bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))

                    # Choose activation type (either quantized_relu or QReLU)
                    if activation_type == 'quantized_relu':
                        quantized_mlp.add(QActivation(quantized_relu(bits=relu_bits, integer=relu_int, use_stochastic_rounding=True)))

                    # Add additional hidden layers
                    for i in range(1, len(layer_sizes)):
                        quantized_mlp.add(QDense(layer_sizes[i], 
                                                 kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                                 bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))

                        # Apply the corresponding activation type to hidden layers
                        if activation_type == 'quantized_relu':
                            quantized_mlp.add(QActivation(quantized_relu(bits=relu_bits, integer=relu_int, use_stochastic_rounding=True)))

                    # Add the output layer (no activation because from_logits=True)
                    quantized_mlp.add(QDense(output_size, 
                                             kernel_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True), 
                                             bias_quantizer=quantized_bits(bits=quant_bits, integer=int_bits, use_stochastic_rounding=True)))

                    # Transfer the weights from the sklearn MLP model to the QKeras model
                    quantized_mlp.layers[0].set_weights([weights[0], biases[0]])
                    for i in range(1, len(layer_sizes)):
                        quantized_mlp.layers[2 * i].set_weights([weights[i], biases[i]])

                    # Compile the QKeras model
                    quantized_mlp.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

                    # Train the model
                    quantized_mlp.fit(X_train_fxp, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=0)

                    # Evaluate the model
                    score = quantized_mlp.evaluate(X_test_fxp, y_test, verbose=0)
                    accuracy = score[1]
                    print(f"Accuracy: {accuracy*100:.2f}%")

                    # Update best parameters if current combination is better
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (quant_bits, int_bits, relu_bits, relu_int, activation_type)

# Output the best quantization parameters, relu configuration, and accuracy
print(f"Best quantization_bits={best_params[0]}, integer_bits={best_params[1]}, relu_bits={best_params[2]}, relu_int={best_params[3]}, activation_type={best_params[4]} with accuracy={best_accuracy*100:.2f}%")
sys.stdout=stdoutput
