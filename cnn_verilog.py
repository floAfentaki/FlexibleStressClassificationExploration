import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import math


# Custom convolution function using PyTorch and TensorFlow
def conv2(input_list, weights):

    weights=torch.Tensor(weights).movedim(2,0)
    weights=tf.squeeze(weights)
    output = []

    for kernel in weights:
        kernel_result = []
        for i in range(len(input_list) - len(kernel) + 1):
            sub_input = input_list[i:i + len(kernel)]
            sum_product = np.sum(sub_input * kernel)
            kernel_result.append(sum_product)
        output.append(kernel_result)

    output = np.array(output).T  
    return output
    
# Define the custom inference function
def custom_inference(input_data, conv_weights, dense_weights):
    input_data = np.array(input_data)
    
    conv_result=conv2(input_data, conv_weights)
    conv_result_flat = np.array(conv_result).flatten()

    # print(f"Custom Conv1D Output (Flattened): {conv_result_flat}")

    output = np.dot(conv_result_flat, dense_weights)
    return output

# Calculate the output bitwidth for a layer
def calculate_output_bitiwdth(inp_bitwidth, weights, num_sums):
        
    max_weight_bitwdth = max(int(x) for x in weights.flatten())
    max_product_value = (2**inp_bitwidth - 1) * max_weight_bitwdth
    max_sum_value = num_sums * max_product_value
    bitwdth = math.ceil(math.log2(max_sum_value))

    return bitwdth

def generate_verilog(conv_weights, dense_weights, ACT_FUNC):

    kernel_size, _, num_kernels =np.shape(conv_weights)
    num_flat_kernels, num_neurons = np.shape(dense_weights)

    print(kernel_size, num_kernels)
    print(num_flat_kernels, num_neurons)

    INP_BITWIDTH=4
    NUM_INPUT=4
    CONV_BITWIDTH=calculate_output_bitiwdth(INP_BITWIDTH, conv_weights, kernel_size)
    DENSE_OUT_BITWIDTH=calculate_output_bitiwdth(CONV_BITWIDTH, dense_weights, num_flat_kernels)

    module_v=f"""
module top(
    input [{NUM_INPUT*INP_BITWIDTH-1}:0] in,
    output [{math.ceil(math.log2(num_neurons))-1}:0] out  
    );"""

    
    conv_v=f"\n\n\t// Convolution Layer"
    conv_v+=f"\n\twire signed [{CONV_BITWIDTH-1}:0] conv_out[{num_flat_kernels-1}:0];"
    cnt=0
    for i in range(num_kernels):
        # conv_v+=f"\n\tassign conv_out[{i}] ="
        for l in range(num_kernels):
            conv_v+=f"\n\tassign conv_out[{cnt}] ="
            cnt+=1

            for j in range(kernel_size):
                conv_v+=f" {conv_weights[j, 0, l]}*in[{(i + j) * INP_BITWIDTH + (INP_BITWIDTH-1)}:{(i + j) * INP_BITWIDTH}] +"
            conv_v=conv_v[0:-2] # Remove the last '+' sign
            conv_v+=";"
        conv_v+="\n"
        if cnt==num_flat_kernels:
            break

    act_v=f"\n\t// Activation Function: {ACT_FUNC}"
    if ACT_FUNC=='linear':
        act_v+=f"\n\twire signed [{CONV_BITWIDTH-1}:0] act_out[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels):
            act_v+=f"\n\tassign act_out[{i}] = conv_out[{i}];"
    elif ACT_FUNC=='relu':
        act_v+=f"\n\twire [{CONV_BITWIDTH-1-1}:0] act_out[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels):
            act_v+=f"\n\tassign act_out[{i}] = (conv_out[{i}][{CONV_BITWIDTH-1}]==1)? 0 : conv_out[{i}][{CONV_BITWIDTH-1-1}:0];"
    act_v+="\n"

    fc_v="\n\t// Fully Connected "
    for i in range(num_neurons):
        fc_v+=f"\n\twire signed [{DENSE_OUT_BITWIDTH-1}:0] dense_out_{i};"
        fc_v+=f"\n\tassign dense_out_{i} ="
        for j in range(num_flat_kernels):
            fc_v+=f" {dense_weights[j,i]}*act_out[{j}] +"
        fc_v=fc_v[0:-2] # Remove the last '+' sign
        fc_v+=";\n" 

    # Argmax Logic
    argmax_v = "\n\n\t// Argmax Logic\n"
    for i in range(num_neurons - 1):
        argmax_v += f"\twire argmax_{i};\n"
        argmax_v += f"\tassign argmax_{i} = (dense_out_{i} > dense_out_{i+1}) ? {i} : {i+1};\n"

    argmax_v += "\tassign out = argmax_0"
    for i in range(1, num_neurons - 1):
        argmax_v += f" > dense_out_{i+1} ? argmax_{i} : {i+1}"

    argmax_v += ";"
    
    verilog_code = module_v+conv_v+act_v+fc_v+argmax_v+"\n\nendmodule"
    return verilog_code



def main(conv_weights, dense_weights):

    kernel_size, _, num_kernels =np.shape(conv_weights)
    _, num_neurons = np.shape(dense_weights)

        
    # Define the model
    model = models.Sequential([
        layers.Conv1D(num_kernels, kernel_size=kernel_size, input_shape=(4, 1), activation='linear', use_bias=False),  # 3 kernels, kernel size 2, 1 input channel
        layers.Flatten(),
        layers.Dense(num_neurons, activation='linear', use_bias=False)  # Output for 2 classes
    ])

    # Set the weights in the model
    model.layers[0].set_weights([conv_weights])
    model.layers[2].set_weights([dense_weights])

    ACT_FUNC=model.layers[0].activation.__name__
    # print()

    # Example input data
    # input_data_keras = np.array([[[1], [2], [3], [4]]])  # Shape: (1, 4, 1)
    input_data_keras = np.array([
        [[1], [2], [3], [4]],  # Sample 1
        [[4], [4], [4], [4]],  # Sample 2
        [[4], [3], [2], [1]],  # Sample 3
        [[2], [2], [2], [2]],  # Sample 4
        [[5], [4], [3], [5]]   # Sample 5
    ])
    # Create a model that outputs the result of the Conv1D layer
    conv_model = models.Model(inputs=model.input, outputs=model.layers[0].output)

    # Get the output of the Conv1D layer
    conv_output_keras = conv_model.predict(input_data_keras)
    # print(f"Keras Conv1D Output: {conv_output_keras.flatten()}")

    # Perform inference using the complete model
    keras_output = model.predict(input_data_keras)
    # print(f"Keras Model Output: {keras_output.flatten()}")
    print(f"Keras Model Output: {keras_output}")

    # Example input data for custom inference
    input_data_custom = np.array([1, 2, 3, 4])

    # Perform custom inference
    custom_output = custom_inference(input_data_custom, conv_weights, dense_weights)
    # print(f"Custom Inference Output: {custom_output}")

    # Generate the Verilog code
    verilog_code = generate_verilog(conv_weights, dense_weights, ACT_FUNC)

    # Print or save the Verilog code
    # print(verilog_code)
    with open("simple_cnn_3kernels_2classes.v", "w") as f:
        f.write(verilog_code)

if __name__=="__main__":

    # Define correct weights for testing
    # conv_weights = np.array([[[1, 2, 3]], [[4, 5, 6]]])  # Shape: (2, 1, 3) - 3 kernels, kernel size 2, 1 input channel
    # dense_weights = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])  # Shape: (9, 2) - 9 inputs to 2 outputs
    # conv_weights = np.array([[[1, 2, 3, 4]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]])
    # dense_weights = np.array([
        # [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], 
        # [13, 14], [15, 16]])

    conv_weights = np.array([
        [[1, -2, 3, -4]],  # coeff 1
        [[-5, 6, -7, 8]],  # coeff 2
        [[9, -10, 11, -12]],  # coeff 3
    ])  # Shape: (3, 1, 4) - 4 kernels, kernel size 3, 1 input channel
    dense_weights = np.array([
        [-1, 2], [-3, 4], [-5, 6], [-7, 8], 
        [9, -10], [11, -12], [13, -14], [15, -16]
    ])

    main(conv_weights, dense_weights)