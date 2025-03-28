import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from convfxp import ConvFxp
from mlp_fxp_ps import mlp_fxp_ps
import math
import torch
import itertools
# from feature_sel_cnn import create_cnn

import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_cnn(num_filters, kernel_size, input_shape=(4, 1), num_classes=2, optimizer='adam'):
    model = models.Sequential([
        layers.Conv1D(num_filters, kernel_size=kernel_size, input_shape=input_shape, activation='linear', use_bias=False),  # 4 kernels, kernel size 3
        layers.Flatten(),
        layers.Dense(num_classes, activation='linear', use_bias=False)  # Output for num_classes classes
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def eval_class(y_true, y_pred):
    print("New accuracy score:", accuracy_score(y_true, y_pred))

def get_maxabs(matrix_list):
    return max(max(abs(np.reshape(matrix, -1))) for matrix in matrix_list)

def get_width(value):
    return math.floor(value).bit_length()

# def convert_to_fixed_point(data, fxp, to_float=False):
#     converted = []
#     for layer_data in data:
#         layer_converted = []
#         for row in layer_data:
#             row_converted = [fxp.to_float(val) if to_float else fxp.to_fixed(val) for val in row]
#             layer_converted.append(row_converted)
#         converted.append(layer_converted)
#     return converted
# def convert_to_fixed_point(data, fxp, to_float=False):
#     if isinstance(data, np.ndarray):  # Check if it's a NumPy array
#         converted = []
#         for row in data:
#             if isinstance(row, np.ndarray):  # If it's an array (e.g., a weight matrix)
#                 row_converted = [fxp.to_float(val) if to_float else fxp.to_fixed(val) for val in row]
#             else:  # If it's a scalar (e.g., a bias term)
#                 row_converted = fxp.to_float(row) if to_float else fxp.to_fixed(row)
#             converted.append(row_converted)
#         return converted
#     else:  # If it's a scalar value
#         return fxp.to_float(data) if to_float else fxp.to_fixed(data)


def prune_cnn(model, sparsity_ratio):
    # Get the weights of the CNN model
    cnn_weights = model.model.get_weights()
    pruned_weights = []

    # Apply L2-norm pruning for each layer (similar to MLP)
    for layer_weights in cnn_weights:
        norms = np.linalg.norm(layer_weights, axis=0)
        threshold = np.percentile(norms, sparsity_ratio * 100)
        pruned_weights_layer = np.where(norms < threshold, 0, layer_weights)
        pruned_weights.append(pruned_weights_layer)
    model.model.set_weights(pruned_weights)
    return model



# def prune_l2_norm(model, sparsity_ratio):
#     for i in range(len(model.coefs_)):
#         weights = model.coefs_[i]
#         # calculaing L2 norms along cols
#         norms = np.linalg.norm(weights, axis=0)
#         threshold = np.percentile(norms, sparsity_ratio * 100)
#         weights[:, norms < threshold] = 0
#     return model


# def prune_hessian_based(model, sparsity_ratio):

#     if not hasattr(model, 'coefs_'):
#         raise ValueError("Model must have a 'coefs_' attribute (MLPClassifier or MLPRegressor).")
    
#     # Loop through each layer's weights
#     for i in range(len(model.coefs_)):
#         # Convert sklearn weights to PyTorch tensor
#         W_np = model.coefs_[i]  # Get weights for layer i (numpy array)
#         W = torch.tensor(W_np, requires_grad=True)  # Convert to PyTorch tensor
        
#         def loss_fn(W):
#             return torch.norm(W)**2
        
#         H = torch.autograd.functional.hessian(loss_fn, W)
#         H_abs = torch.abs(H)
#         W_metric = H_abs.sum(dim=(0, 1))  # Compute a metric for each weight based on the Hessian
#         num_elements = W_metric.numel()
#         prune_elements = int(num_elements * sparsity_ratio)
#         sorted_indices = torch.argsort(W_metric.flatten())
#         prune_indices = sorted_indices[:prune_elements]
        
#         # Prune the selected weights (set them to zero)
#         W_flat = W.flatten().clone()  # Clone to avoid modifying a tensor with requires_grad=True
#         W_flat[prune_indices] = 0  # Set the selected weights to zero
#         W_pruned = W_flat.reshape(W.shape)
#         model.coefs_[i] = W_pruned.detach().numpy() 
    
#     return model


def prune_hessian_based_pytorch(model, sparsity_ratio):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            def loss_fn(W):
                return torch.norm(W)**2
            H = torch.autograd.functional.hessian(loss_fn, W)
            H_abs = torch.abs(H)
            W_metric = H_abs.sum(dim=(0, 1))
            num_elements = W_metric.numel()
            prune_elements = int(num_elements * sparsity_ratio)
            sorted_indices = torch.argsort(W_metric.flatten())
            prune_indices = sorted_indices[:prune_elements]
            prune_indices = prune_indices[prune_indices < W.numel()]
            W_mask = torch.zeros_like(W.flatten(), dtype=torch.bool)
            W_mask[prune_indices] = True
            with torch.no_grad():
                W.view(-1)[W_mask] = 0
    return model


def prune_wanda_mod(model, input_matrix, sparsity_ratio=0.5):
    
    if not hasattr(model, 'coefs_'):
        raise ValueError("Model must have a 'coefs_' attribute (MLPClassifier or MLPRegressor).")

    # Convert input data to PyTorch tensor and ensure it's in float type (32-bit)
    inps = torch.tensor(input_matrix).float()
    

    # Store the PyTorch versions of the scikit-learn model's weights and biases
    layers = []
    for i in range(len(model.coefs_)):
        W_np = model.coefs_[i]  # Get weights for layer i (numpy array)
        W_torch = torch.tensor(W_np, requires_grad=False).float()  # Convert to PyTorch tensor and ensure float type
        layers.append(W_torch)

    activations = [inps]  # Initialize with input data as first activation
    
    # Compute activations for each layer (ReLU activation assumed)
    for i, W in enumerate(layers):
        activation = torch.relu(activations[-1] @ W)  # Forward pass with ReLU
        activations.append(activation)
    
    # Now apply WANDA pruning to each layer based on activations
    for layer_idx, W in enumerate(layers):
        activation = activations[layer_idx]
        # activation_norm = torch.norm(activation, p=2, dim=0)
        activation_norm = np.linalg.norm(activation, ord=2, axis=0)
        W_metric = torch.abs(W) * torch.sqrt(activation_norm).view(-1, 1).expand_as(W)
        threshold = torch.quantile(W_metric, sparsity_ratio)
        W_mask = W_metric <= threshold
        W_pruned = W.clone()  # Create a clone to modify
        W_pruned[W_mask] = 0
        model.coefs_[layer_idx] = W_pruned.detach().numpy()  # Convert back to numpy
    
    return model

def prune_wanda_keras(model, input_matrix, sparsity_ratio=0.5):
    inps = input_matrix 
    for i, W in enumerate(model.coefs_):
        activation = np.dot(inps, W) 
        activation_norm = np.linalg.norm(activation, ord=2, axis=0)
        W_metric = np.abs(W) * np.sqrt(activation_norm).reshape(-1, 1)
        threshold = np.quantile(W_metric, sparsity_ratio)
        W_mask = W_metric <= threshold
        W[W_mask] = 0
        model.coefs_[i] = W
    return model

def prune_wanda_pytorch(model, input_matrix, sparsity_ratio=0.5):
    inps = torch.tensor(input_matrix).float()
    layers = [module for module in model if isinstance(module, nn.Linear)]

    activations = []
    hooks = []

    def hook_fn(module, input, output):
        activations.append(input[0].detach())

    for layer in layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = model(inps)
    for hook in hooks:
        hook.remove()
    for layer_idx, layer in enumerate(layers):
        activation = activations[layer_idx]
        activation_norm = torch.norm(activation, p=2, dim=0)
        W_metric = torch.abs(layer.weight.data) * torch.sqrt(activation_norm).view(1, -1).expand_as(layer.weight.data)
        threshold = torch.quantile(W_metric, sparsity_ratio)
        W_mask = W_metric <= threshold
        layer.weight.data[W_mask] = 0
        print(f"Layer {layer_idx} pruning completed with sparsity ratio {sparsity_ratio}.")

    return model


# def retrain_model(model, X_train, y_train):
#     model.fit(X_train, y_train)
#     return model

def retrain_model(model, X_train, y_train, learning_rate=0.0001, epochs=10):
    # Compile the model with a lower learning rate for fine-tuning
    model.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    return model


def convertCoef (mdata, fxp, toFP):
 transp=[np.transpose(i).tolist() for i in mdata]
 coefficients=[]
 for i, l in enumerate(transp):
    newn=list()
    for j,n in enumerate(l):
        neww=list()
        for k,w in enumerate(n):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        newn.append(neww)
    coefficients.append(newn)
 return coefficients

def convertIntercepts (mdata, lfxp, toFP):
 transp=[np.transpose(i).tolist() for i in mdata]
 intercepts=[]
 for i, l in enumerate(transp):
        fxp=lfxp[i]
        neww=list()
        for k,w in enumerate(l):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        intercepts.append(neww)
 return intercepts

def quantize_cnn(model, X_train, X_test, y_train, y_test):
    cnn_weights = model.model.get_weights()
    # fxp_weights = [convert_to_fixed_point(w, ConvFxp(0, 0, 4)) for w in cnn_weights]
    y_pred = model.predict(X_test)

    # Check if the predictions are probabilities (2D) or labels (1D)
    if len(y_pred.shape) > 1:  # If predictions are 2D, get the argmax (class prediction)
        y_pred = np.argmax(y_pred, axis=1)

    # Similarly, check if y_test needs to be transformed
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
    # return fxp_weights, accuracy

def quantize_model_to_fixed_point(model, inpfxp, wfxp, bfxp):
    fxp_weights = convertCoef(model.coefs_, wfxp, False)
    fxp_biases = convertIntercepts(model.intercepts_, bfxp, False)
    return fxp_weights, fxp_biases

def save_model_layers(coefs, intercepts, folder_path, num_f, step):
    for i, coef in enumerate(coefs):
        np.save(os.path.join(folder_path, f"weights_layer_{i}.npy"), coef)
    for i, intercept in enumerate(intercepts):
        np.save(os.path.join(folder_path, f"biases_layer_{i}.npy"), intercept)

def create_and_save_model(model, stage_folder, num_f, step):
    if not os.path.exists(stage_folder):
        os.makedirs(stage_folder)

    joblib.dump(model, os.path.join(stage_folder, f"{num_f}_{step}.joblib"))
    # save_model_layers(model.coefs_, model.intercepts_, stage_folder, num_f, step)

def prune_quantize_fine_tune_model(dataset, model, fs_method, pruning_method, sparsity_ratios):

    folder_path = f'./{model}/{dataset}/{fs_method}'
    float_path = f'{folder_path}/float_models'
    pruned_path = f'{folder_path}/pruning_{pruning_method}'
    ft_pruned_path = f'{folder_path}/pruning_ft_{pruning_method}'
    feature_sizes = [5, 10, 15, 20, 25, 30]

    if not os.path.exists(pruned_path):
        os.makedirs(pruned_path)

    X_train = np.load(os.path.join(float_path, f"X_train_{feature_sizes[-1]}.npy"))
    X_test = np.load(os.path.join(float_path, f"X_test_{feature_sizes[-1]}.npy"))
    y_train = np.load(os.path.join(float_path, "y_train.npy"))
    y_test = np.load(os.path.join(float_path, "y_test.npy"))

    for num_f in feature_sizes:
        model_path = f"{float_path}/{num_f}.joblib"
        model = joblib.load(model_path)

        for sparsity_ratio in sparsity_ratios:
            # Pruning step
            if isinstance(model, MLPClassifier):
                # Prune MLP model using coefs_ and intercepts_
                if pruning_method == 'l2':
                    pruned_model = prune_l2_norm(model, sparsity_ratio)
                elif pruning_method == 'wanda':
                    pruned_model = prune_wanda_mod(model, X_train[:, :num_f], sparsity_ratio)
                elif pruning_method == 'hessian':
                    pruned_model = prune_hessian_based(model, sparsity_ratio)
                
            elif isinstance(model, KerasClassifier):
                # Prune CNN model using get_weights and set_weights
                pruned_model = prune_cnn(model, sparsity_ratio)
                print(f"pruned using keras Classifier {pruned_model}")
            
            str_spar = str(sparsity_ratio).replace('.', '_')
            create_and_save_model(pruned_model, pruned_path, num_f, f'sparsity_{str_spar}')

            # Quantization step
            if isinstance(model, MLPClassifier):
                # Quantize MLP model
                inpfxp = ConvFxp(0, 0, 4)
                wfxp = ConvFxp(1, get_width(get_maxabs(pruned_model.coefs_)), 8 - 1 - get_width(get_maxabs(pruned_model.coefs_)))
                bfxp = [ConvFxp(1, get_width(get_maxabs(b)), inpfxp.frac + (i + 1) * wfxp.frac - 0) for i, b in enumerate(pruned_model.intercepts_)]
                fxp_weights, fxp_biases = quantize_model_to_fixed_point(pruned_model, inpfxp, wfxp, bfxp)

                # Evaluation
                fxp_y_train = [np.argmax(y, axis=None, out=None) for y in y_train[:]]
                fxp_y_test = [np.argmax(y, axis=None, out=None) for y in y_test[:]]
                fxp_model = mlp_fxp_ps(fxp_weights, fxp_biases, inpfxp, wfxp, 0, fxp_y_train, pruned_model)
                fxp_accuracy = fxp_model.get_accuracy(X_test[:, :num_f], fxp_y_test)

            elif isinstance(model, KerasClassifier):
                # Quantize CNN model using get_weights and set_weights
                cnn_weights = pruned_model.model.get_weights()
                wfxp = ConvFxp(1, get_width(get_maxabs(cnn_weights)), 8 - 1 - get_width(get_maxabs(cnn_weights)))  # Quantize weights
                fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)
                # fxp_weights, fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)

            print(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Accuracy: {fxp_accuracy}")
            with open(os.path.join(pruned_path, "qt_model_accuracies.txt"), "a") as log_file:
                log_file.write(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Accuracy: {fxp_accuracy}\n")

            # Finetuning step
            pruned_model = retrain_model(pruned_model, X_train[:, :num_f], y_train)
            create_and_save_model(pruned_model, ft_pruned_path, num_f, f'sparsity_{str_spar}')

            # Quantization
            if isinstance(pruned_model, KerasClassifier):
                # fxp_weights, fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)
                fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)

            print(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Post-finetuning Accuracy: {fxp_accuracy}")
            with open(os.path.join(ft_pruned_path, "qt_model_accuracies.txt"), "a") as log_file:
                log_file.write(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}\n")
                log_file.write(f"Quantized Accuracy in Fixed Point: {fxp_accuracy}\n\n")

    # Save data to pruned path
    np.save(os.path.join(pruned_path, f"X_train_{feature_sizes[-1]}.npy"), X_train)
    np.save(os.path.join(pruned_path, f"X_test_{feature_sizes[-1]}.npy"), X_test)
    np.save(os.path.join(pruned_path, "y_train.npy"), y_train)
    np.save(os.path.join(pruned_path, "y_test.npy"), y_test)

    np.save(os.path.join(ft_pruned_path, f"X_train_{feature_sizes[-1]}.npy"), X_train)
    np.save(os.path.join(ft_pruned_path, f"X_test_{feature_sizes[-1]}.npy"), X_test)
    np.save(os.path.join(ft_pruned_path, "y_train.npy"), y_train)
    np.save(os.path.join(ft_pruned_path, "y_test.npy"), y_test)



# dataset=['wesad','spd']
dataset=['wesad']
# model=['MLP']
model=['CNN']
# fs_method = ['jmi', 'disr','fisher_score']
fs_method = ['jmi']
# pruning_method = ['l2', 'wanda', 'hessian']
pruning_method = ['wanda']
sparsity_ratios = [0.0, 0.2, 0.5, 0.9]

combinations = list(itertools.product(dataset, model, fs_method, pruning_method))

for combo in combinations:
    prune_quantize_fine_tune_model(combo[0], combo[1], combo[2], combo[3], sparsity_ratios)