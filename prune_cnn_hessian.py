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

import tensorflow as tf
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

def convert_to_fixed_point(data, fxp, to_float=False):
    if isinstance(data, np.ndarray):  # Check if it's a NumPy array
        converted = []
        for row in data:
            if isinstance(row, np.ndarray):  # If it's an array (e.g., a weight matrix)
                row_converted = [fxp.to_float(val) if to_float else fxp.to_fixed(val) for val in row]
            else:  # If it's a scalar (e.g., a bias term)
                row_converted = fxp.to_float(row) if to_float else fxp.to_fixed(row)
            converted.append(row_converted)
        return converted
    else:  # If it's a scalar value
        return fxp.to_float(data) if to_float else fxp.to_fixed(data)



def retrain_model(model, X_train, y_train, learning_rate=0.0001, epochs=10):
    # Compile the model with a lower learning rate for fine-tuning
    # model.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
    #                     loss='categorical_crossentropy',
    #                     metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    cnn_weights = model.get_weights()
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

# Gradient extraction function
def get_gradients(model, x, y):
    # Use GradientTape to compute gradients of loss w.r.t. model's trainable weights
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

# Hessian extraction function
def get_hessian(keras_clf, x, y):
    model = keras_clf.model
    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape() as tape2:
            logits = model(x, training=True)  # Use the underlying Keras model
            loss = tf.keras.losses.categorical_crossentropy(y, logits)
        gradients = tape2.gradient(loss, model.trainable_variables)
    hessians = []
    for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
        if 'bias' not in var.name:
            hessian = tape1.gradient(grad, var)
            hessians.append(hessian)
    return hessians

def hessian_based_pruning(model, hessians, pruning_percentage=0.2):
    hessian_idx = 0  
    # Loop over each layer's weights in the model
    for i, layer in enumerate(model.layers):
        # Check if the layer has trainable weights (i.e., not a layer like Flatten)
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]  # Get the weights of the layer (exclude biases)
            print(f"Layer {i} weights shape: {weights.shape}")

            biases = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
            if biases is not None:
                print(f"Layer {i} biases shape: {biases.shape}")
            weight_hessian = hessians[hessian_idx]  
            print(f"Layer {i} weight_hessian shape: {weight_hessian.shape}")
            hessian_idx += 1
            flattened_weights = tf.reshape(weights, [-1])
            flattened_hessians = tf.reshape(weight_hessian, [-1])
            print(f"flattened_hessians : {flattened_hessians}")
            num_weights_to_prune = int(pruning_percentage * int(tf.size(flattened_weights)))
            print("num_weights",num_weights_to_prune)
            prune_indices = tf.argsort(tf.abs(flattened_hessians))[:num_weights_to_prune]
            print(f"prune_indices{prune_indices}")
            mask = tf.ones_like(flattened_weights)
            mask = tf.tensor_scatter_nd_update(
                mask, tf.expand_dims(prune_indices, 1), 
                tf.zeros([num_weights_to_prune], dtype=mask.dtype)
            )
            print(f"mask{mask}")
            pruned_flattened_weights = flattened_weights * mask

            # Reshape pruned weights back to original shape
            pruned_weights = tf.reshape(pruned_flattened_weights, weights.shape)

            # Set pruned weights back into the layer (keeping biases unchanged)
            if biases is not None:
                model.layers[i].set_weights([pruned_weights.numpy(), biases])
            else:
                model.layers[i].set_weights([pruned_weights.numpy()])

    return model



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

        # Ensure the model is fitted before pruning
        if isinstance(model, KerasClassifier) and not hasattr(model, 'model_'):
            print(f"Fitting model for feature size {num_f}")
            model.fit(X_train[:, :num_f], y_train)  # Fit the model if not fitted

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
                if pruning_method == 'l2':
                    pruned_model = prune_cnn(model, sparsity_ratio)

                elif pruning_method == 'hessian':
                        # Extract Hessians
                        hessians = get_hessian(model, X_train[:, :num_f], y_train)
                        # Prune the model using Hessian-based pruning
                        pruned_model = hessian_based_pruning(model.model, hessians, sparsity_ratio)
                elif pruning_method == 'wanda':
                    pruned_model = prune_wanda_keras(model, X_train[:, :num_f], sparsity_ratio)

            str_spar = str(sparsity_ratio).replace('.', '_')
            create_and_save_model(pruned_model, pruned_path, num_f, f'sparsity_{str_spar}')

            # Quantization step
            if isinstance(model, MLPClassifier):
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
                # cnn_weights = pruned_model.model.get_weights()
                # wfxp = ConvFxp(1, get_width(get_maxabs(cnn_weights)), 8 - 1 - get_width(get_maxabs(cnn_weights)))  # Quantize weights
                # fxp_weights, fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)
                cnn_weights = pruned_model.get_weights()  # Directly access get_weights()
                wfxp = ConvFxp(1, get_width(get_maxabs(cnn_weights)), 8 - 1 - get_width(get_maxabs(cnn_weights)))  # Quantize weights
                fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)

            print(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Accuracy: {fxp_accuracy}")
            with open(os.path.join(pruned_path, "qt_model_accuracies.txt"), "a") as log_file:
                log_file.write(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Accuracy: {fxp_accuracy}\n")

            # Finetuning step
            pruned_model = retrain_model(pruned_model, X_train[:, :num_f], y_train)
            create_and_save_model(pruned_model, ft_pruned_path, num_f, f'sparsity_{str_spar}')

            # Quantization
            if isinstance(pruned_model, KerasClassifier):
                fxp_accuracy = quantize_cnn(pruned_model, X_train[:, :num_f], X_test[:, :num_f], y_train, y_test)

            print(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}, Post-finetuning Accuracy: {fxp_accuracy}")
            with open(os.path.join(ft_pruned_path, "qt_model_accuracies.txt"), "a") as log_file:
                log_file.write(f"Feature Size: {num_f}, Sparsity Ratio: {sparsity_ratio}\n")
                log_file.write(f"Quantized Accuracy in Fixed Point: {fxp_accuracy}\n\n")

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
pruning_method = ['hessian']
sparsity_ratios = [0.0, 0.2, 0.5, 0.9]

combinations = list(itertools.product(dataset, model, fs_method, pruning_method))

for combo in combinations:
    prune_quantize_fine_tune_model(combo[0], combo[1], combo[2], combo[3], sparsity_ratios)
