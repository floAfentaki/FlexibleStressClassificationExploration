import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Conv1DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * input_size, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def uniform_quantize_to_int(tensor, bits):
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (bits - 1) - 1)
    quantized = torch.round(tensor / scale).clamp(-2**(bits-1), 2**(bits-1) - 1)
    return quantized.to(dtype=torch.int16 if bits == 16 else torch.int8), scale

def apply_activation_aware_quantization(model, calibration_data, bits=8):
    device = next(model.parameters()).device
    calibration_data = calibration_data.to(device)
    scales = {}

    def compute_scale(module, input, output):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            with torch.no_grad():
                W = module.weight.data
                X = input[0]
                alpha = torch.quantile(torch.abs(X), 0.99, dim=0)
                scale = (alpha.unsqueeze(1) @ torch.abs(W).max(dim=0)[0].unsqueeze(0)) ** 0.5
                scale = scale / (2 ** (bits - 1) - 1)
                scales[module] = scale.view(-1)

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            handle = module.register_forward_hook(compute_scale)
            handles.append(handle)

    model(calibration_data)

    for handle in handles:
        handle.remove()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            scale = scales[module]
            if isinstance(module, nn.Linear):
                module.weight.data = torch.round(module.weight.data / scale.unsqueeze(1)).clamp(-2**(bits-1), 2**(bits-1) - 1)
                module.weight.data = module.weight.data * scale.unsqueeze(1)
            elif isinstance(module, nn.Conv1d):
                module.weight.data = torch.round(module.weight.data / scale.unsqueeze(1).unsqueeze(2)).clamp(-2**(bits-1), 2**(bits-1) - 1)
                module.weight.data = module.weight.data * scale.unsqueeze(1).unsqueeze(2)
            
            if module.bias is not None:
                module.bias.data = torch.round(module.bias.data / scale).clamp(-2**(bits-1), 2**(bits-1) - 1)
                module.bias.data = module.bias.data * scale

    return model, scales

def run_experiments():
    feature_selection_results_path = '/home/snakkill/pm/pe_new/svm/FS/feature_selection_results.csv'
    data_path = '/home/snakkill/pm/pe_new/data/90sec_features_n_label_2_SMOTE_True.csv'

    feature_selection_results = pd.read_csv(feature_selection_results_path)

    data = pd.read_csv(data_path)
    y = data['label'].values

    tuned_parameters = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50), (100, 100)],
        'activation': ['relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
        'learning_rate': ['constant', 'adaptive'],
    }

    methods = feature_selection_results['Method'].unique()
    feature_sizes = [5, 10, 15, 20, 25, 30]
    quantization_bits = [4, 8, 16]
    architectures = ['MLP', 'CNN']

    results_csv_path = '/home/snakkill/pm/pe_new/mlp/quantization/quantization_results.csv'
    
    if os.path.exists(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
    else:
        existing_results = pd.DataFrame(columns=['Method', 'Size', 'Architecture', 'Quantization_Bits', 'Selected_Features', 'Best_Params', 'Accuracy'])

    for method in methods:
        for size in feature_sizes:
            selected_features_info = feature_selection_results[
                (feature_selection_results['Method'] == method) &
                (feature_selection_results['Size'] == size)
            ]

            if selected_features_info.empty:
                continue

            selected_features = selected_features_info.iloc[0]['Selected_Features'].strip('[]').replace("'", "").split(', ')
            X_selected = data[selected_features].values
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

            for architecture in architectures:
                for bits in quantization_bits:
                    if not existing_results[
                        (existing_results['Method'] == method) &
                        (existing_results['Size'] == size) &
                        (existing_results['Architecture'] == architecture) &
                        (existing_results['Quantization_Bits'] == bits)
                    ].empty:
                        print(f"Results for Method: {method}, Size: {size}, Architecture: {architecture}, Quantization Bits: {bits} already exist. Skipping...")
                        continue

                    input_size = len(selected_features)
                    
                    if architecture == 'MLP':
                        model = MLP(input_size=input_size, hidden_size=100, num_classes=2)
                        calibration_data = torch.FloatTensor(X_train)
                    else:  # CNN
                        model = Conv1DModel(input_size=input_size, num_classes=2)
                        calibration_data = torch.FloatTensor(X_train).unsqueeze(1) 

                    quantized_model, scales = apply_activation_aware_quantization(model, calibration_data, bits)

                    quantized_params = []
                    for name, param in quantized_model.named_parameters():
                        quantized_params.append(param.detach().cpu().numpy())

                    folder_path = f'/home/snakkill/pm/pe_new/mlp/quantization/{size}'
                    os.makedirs(folder_path, exist_ok=True)
                    for i, param in enumerate(quantized_params):
                        np.save(f'{folder_path}/{method}_{architecture}_{size}_bits_{bits}_param_{i}.npy', param)
                    np.save(f'{folder_path}/{method}_{architecture}_{size}_bits_{bits}_scales.npy', np.array(list(scales.values())))

                    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
                    grid = GridSearchCV(mlp, tuned_parameters, refit=True, verbose=3, cv=3)
                    grid.fit(X_train, y_train)

                    best_params = grid.best_params_
                    accuracy = grid.score(X_test, y_test)

                    model_file = os.path.join(folder_path, f'{method}_{architecture}_{size}_bits_{bits}.joblib')
                    joblib.dump(grid, model_file)

                    result = {
                        'Method': method,
                        'Size': size,
                        'Architecture': architecture,
                        'Quantization_Bits': bits,
                        'Selected_Features': selected_features,
                        'Best_Params': best_params,
                        'Accuracy': accuracy
                    }
                    results_df = pd.DataFrame([result])
                    if os.path.isfile(results_csv_path):
                        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
                    else:
                        results_df.to_csv(results_csv_path, index=False)

                    print(f"Method: {method}, Size: {size}, Architecture: {architecture}, Quantization Bits: {bits}, Best Params: {best_params}, Accuracy: {accuracy}")

if __name__ == "__main__":
    run_experiments()
