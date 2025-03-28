import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def save_model_params(model, folder_path):
    conv1_weights = model.conv1.weight.detach().numpy()
    conv1_bias = model.conv1.bias.detach().numpy()
    fc1_weights = model.fc1.weight.detach().numpy()
    fc1_bias = model.fc1.bias.detach().numpy()

    np.save(os.path.join(folder_path, 'conv1_weights.npy'), conv1_weights)
    np.save(os.path.join(folder_path, 'conv1_bias.npy'), conv1_bias)
    np.save(os.path.join(folder_path, 'fc1_weights.npy'), fc1_weights)
    np.save(os.path.join(folder_path, 'fc1_bias.npy'), fc1_bias)

def save_train_test_data(X_train, X_test, y_train, y_test, folder_path):
    np.save(os.path.join(folder_path, 'X_train.npy'), X_train)
    np.save(os.path.join(folder_path, 'X_test.npy'), X_test)
    np.save(os.path.join(folder_path, 'y_train.npy'), y_train)
    np.save(os.path.join(folder_path, 'y_test.npy'), y_test)

def train_conv1d_model(X_train, y_train, X_test, y_test, folder_path):
    print(f"Initial shape of X_train: {X_train.shape}")
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model = Conv1DModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], 1, -1)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], 1, -1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"Shape of X_train after reshaping: {X_train.shape}")

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted)
    
    save_model_params(model, folder_path)
    save_train_test_data(X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy(), folder_path)
    
    return model, accuracy

def run_conv1d_with_feature_sets():
    feature_selection_results_path = '../../svm/FS/feature_selection_results.csv'
    feature_selection_results = pd.read_csv(feature_selection_results_path)

    data = pd.read_csv('../../data/90sec_features_n_label_2_SMOTE_True.csv')
    y = data['label'].values

    methods = feature_selection_results['Method'].unique()
    feature_sizes = [5, 10, 15, 20, 25, 30]

    results_csv_path = './conv1d/FS/conv1d_results.csv'
    if os.path.exists(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
    else:
        existing_results = pd.DataFrame(columns=['Method', 'Size', 'Selected_Features', 'Accuracy'])

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
            print(f"Shape of X_selected before reshaping: {X_selected.shape}")
            X_selected = X_selected.reshape(X_selected.shape[0], -1)
            print(f"Shape of X_selected after reshaping: {X_selected.shape}")
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

            if not existing_results[
                (existing_results['Method'] == method) &
                (existing_results['Size'] == size)
            ].empty:
                print(f"Results for Method: {method}, Size: {size} already exist. Skipping...")
                continue

            folder_path = f'./conv1d/FS/new_arch/{size}'
            os.makedirs(folder_path, exist_ok=True)

            model, accuracy = train_conv1d_model(X_train, y_train, X_test, y_test, folder_path)

            model_file = os.path.join(folder_path, f'{method}_conv1d_{size}.pt')
            torch.save(model.state_dict(), model_file)

            result = {
                'Method': method,
                'Size': size,
                'Selected_Features': selected_features,
                'Accuracy': accuracy
            }
            results_df = pd.DataFrame([result])
            if os.path.isfile(results_csv_path):
                results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(results_csv_path, index=False)

            print(f"Method: {method}, Size: {size}, Accuracy: {accuracy}")

run_conv1d_with_feature_sets()