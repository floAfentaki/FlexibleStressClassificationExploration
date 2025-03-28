import numpy as np
import joblib
import os

# Folder containing the models and data (e.g., 'jmi' folder)
method_name = 'jmi'
folder_path = f'./{method_name}'

# Feature sizes you want to evaluate
feature_sizes = [5, 10, 15, 20, 25, 30]

# Load y_test once (since it doesn't change with feature sizes)
y_train = np.load(os.path.join(folder_path, "y_train.npy"))
y_test = np.load(os.path.join(folder_path, "y_test.npy"))
X_train = np.load(os.path.join(folder_path, f"X_train_{feature_sizes[-1]}.npy"))
X_test = np.load(os.path.join(folder_path, f"X_test_{feature_sizes[-1]}.npy"))

# Iterate over different feature sizes
for num_fea in feature_sizes:
    # Load the corresponding X_test for the current feature size
    X_train_sorted=X_train[:,:num_fea]
    X_test_sorted=X_test[:,:num_fea]
    # Load the trained model from the .joblib file
    model_filename = os.path.join(folder_path, f'features_{num_fea}.joblib')
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        print(f"\nLoaded model for {num_fea} features from {model_filename}")

        # Evaluate the model on the test data
        accuracy = model.score(X_train_sorted, y_train)
        print(f"Accuracy for model with {num_fea} features: {accuracy:.4f}")
        accuracy = model.score(X_test_sorted, y_test)
        print(f"Accuracy for model with {num_fea} features: {accuracy:.4f}")
    else:
        print(f"Model for {num_fea} features not found!")
