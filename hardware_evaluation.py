import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the weights from the .npy files
weights_layer0 = np.load('./5/mlp_new/weights_jmi_5_layer0.npy')
biases_layer0 = np.load('./5/mlp_new/biases_jmi_5_layer0.npy')
weights_layer1 = np.load('./5/mlp_new/weights_jmi_5_layer1.npy')
biases_layer1 = np.load('./5/mlp_new/biases_jmi_5_layer1.npy')
weights_layer2 = np.load('./5/mlp_new/weights_jmi_5_layer2.npy')
biases_layer2 = np.load('./5/mlp_new/biases_jmi_5_layer2.npy')

# Step 2: Print the shapes of the weights and biases
print(f"weights_layer0 shape: {weights_layer0.shape}")
print(f"biases_layer0 shape: {biases_layer0.shape[0]}")
print(f"weights_layer1 shape: {weights_layer1.shape}")
print(f"biases_layer1 shape: {biases_layer1.shape[0]}")
print(f"weights_layer2 shape: {weights_layer2.shape}")
print(f"biases_layer2 shape: {biases_layer2.shape[0]}")

X_train = np.load('./5/mlp_new/X_train_jmi_5.npy')
X_test = np.load('./5/mlp_new/X_test_jmi_5.npy')
Y_train = np.load('./5/mlp_new/y_train_jmi_5.npy')
Y_test = np.load('./5/mlp_new/y_test_jmi_5.npy')

# Step 2: Create an MLPClassifier with the appropriate architecture
# Make sure the architecture matches the number of layers and units in your loaded weights
model = MLPClassifier(hidden_layer_sizes=(biases_layer0.shape[0], biases_layer1.shape[0]))
model.fit(X_train, Y_train)
# Initialize the model to use the preloaded weights and biases
model.coefs_ = [weights_layer0, weights_layer1, weights_layer2]
model.intercepts_ = [biases_layer0, biases_layer1, biases_layer2]

# Step 4: Predict using the model
Y_pred = model.predict(X_test)

# Step 5: Calculate the accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Test Accuracy: {accuracy:.4f}')


# Step 4: Predict using the model
Y_pred = model.predict(X_train)

# Step 5: Calculate the accuracy
accuracy = accuracy_score(Y_train, Y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

