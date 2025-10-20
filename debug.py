import numpy as np
import pandas as pd
from mlp2 import MLP, load_dataset, split_dataset, normalize_data, one_hot_encode

# Load and prepare data
X, y = load_dataset('data.csv')
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Unique labels: {np.unique(y, return_counts=True)}")

# Split data
X_train, X_valid, y_train, y_valid = split_dataset(X, y, validation_split=0.2, seed=42)
X_train, X_valid = normalize_data(X_train, X_valid)

# Encode labels
num_classes = len(np.unique(y))
y_train_encoded = one_hot_encode(y_train, num_classes)
y_valid_encoded = one_hot_encode(y_valid, num_classes)

print(f"Number of classes: {num_classes}")
print(f"Training labels distribution: {np.unique(y_train, return_counts=True)}")
print(f"Validation labels distribution: {np.unique(y_valid, return_counts=True)}")

# Create and test model
mlp = MLP(input_size=X_train.shape[1], hidden_layers=[32, 16], output_size=num_classes, learning_rate=0.01, seed=42)

# Test forward pass
print(f"\nModel architecture: {mlp.layers}")
print(f"Weights shapes: {[w.shape for w in mlp.weights]}")
print(f"Biases shapes: {[b.shape for b in mlp.biases]}")

# Test with a small sample
sample_X = X_train[:5]
activations = mlp.forward(sample_X)
print(f"\nForward pass on 5 samples:")
print(f"Input shape: {sample_X.shape}")
print(f"Final activations shape: {activations[-1].shape}")
print(f"Final activations (probabilities): \n{activations[-1]}")

# Check predictions
predictions, probabilities = mlp.predict(sample_X)
print(f"Predictions: {predictions}")
print(f"True labels for these 5 samples: {y_train[:5]}")

# Check if softmax is working correctly
print(f"\nSoftmax output sums: {np.sum(activations[-1], axis=1)}")

# Check weights initialization
print(f"\nInitial weights statistics:")
for i, w in enumerate(mlp.weights):
    print(f"Layer {i}: mean={np.mean(w):.6f}, std={np.std(w):.6f}, min={np.min(w):.6f}, max={np.max(w):.6f}")
