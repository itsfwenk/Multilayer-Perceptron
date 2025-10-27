import numpy as np
import pandas as pd
from mlp import MLP, load_dataset, split_dataset, normalize_data, one_hot_encode

# Load and prepare data
X, y = load_dataset('data.csv')
X_train, X_valid, y_train, y_valid = split_dataset(X, y, validation_split=0.2, seed=42)
X_train, X_valid = normalize_data(X_train, X_valid)

num_classes = len(np.unique(y))
y_train_encoded = one_hot_encode(y_train, num_classes)
y_valid_encoded = one_hot_encode(y_valid, num_classes)

print(f"Training set balance: Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"Validation set balance: Class 0: {np.sum(y_valid == 0)}, Class 1: {np.sum(y_valid == 1)}")

# Create model
mlp = MLP(input_size=X_train.shape[1], hidden_layers=[32, 16], output_size=num_classes, learning_rate=0.01, seed=42)

# Check initial predictions
initial_preds, initial_probs = mlp.predict(X_valid)
print(f"\nInitial predictions distribution: {np.unique(initial_preds, return_counts=True)}")
print(f"Initial accuracy: {np.mean(initial_preds == y_valid):.4f}")

# Train for a few epochs and check what happens
print(f"\nTraining for 10 epochs...")
mlp.train(X_train, y_train_encoded, X_valid, y_valid_encoded, epochs=10)

# Check final predictions
final_preds, final_probs = mlp.predict(X_valid)
print(f"\nFinal predictions distribution: {np.unique(final_preds, return_counts=True)}")
print(f"Final accuracy: {np.mean(final_preds == y_valid):.4f}")

# Check some probability distributions
print(f"\nSample of final probabilities:")
print(f"Probabilities for first 10 samples:")
for i in range(10):
    print(f"Sample {i}: probs={final_probs[i]}, pred={final_preds[i]}, true={y_valid[i]}")

# Check if the model is getting stuck
print(f"\nProbability statistics:")
print(f"Class 0 prob - Mean: {np.mean(final_probs[:, 0]):.4f}, Std: {np.std(final_probs[:, 0]):.4f}")
print(f"Class 1 prob - Mean: {np.mean(final_probs[:, 1]):.4f}, Std: {np.std(final_probs[:, 1]):.4f}")

# Check weights after training
print(f"\nFinal weights statistics:")
for i, w in enumerate(mlp.weights):
    print(f"Layer {i}: mean={np.mean(w):.6f}, std={np.std(w):.6f}, min={np.min(w):.6f}, max={np.max(w):.6f}")
