import numpy as np
import pandas as pd
from mlp2 import MLP, load_dataset

# Load data
X, y = load_dataset('data.csv')

# Load trained model
mlp = MLP(input_size=X.shape[1])
normalization_params = mlp.load_model('./saved_model.npy')

# Normalize data using saved parameters
if normalization_params is not None:
    mean = normalization_params['mean']
    std = normalization_params['std']
    X_normalized = (X - mean) / std

# Get predictions
predictions, probabilities = mlp.predict(X_normalized)

print(f"Prediction distribution: {np.unique(predictions, return_counts=True)}")
print(f"True label distribution: {np.unique(y, return_counts=True)}")
print(f"Overall accuracy: {np.mean(predictions == y):.4f}")

# Show some examples of each class prediction
print(f"\nSamples predicted as Class 0 (Benign):")
class_0_indices = np.where(predictions == 0)[0][:5]
for i in class_0_indices:
    print(f"  Sample {i}: pred=0, true={y[i]}, prob=[{probabilities[i][0]:.3f}, {probabilities[i][1]:.3f}]")

print(f"\nSamples predicted as Class 1 (Malignant):")
class_1_indices = np.where(predictions == 1)[0][:5]
for i in class_1_indices:
    print(f"  Sample {i}: pred=1, true={y[i]}, prob=[{probabilities[i][0]:.3f}, {probabilities[i][1]:.3f}]")

# Check prediction confidence
print(f"\nPrediction confidence statistics:")
print(f"Max probability per sample - Mean: {np.mean(np.max(probabilities, axis=1)):.3f}, Std: {np.std(np.max(probabilities, axis=1)):.3f}")
print(f"Min probability per sample - Mean: {np.mean(np.min(probabilities, axis=1)):.3f}, Std: {np.std(np.min(probabilities, axis=1)):.3f}")
