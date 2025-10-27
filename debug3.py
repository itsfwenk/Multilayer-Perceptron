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

# Try with higher learning rate
mlp = MLP(input_size=X_train.shape[1], hidden_layers=[32, 16], output_size=num_classes, learning_rate=0.1, seed=42)

print(f"Training with learning rate 0.1 for 50 epochs...")
mlp.train(X_train, y_train_encoded, X_valid, y_valid_encoded, epochs=50)

# Check final predictions
final_preds, final_probs = mlp.predict(X_valid)
print(f"\nFinal predictions distribution: {np.unique(final_preds, return_counts=True)}")
print(f"Final accuracy: {np.mean(final_preds == y_valid):.4f}")

print(f"\nSample of final probabilities:")
for i in range(10):
    print(f"Sample {i}: probs=[{final_probs[i][0]:.3f}, {final_probs[i][1]:.3f}], pred={final_preds[i]}, true={y_valid[i]}")

print(f"\nProbability statistics:")
print(f"Class 0 prob - Mean: {np.mean(final_probs[:, 0]):.4f}, Std: {np.std(final_probs[:, 0]):.4f}")
print(f"Class 1 prob - Mean: {np.mean(final_probs[:, 1]):.4f}, Std: {np.std(final_probs[:, 1]):.4f}")
