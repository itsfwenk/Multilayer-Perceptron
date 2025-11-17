"""
A PyTorch rewrite of the original NumPy MLP in `mlp.py`.
This file implements:
- a configurable MLP using torch.nn.Module
- training loop with DataLoader
- save/load helpers
- command-line interface similar to the original

Notes:
- This file requires PyTorch (install with `pip install torch`).
- The code mirrors the original data loading and preprocessing in `mlp.py`.

Comments explain the common PyTorch syntax.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(filepath):
    """Load dataset from CSV file. Matches original `mlp.py` behavior.
    CSV assumed to have no header; features start at column index 2;
    column 1 contains labels 'M' or 'B'.
    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of integer labels (0 or 1)
    """
    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, 2:].values.astype(float)
    y_str = df.iloc[:, 1].values
    y = np.where(y_str == 'M', 1, 0)
    return X, y


def normalize_data(X_train, X_valid):
    """Normalize features using training mean/std and return normalization params."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    X_train_n = (X_train - mean) / std
    X_valid_n = (X_valid - mean) / std
    return X_train_n, X_valid_n, {'mean': mean, 'std': std}


def split_dataset(X, y, validation_split=0.2, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_valid = int(n * validation_split)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]


class MLPTorch(nn.Module):
    """A simple feed-forward MLP built with PyTorch.

    We build the network dynamically from a list of layer sizes.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        layers = []
        in_features = input_size

        # Create hidden layers: fully-connected + ReLU
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # Final layer (logits). No softmax here because
        # loss function (CrossEntropyLoss) applies log-softmax internally.
        layers.append(nn.Linear(in_features, output_size))

        # nn.Sequential chains the layers into a single module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass. PyTorch convention: implement computation in forward().

        Args:
            x: Tensor of shape (batch_size, input_size)
        Returns:
            logits: raw outputs (before softmax) shape (batch_size, output_size)
        """
        return self.net(x)


def train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=10):
    """Train the model with a standard PyTorch training loop.

    Explanation of common PyTorch concepts used here:
    - model.train()/model.eval(): toggles behaviors like dropout or batchnorm.
    - optimizer.zero_grad(): clears accumulated gradients from previous step.
    - loss.backward(): computes gradients via backpropagation.
    - optimizer.step(): updates parameters using gradients.
    - torch.no_grad(): context manager to disable autograd for evaluation.
    """
    model.to(device)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)  # forward pass
            loss = criterion(logits, y_batch)  # compute loss
            loss.backward()  # backward pass (compute gradients)
            optimizer.step()  # update weights

            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:03d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    return model


def save_model(model, filepath, normalization_params=None):
    """Save model state_dict and normalization parameters in a dict."""
    data = {
        'state_dict': model.state_dict(),
        'normalization': normalization_params
    }
    torch.save(data, filepath)
    print(f"> saved model to {filepath}")


def load_model(filepath, model_class, *model_args, device='cpu'):
    """Load a checkpoint and return a model instance plus normalization params.

    This function handles several formats:
    - PyTorch checkpoints saved with `torch.save({'state_dict': ..., 'normalization': ...})`.
    - Raw state_dicts (a dict of parameter tensors).
    - NumPy `.npy` files produced by the original `mlp.py` (contains 'weights'/'biases').

    It also makes a best-effort attempt to work around PyTorch's stricter
    unpickling defaults by adding NumPy's reconstruct function to the
    safe globals, or by reloading with `weights_only=False` if necessary.
    """
    import os
    import pickle

    ext = os.path.splitext(filepath)[1].lower()

    # Helper to construct a model and load a state_dict-like mapping
    def _load_state_dict_into_model(state_dict, norm=None):
        model = model_class(*model_args)
        model.load_state_dict(state_dict)
        model.to(device)
        return model, norm

    # If this is a NumPy-saved checkpoint (.npy), load with numpy and
    # convert the stored 'weights'/'biases' into the PyTorch state_dict.
    if ext == '.npy':
        raw = np.load(filepath, allow_pickle=True).item()
        # If it contains a PyTorch-like state_dict, use it directly.
        if isinstance(raw, dict) and 'state_dict' in raw:
            return _load_state_dict_into_model(raw['state_dict'], raw.get('normalization') or raw.get('normalization_params'))

        # If it contains NumPy-style 'weights' and 'biases' (from original mlp.py)
        if isinstance(raw, dict) and 'weights' in raw and 'biases' in raw:
            model = model_class(*model_args)
            sd = model.state_dict()
            # Find Linear layer indices inside the Sequential module
            linear_indices = [i for i, layer in enumerate(model.net) if isinstance(layer, nn.Linear)]
            for j, layer_idx in enumerate(linear_indices):
                w_np = raw['weights'][j]  # shape (in, out)
                b_np = raw['biases'][j]
                key_w = f'net.{layer_idx}.weight'
                key_b = f'net.{layer_idx}.bias'
                # PyTorch Linear weight shape is (out, in)
                sd[key_w] = torch.from_numpy(w_np.T.copy()).to(device)
                sd[key_b] = torch.from_numpy(np.squeeze(b_np)).to(device)
            model.load_state_dict(sd)
            model.to(device)
            # normalization params might be stored under different keys
            norm = raw.get('normalization_params') or raw.get('normalization') or raw.get('normalization_params')
            return model, norm

        # Unknown .npy format
        raise RuntimeError(f"Unrecognized .npy checkpoint format: {filepath}")

    # Otherwise, try loading using torch.load with increasing permissiveness
    try:
        data = torch.load(filepath, map_location=device)
    except Exception:
        # First attempt to allow necessary NumPy globals in a safe way.
        try:
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        except Exception:
            # If the helper isn't available in this PyTorch version, ignore and continue
            pass

        # Re-try load allowing non-weights-only objects. This is more permissive
        # and may execute code from the checkpoint; do this only for trusted files.
        data = torch.load(filepath, map_location=device, weights_only=False)

    # Interpret loaded object
    if isinstance(data, dict):
        # Common patterns: {'state_dict': ..., 'normalization': ...}
        if 'state_dict' in data:
            return _load_state_dict_into_model(data['state_dict'], data.get('normalization') or data.get('normalization_params'))

        # Some checkpointers use 'model_state_dict'
        if 'model_state_dict' in data:
            return _load_state_dict_into_model(data['model_state_dict'], data.get('normalization') or data.get('normalization_params'))

        # If the dict looks like a raw state_dict (keys like 'net.0.weight'), try loading
        sample_keys = list(data.keys())[:5]
        if any(k.startswith('net.') or k.endswith('.weight') for k in sample_keys):
            return _load_state_dict_into_model(data, None)

        # If the checkpoint uses NumPy-style arrays saved inside a torch file
        if 'weights' in data and 'biases' in data:
            model = model_class(*model_args)
            sd = model.state_dict()
            linear_indices = [i for i, layer in enumerate(model.net) if isinstance(layer, nn.Linear)]
            for j, layer_idx in enumerate(linear_indices):
                w_np = data['weights'][j]
                b_np = data['biases'][j]
                key_w = f'net.{layer_idx}.weight'
                key_b = f'net.{layer_idx}.bias'
                sd[key_w] = torch.from_numpy(w_np.T.copy()).to(device)
                sd[key_b] = torch.from_numpy(np.squeeze(b_np)).to(device)
            model.load_state_dict(sd)
            model.to(device)
            return model, data.get('normalization_params') or data.get('normalization')

    # If we reached here, try to assume the file itself is a state_dict mapping
    try:
        model = model_class(*model_args)
        model.load_state_dict(data)
        model.to(device)
        return model, None
    except Exception as e:
        raise RuntimeError(f"Unsupported checkpoint format or failed to load model: {e}") from e


def predict(model, X, device='cpu'):
    """Make predictions and return (preds_numpy, probs_numpy)."""
    model.eval()
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=32):
    """Convert numpy arrays to DataLoaders for PyTorch training."""
    X_tr_t = torch.from_numpy(X_train.astype(np.float32))
    y_tr_t = torch.from_numpy(y_train.astype(np.longlong))
    X_val_t = torch.from_numpy(X_valid.astype(np.float32))
    y_val_t = torch.from_numpy(y_valid.astype(np.longlong))

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    valid_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser(description='PyTorch MLP rewrite')
    parser.add_argument('--dataset', type=str, default='data.csv')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--model', type=str, default='saved_model.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 16])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Device selection: CPU or GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")

    X, y = load_dataset(args.dataset)

    if args.mode == 'train':
        X_train, X_valid, y_train, y_valid = split_dataset(X, y, validation_split=0.2, seed=args.seed)
        X_train, X_valid, norm = normalize_data(X_train, X_valid)

        # Build model
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y))
        model = MLPTorch(input_size=input_size, hidden_layers=args.hidden, output_size=num_classes)

        # Build dataloaders
        train_loader, valid_loader = build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=args.batch_size)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()  # expects raw logits and integer class labels
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        # Train
        trained = train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=args.epochs)

        # Save model + normalization params
        save_model(trained, args.model, normalization_params=norm)

    else:  # predict
        # We need to reconstruct model architecture; assume same defaults used during training
        input_size = X.shape[1]
        num_classes = len(np.unique(y))
        model, norm = load_model(args.model, MLPTorch, input_size, args.hidden, num_classes, device=device)

        if norm is not None:
            mean = norm['mean']
            std = norm['std']
            std[std == 0] = 1.0
            X_n = (X - mean) / std
        else:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1.0
            X_n = (X - X_mean) / X_std

        preds, probs = predict(model, X_n, device=device)
        loss_fn = nn.CrossEntropyLoss()
        # Compute loss for full dataset
        logits = model(torch.from_numpy(X_n.astype(np.float32)).to(device))
        loss = loss_fn(logits, torch.from_numpy(y.astype(np.longlong)).to(device)).item()

        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {np.mean(preds == y):.4f}")
        print("First 10 predictions:", preds[:10])
        print("First 10 true:", y[:10])


if __name__ == '__main__':
    main()
