import numpy as np
import argparse
import sys

class MLP:
    def __init__(self, input_size, hidden_layers=[32, 16], output_size=2, learning_rate=0.01, seed=42):
        """
        Initialize the multilayer perceptron with at least 2 hidden layers.

        Args:
            input_size: Number of input features
            hidden_layers: List of neurons in each hidden layer (default: [32, 16])
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            # He initialization for better convergence
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Derivative of sigmoid"""
        return a * (1 - a)

    def softmax(self, z):
        """Softmax activation for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation through the network.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            activations: List of activations for each layer
        """
        activations = [X]

        # Forward through hidden layers with sigmoid
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

        # Output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.softmax(z)
        activations.append(a)

        return activations

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Calculate binary cross-entropy loss.

        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities

        Returns:
            loss: Average loss
        """
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def backward(self, X, y, activations):
        """
        Backward propagation to compute gradients.

        Args:
            X: Input data
            y: True labels (one-hot encoded)
            activations: List of activations from forward pass
        """
        m = X.shape[0]
        gradients_w = []
        gradients_b = []

        # Output layer gradient (softmax + cross-entropy)
        delta = activations[-1] - y

        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X_train, y_train, X_valid, y_valid, epochs=70):
        """
        Train the neural network.

        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_valid: Validation data
            y_valid: Validation labels (one-hot encoded)
            epochs: Number of training epochs
        """
        print(f"x_train shape : {X_train.shape}")
        print(f"x_valid shape : {X_valid.shape}")

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X_train)

            # Backward pass
            self.backward(X_train, y_train, activations)

            # Calculate losses
            train_loss = self.binary_cross_entropy(y_train, activations[-1])

            valid_activations = self.forward(X_valid)
            valid_loss = self.binary_cross_entropy(y_valid, valid_activations[-1])

            print(f"epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {valid_loss:.4f}")

    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X: Input data

        Returns:
            predictions: Predicted class labels
            probabilities: Predicted probabilities
        """
        activations = self.forward(X)
        probabilities = activations[-1]
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    def save_model(self, filepath):
        """Save model weights and architecture to disk"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layers': self.layers,
            'learning_rate': self.learning_rate
        }
        np.save(filepath, model_data)
        print(f"> saving model '{filepath}' to disk...")

    def load_model(self, filepath):
        """Load model weights and architecture from disk"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layers = model_data['layers']
        self.learning_rate = model_data['learning_rate']
        print(f"> loading model '{filepath}' from disk...")


def load_dataset(filepath):
    """Load dataset from CSV file"""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def split_dataset(X, y, validation_split=0.2, seed=42):
    """Split dataset into training and validation sets"""
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    n_valid = int(n_samples * validation_split)
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]

    X_train, X_valid = X[train_indices], X[valid_indices]
    y_train, y_valid = y[train_indices], y[valid_indices]

    return X_train, X_valid, y_train, y_valid


def normalize_data(X_train, X_valid):
    """Normalize features using mean and std from training set"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    return X_train, X_valid


def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding"""
    n_samples = y.shape[0]
    y_encoded = np.zeros((n_samples, num_classes))
    y_encoded[np.arange(n_samples), y] = 1
    return y_encoded


def main():
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Mode: train or predict')
    parser.add_argument('--model', type=str, default='./saved_model.npy',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32, 16],
                        help='Number of neurons in hidden layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load dataset
    X, y = load_dataset(args.dataset)

    if args.mode == 'train':
        # Split dataset
        X_train, X_valid, y_train, y_valid = split_dataset(X, y, validation_split=0.2, seed=args.seed)

        # Normalize data
        X_train, X_valid = normalize_data(X_train, X_valid)

        # One-hot encode labels
        num_classes = len(np.unique(y))
        y_train_encoded = one_hot_encode(y_train, num_classes)
        y_valid_encoded = one_hot_encode(y_valid, num_classes)

        # Create and train model
        mlp = MLP(
            input_size=X_train.shape[1],
            hidden_layers=args.hidden_layers,
            output_size=num_classes,
            learning_rate=args.learning_rate,
            seed=args.seed
        )

        mlp.train(X_train, y_train_encoded, X_valid, y_valid_encoded, epochs=args.epochs)

        # Save model
        mlp.save_model(args.model)

    elif args.mode == 'predict':
        # Load model
        mlp = MLP(input_size=X.shape[1])  # Dummy initialization
        mlp.load_model(args.model)

        # Normalize data (in practice, save normalization params during training)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_normalized = (X - X_mean) / X_std

        # Make predictions
        predictions, probabilities = mlp.predict(X_normalized)

        # One-hot encode true labels for loss calculation
        num_classes = len(np.unique(y))
        y_encoded = one_hot_encode(y, num_classes)

        # Calculate loss
        loss = mlp.binary_cross_entropy(y_encoded, probabilities)

        print(f"\nPrediction Results:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {np.mean(predictions == y):.4f}")
        print(f"\nFirst 10 predictions: {predictions[:10]}")
        print(f"First 10 true labels: {y[:10]}")


if __name__ == "__main__":
    main()