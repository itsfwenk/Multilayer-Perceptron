import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_layers=[32, 16], output_size=1, learning_rate=0.01, seed=42):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / (self.layers[i] + self.layers[i + 1]))
            bias_vector = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
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

        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

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
        N = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / N
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

        delta = activations[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

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
        t_loss = np.zeros(epochs)
        v_loss = np.zeros(epochs)
        t_accuracy = np.zeros(epochs)
        v_accuracy = np.zeros(epochs)
        epochs_list = np.arange(epochs)

        y_train_labels = np.argmax(y_train, axis=1)
        y_valid_labels = np.argmax(y_valid, axis=1)

        for epoch in range(epochs):
            activations = self.forward(X_train)
            self.backward(X_train, y_train, activations)
            train_loss = self.binary_cross_entropy(y_train, activations[-1])
            valid_activations = self.forward(X_valid)
            valid_loss = self.binary_cross_entropy(y_valid, valid_activations[-1])

            train_pred = np.argmax(activations[-1], axis=1)
            valid_pred = np.argmax(valid_activations[-1], axis=1)
            train_acc = np.mean(train_pred == y_train_labels)
            valid_acc = np.mean(valid_pred == y_valid_labels)

            print(f"epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {valid_loss:.4f} - acc: {train_acc:.4f} - val_acc: {valid_acc:.4f}")

            t_loss[epoch] = train_loss
            v_loss[epoch] = valid_loss
            t_accuracy[epoch] = train_acc
            v_accuracy[epoch] = valid_acc

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(epochs_list, t_loss, label='Training Loss')
        axes[0].plot(epochs_list, v_loss, label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_list, t_accuracy, label='Training Accuracy')
        axes[1].plot(epochs_list, v_accuracy, label='Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return t_loss, v_loss

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

    def save_model(self, filepath, normalization_params=None):
        """Save model weights and architecture to disk"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'normalization_params': normalization_params
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
        return model_data.get('normalization_params', None)


def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding"""
    n_samples = y.shape[0]
    y_encoded = np.zeros((n_samples, num_classes))
    y_encoded[np.arange(n_samples), y] = 1
    return y_encoded

def normalize_data(X_train, X_valid):
    """Normalize features using mean and std from training set"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    return X_train, X_valid, {'mean': mean, 'std': std}

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

def load_dataset(filepath):
    """Load dataset from CSV file"""


    df = pd.read_csv(filepath, header=None)

    X = df.iloc[:, 2:].values.astype(float)

    y_str = df.iloc[:, 1].values
    y = np.where(y_str == 'M', 1, 0)  # M=1 (malignant), B=0 (benign)

    return X, y

def main():
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--dataset', type=str, default='data.csv', help='Path to dataset CSV file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Mode: train or predict')
    parser.add_argument('--model', type=str, default='./saved_model.npy',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32, 16],
                        help='Number of neurons in hidden layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    X, y = load_dataset(args.dataset)


    if args.mode == 'train':
        X_train, X_valid, y_train, y_valid = split_dataset(X, y, validation_split=0.2, seed=args.seed)

        X_train, X_valid, normalization_params = normalize_data(X_train, X_valid)

        num_classes = len(np.unique(y))
        y_train_encoded = one_hot_encode(y_train, num_classes)
        y_valid_encoded = one_hot_encode(y_valid, num_classes)

        mlp = MLP(
            input_size=X_train.shape[1],
            hidden_layers=args.hidden_layers,
            output_size=num_classes,
            learning_rate=args.learning_rate,
            seed=args.seed
        )

        mlp.train(X_train, y_train_encoded, X_valid, y_valid_encoded, epochs=args.epochs)
        mlp.save_model(args.model, normalization_params)

    elif args.mode == 'predict':
        mlp = MLP(input_size=X.shape[1])
        normalization_params = mlp.load_model(args.model)

        if normalization_params is not None:
            mean = normalization_params['mean']
            std = normalization_params['std']
            X_normalized = (X - mean) / std
        else:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1
            X_normalized = (X - X_mean) / X_std

        predictions, probabilities = mlp.predict(X_normalized)

        num_classes = len(np.unique(y))
        y_encoded = one_hot_encode(y, num_classes)

        loss = mlp.binary_cross_entropy(y_encoded, probabilities)

        print(f"\nPrediction Results:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {np.mean(predictions == y):.4f}")
        print(f"\nFirst 10 predictions: {predictions[:10]}")
        print(f"First 10 true labels: {y[:10]}")
        print(f"Last 10 predictions: {predictions[-10:]}")
        print(f"Last 10 true labels: {y[-10:]}")


if __name__ == "__main__":
    main()