# Multilayer Perceptron for Breast Cancer Classification

A from-scratch implementation of a Multilayer Perceptron (MLP) neural network for binary classification of breast cancer tumors as malignant or benign.

## 🎯 Project Overview

This project implements a fully functional neural network without using high-level machine learning frameworks like TensorFlow or PyTorch. The implementation includes:

- **Forward propagation** with sigmoid and softmax activations
- **Backpropagation** with gradient descent optimization
- **Xavier/Glorot weight initialization** for stable training
- **Data normalization** and proper train/validation splitting
- **Model persistence** (save/load functionality)
- **Training visualization** with loss and accuracy plots

## 🏥 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains:
- **569 samples** of breast cancer biopsies
- **30 features** computed from digitized images of cell nuclei
- **2 classes**: Malignant (M) and Benign (B)

### Features include:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points, symmetry, fractal dimension
- Mean, standard error, and "worst" values for each measurement

## 🚀 Performance

The model achieves:
- **98.23% validation accuracy**
- **97.59% training accuracy**
- **No overfitting** (validation accuracy ≥ training accuracy)
- **Stable convergence** in ~60-70 epochs

## 📋 Requirements

```bash
pip install numpy pandas matplotlib
```

## 🔧 Usage

### Training a New Model

```bash
# Basic training with default parameters
python mlp2.py --mode train

# Custom training with specific parameters
python mlp2.py --mode train \
    --epochs 100 \
    --learning_rate 0.1 \
    --hidden_layers 64 32 16 \
    --model ./my_model.npy
```

### Making Predictions

```bash
# Predict using saved model
python mlp2.py --mode predict --model ./saved_model.npy
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data.csv` | Path to dataset CSV file |
| `--mode` | `train` | Mode: `train` or `predict` |
| `--model` | `./saved_model.npy` | Path to save/load model |
| `--epochs` | `70` | Number of training epochs |
| `--learning_rate` | `0.1` | Learning rate for optimization |
| `--hidden_layers` | `[32, 16]` | List of hidden layer sizes |
| `--seed` | `42` | Random seed for reproducibility |

## 🏗️ Architecture

### Default Network Structure
```
Input Layer (30 features)
    ↓
Hidden Layer 1 (32 neurons, Sigmoid)
    ↓
Hidden Layer 2 (16 neurons, Sigmoid)
    ↓
Output Layer (2 classes, Softmax)
```

### Key Components

1. **Xavier/Glorot Initialization**: Prevents vanishing/exploding gradients
2. **Sigmoid Activation**: For hidden layers with gradient clipping
3. **Softmax Output**: For probability distribution over classes
4. **Binary Cross-Entropy Loss**: Optimized for binary classification
5. **Mini-batch Gradient Descent**: Efficient weight updates

## 📊 Training Output

```
x_train shape : (456, 30)
x_valid shape : (113, 30)
epoch 01/100 - loss: 0.8459 - val_loss: 0.6959 - acc: 0.3728 - val_acc: 0.6283
epoch 02/100 - loss: 0.6939 - val_loss: 0.6752 - acc: 0.6250 - val_acc: 0.6283
...
epoch 100/100 - loss: 0.1048 - val_loss: 0.0865 - acc: 0.9759 - val_acc: 0.9823
> saving model './saved_model.npy' to disk...
```

## 📈 Visualization

The training process automatically generates plots showing:
- **Loss curves**: Training vs validation loss over epochs
- **Accuracy curves**: Training vs validation accuracy over epochs

Both plots help identify:
- Convergence behavior
- Overfitting (if validation metrics diverge)
- Optimal stopping points

## 🔬 Technical Implementation

### Forward Propagation
```python
# Hidden layers with sigmoid activation
for i in range(len(self.weights) - 1):
    z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
    a = self.sigmoid(z)
    activations.append(a)

# Output layer with softmax activation
z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
a = self.softmax(z)
```

### Backpropagation
```python
# Compute gradients using chain rule
delta = activations[-1] - y  # Softmax + Cross-entropy gradient
for i in range(len(self.weights) - 1, -1, -1):
    grad_w = np.dot(activations[i].T, delta) / m
    grad_b = np.sum(delta, axis=0, keepdims=True) / m
    # Propagate error backward
    if i > 0:
        delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
```

### Weight Initialization
```python
# Xavier/Glorot initialization for stable gradients
weight_matrix = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
```

## 📁 Project Structure

```
├── mlp2.py              # Main implementation
├── data.csv             # Breast cancer dataset
├── saved_model.npy      # Trained model (created after training)
├── .gitignore          # Git ignore file
├── README.md           # This file
└── debug files/        # Development and testing scripts
    ├── debug.py
    ├── debug2.py
    ├── debug3.py
    └── final_test.py
```

## 🧠 Key Learning Points

1. **Importance of Proper Initialization**: Xavier initialization prevents gradient problems
2. **Validation Set Usage**: Essential for detecting overfitting and model selection
3. **Gradient Computation**: Understanding backpropagation through matrix operations
4. **Data Preprocessing**: Normalization crucial for neural network performance
5. **Learning Rate Selection**: Balance between convergence speed and stability

## 🎓 Educational Value

This implementation serves as an excellent educational resource for understanding:
- Neural network fundamentals from first principles
- Gradient descent optimization
- Forward and backward propagation
- Proper machine learning workflow (train/validation/test)
- Python numerical computing with NumPy

## 🚀 Future Enhancements

Potential improvements could include:
- [ ] Batch/mini-batch training for larger datasets
- [ ] Different activation functions (ReLU, Leaky ReLU)
- [ ] Regularization techniques (L1/L2, dropout)
- [ ] Learning rate scheduling
- [ ] Early stopping based on validation loss
- [ ] Cross-validation for robust evaluation
- [ ] ROC curves and additional metrics

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Breast Cancer Wisconsin (Diagnostic) Dataset from UCI Machine Learning Repository
- Implementation inspired by foundational neural network research

---

**Note**: This is an educational implementation. For production use, consider using established frameworks like TensorFlow, PyTorch, or scikit-learn.
