# https://towardsdatascience.com/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c/
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Create our simple 2D dataset
df = pd.DataFrame({
    '🌞   ': [0, 1, 1, 2, 3, 3, 2, 3, 0, 0, 1, 2, 3],
    '💧   ': [0, 0, 1, 0, 1, 2, 3, 3, 1, 2, 3, 2, 1],
    'y': [1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
}, index=range(1, 14))

# Split into training and test sets
train_df, test_df = df.iloc[:8].copy(), df.iloc[8:].copy()
X_train, y_train = train_df[['🌞   ', '💧   ']], train_df['y']
X_test, y_test = test_df[['🌞   ', '💧   ']], test_df['y']

# Create and configure our neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(3, 2), # Creates a 2-3-2-1 architecture as discussed
    activation='relu',         # ReLU activation for hidden layers
    solver='sgd',              # Stochastic Gradient Descent optimizer
    learning_rate_init=0.1,    # Step size for weight updates
    max_iter=1000,             # Maximum number of epochs
    momentum=0,                # Disable momentum for pure SGD as discussed
    random_state=42            # For reproducible results
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")