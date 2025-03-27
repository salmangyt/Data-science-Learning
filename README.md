# Hyperparameter Tuning in Neural Networks

## Overview
Hyperparameter tuning is a crucial step in optimizing neural networks. Unlike model parameters learned during training, hyperparameters must be set before training begins. Proper tuning can lead to improved accuracy, faster training, and better generalization to unseen data.

This notebook explores:
- Key hyperparameters in neural networks
- Various tuning techniques (Grid Search, Random Search, Bayesian Optimization)
- Practical implementation in TensorFlow/Keras

## Installation
Ensure you have the required dependencies installed before running the notebook:
```bash
pip install tensorflow keras keras-tuner matplotlib numpy
```

## Dataset
This notebook uses the **MNIST dataset**, a collection of handwritten digits commonly used for training image classification models.

### Loading the Dataset
```python
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
```

## Key Hyperparameters
1. **Learning Rate (lr)** - Controls the step size for updating weights.
2. **Batch Size** - Number of training samples used per iteration.
3. **Number of Layers & Neurons** - Determines model complexity.
4. **Activation Functions** - Introduce non-linearity into the model.
5. **Dropout & Regularization** - Prevents overfitting.

## Hyperparameter Tuning Methods

### 1. Grid Search
A brute-force method that evaluates all possible combinations of hyperparameters.
```python
from sklearn.model_selection import ParameterGrid

param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'batch_size': [32, 64, 128]}
all_params = list(ParameterGrid(param_grid))
print("Total combinations:", len(all_params))
```

### 2. Random Search
Selects random hyperparameter combinations, making it faster than Grid Search.
```python
from random import choice
random_params = {key: choice(values) for key, values in param_grid.items()}
print(random_params)
```

### 3. Bayesian Optimization
Uses probability models to find the best hyperparameters efficiently.
```python
from keras_tuner.tuners import BayesianOptimization

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hp.Int('units', 32, 512, step=32), activation='relu'),
        keras.layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0001])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = BayesianOptimization(build_model, objective='val_accuracy', max_trials=10)
tuner.search(x_train, y_train, epochs=5, validation_split=0.2)
```

## Results and Evaluation
After tuning, the best hyperparameters are selected based on validation accuracy.
```python
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)
```

### Metrics Considered:
- **Accuracy**: Measures model performance.
- **Loss**: Determines error in predictions.
- **Training Time**: Ensures computational efficiency.

## Conclusion & Best Practices
- Start with a small search space before expanding.
- Use **Bayesian Optimization** for complex models.
- Evaluate models on **validation sets** to avoid overfitting.
- Automate hyperparameter tuning with **Keras Tuner**.

## References
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Tuner: https://keras.io/guides/keras_tuner/
