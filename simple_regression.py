import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

# Simple training data representing a linear relationship (y = 2x - 1 roughly)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Build a single-layer neural network model
model = Sequential([
    Input(shape=(1,)),
    Dense(units=1)
])

# Compile the model with SGD optimizer and mean squared error loss
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model for 500 epochs
model.fit(xs, ys, epochs=500, verbose=0)

# Test prediction for a new value (e.g., input 10.0)
print("Predicted value for 10 is:", model.predict(np.array([[10.0]])))
