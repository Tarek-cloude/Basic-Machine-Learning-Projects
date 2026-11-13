import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

# 1. Load the MNIST dataset of handwritten digits
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# 2. Build a simple neural network model
model = models.Sequential([
    Input(shape=(28, 28)),                  # Input layer for 28x28 pixel images
    layers.Flatten(),                       # Flatten 2D images into a 1D array
    layers.Dense(128, activation='relu'),   # Hidden layer with ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer for 10 digit classes (0-9)
])

# 3. Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

# 4. Evaluate model performance on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")
