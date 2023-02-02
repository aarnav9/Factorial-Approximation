import numpy as np
import tensorflow as tf
import pandas as pd

# Load data from Excel file
data = pd.read_excel('factorial_data.xlsx')
n = data['n'].values
y = np.log(data['factorial'].values)

# Define the optimal number of neurons
neurons = min(2048, 2 * int(n.shape[0] / 3))

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(neurons, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(neurons, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Use early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train and update the model multiple times
approximated_factorial = np.zeros(n.shape[0])
for i in range(100):
    # Train the model
    history = model.fit(n, y, epochs=500, verbose=1)
    
    # Use the trained model to generate an equation
    weights = model.layers[0].kernel.numpy()
    bias = model.layers[0].bias.numpy()

    a = weights[0][0]
    b = bias[0]
    approximated_factorial = np.exp(a*n + b)

# Calculate the final error
error = np.abs(np.exp(approximated_factorial) - np.exp(y))
mean_error = np.mean(error)
print('Final mean error:', mean_error)
    
# Output the final equation
print('The final equation is: f(n) = exp(%f * n + %f)' % (a, b))
