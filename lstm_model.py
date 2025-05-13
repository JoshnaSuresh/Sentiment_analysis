import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and verify data
data = np.load("training_data.npz")
X, y = data['X'], data['y']
print(f"X shape: {X.shape}, y shape: {y.shape}")  # Should be (samples, 7, 6) and (samples,)

# 2. Fixed model architecture
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True)(inputs)
    att = Attention()([lstm, lstm])
    
    # Add Flatten before final Dense layer
    flattened = Flatten()(att)
    
    output = Dense(1, activation='sigmoid')(flattened)
    return tf.keras.Model(inputs, output)

# 3. Build and compile
model = build_model(X.shape[1:])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train with callbacks
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

model.save("stock_predictor_fixed.h5")