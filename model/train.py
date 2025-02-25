from tensorflow.keras.callbacks import EarlyStopping
from build import build_cnn_model

def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=30, batch_size=32):
    # Build the model
    model = build_cnn_model(input_shape, num_classes)

    # Early stopping callback to monitor validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return model