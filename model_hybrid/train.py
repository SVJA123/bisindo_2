from tensorflow.keras.callbacks import EarlyStopping
from build import build_hybrid_model

def train_model(X_train_coords, X_train_angles, y_train, X_val_coords, X_val_angles, y_val, input_shape_coords, input_shape_angles, num_classes, epochs=30, batch_size=32):
    # build the hybrid model
    model = build_hybrid_model(input_shape_coords, input_shape_angles, num_classes)

    model.summary()

    # early stopping callback to monitor validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # train the model
    history = model.fit(
        [X_train_coords, X_train_angles], y_train,
        validation_data=([X_val_coords, X_val_angles], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history
