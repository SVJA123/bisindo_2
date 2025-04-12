from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam

def build_hybrid_model(input_shape_coords, input_shape_angles, num_classes):
    # for coordinates
    coords_input = Input(shape=input_shape_coords, name='coords_input')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(coords_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    # for angles
    angles_input = Input(shape=input_shape_angles, name='angles_input')
    y = Dense(32, activation='relu')(angles_input)
    y = Dropout(0.5)(y)

    # merge
    combined = concatenate([x, y])

    # fully connected layers
    z = Dense(256, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    z = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[coords_input, angles_input], outputs=z)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
