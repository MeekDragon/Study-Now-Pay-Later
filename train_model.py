import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

def create_data_flow():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/train',
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=32,
        label_mode='categorical',
        shuffle=True,
        seed=42
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/val',
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=32,
        label_mode='categorical',
        shuffle=False
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    return train_ds, val_ds


def create_proper_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    train_ds, val_ds = create_data_flow()
    model = create_proper_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )

    print(f"Final Validation Accuracy: {max(history.history['val_accuracy']):.2%}")
    model.save('emotion_model.h5')

if __name__ == "__main__":
    train_model()