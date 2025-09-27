
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os


def evaluate_model():

    model_path = os.path.join('..', 'static', 'models', 'emotion_model.h5')
    test_data_dir = os.path.join('dataset', 'test')


    img_size = (48, 48)
    batch_size = 32
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


    model = tf.keras.models.load_model(model_path)


    print("\nModel Loss Function:", model.loss)
    print("Input Shape:", model.input_shape)
    print("Output Shape:", model.output_shape)


    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluating the model
    loss, accuracy = model.evaluate(test_generator)
    print(f'\nTest Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Generating the predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)


    y_true = np.argmax(test_generator.labels, axis=1) if test_generator.labels.ndim > 1 else test_generator.labels

    # making classification report
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred_classes, target_names=classes))


    cm = confusion_matrix(y_true, y_pred_classes)
    plt.ylabel('True Labels')
    plt.title('Emotion Classification Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    evaluate_model()