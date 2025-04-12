import tensorflow as tf

model = tf.keras.models.load_model("model_hybrid/hand_gesture_model_hybrid_data_add.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("hand_gesture_hybrid_data_add.tflite", "wb") as f:
    f.write(tflite_model)