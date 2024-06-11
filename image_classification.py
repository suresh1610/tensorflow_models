import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import logging

from tensorflow import keras
from tensorflow.keras import layers

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

num_classes = len(class_names)

input_shape = (180, 180, 3)
model = keras.Sequential()
model.add(tf.keras.Input(shape=(input_shape)))  # Explicit Input layer
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

save_model_dir = '/home/suresh/internship/c_tf_training/savedmodel'

tf.saved_model.save(model, save_model_dir)


converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir)

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()


tflite_model_path = '/home/suresh/internship/c_tf_training/savedmodel/model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
# # Save the Keras model
# keras_file = "image_class.keras"
# keras.models.save_model(model, keras_file)

# # Load the saved Keras model
# model = tf.keras.models.load_model(keras_file)

# # Convert to TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# # Enable logging for debugging
# logging.basicConfig(level=logging.DEBUG)

# # Convert the model
# tflite_model = converter.convert()

# # Save the TensorFlow Lite model
# tflite_model_file = "model.tflite"
# with open(tflite_model_file, 'wb') as f:
#     f.write(tflite_model)

# print("TensorFlow Lite model saved successfully.")


