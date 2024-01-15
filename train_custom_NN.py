# coding = utf-8

# IMPORTS---------------------------------------------------------------------------------------------------------------
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Data preprocessing
rescale_dataset = tf.keras.Sequential([
    keras.layers.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([ 
  keras.layers.RandomFlip("horizontal_and_vertical"),
  keras.layers.RandomRotation(0.5),
  keras.layers.RandomZoom(0.5),
  keras.layers.RandomContrast(0.5),
  keras.layers.RandomTranslation(0.25, 0.25),
])

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (rescale_dataset(x), y), 
              num_parallel_calls=AUTOTUNE)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

# LOAD TRAIN AND TEST DATA----------------------------------------------------------------------------------------------
# IMAGE PATH ON DISK FOR TRAIN AND TEST IMAGES
directory = '/home/rohan/Projects/EmbeddedMachineLearning/imgs/train'
test_directory = '/home/rohan/Projects/EmbeddedMachineLearning/imgs/test'
classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

t_ds = image_dataset_from_directory(directory,
                                  validation_split=0.2, label_mode='categorical', subset="training",
                                  image_size=(224,224), interpolation="mitchellcubic",
                                  crop_to_aspect_ratio=True, color_mode='rgb',
                                  seed=42, shuffle=True, batch_size=32)

v_ds = image_dataset_from_directory(directory,
                                  validation_split=0.2, label_mode='categorical', subset="validation",
                                  image_size=(224,224), interpolation="mitchellcubic",
                                  crop_to_aspect_ratio=True, color_mode='rgb',
                                  seed=42, shuffle=True, batch_size=32)

# Data preprocessing
train_ds = prepare(t_ds, True)
val_ds = prepare(v_ds, True)

num_classes = len(train_ds)
# MODEL ARCHITECTURE-------------------------------------------------------------------------------------------------
# Using MobileNetV1
mobilenetv1 = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top=False, weights=None, alpha=0.25, dropout=0.25)
mobilenetv1.trainable = True

inputs = keras.Input(shape=(224,224,3))
x = mobilenetv1(inputs)
x = keras.layers.GlobalMaxPooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv1_0.25_160_120")


# SUMMARY OF MODEL LAYERS AND PARAMETERS--------------------------------------------------------------------------------
model.summary()


# Define a learning rate schedule
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.96
)

# Create an optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# COMPILE MODEL---------------------------------------------------------------------------------------------------------
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics=['accuracy'])
#callbacks = [imports.EarlyStopping(monitor='val_accuracy', patience=5)]


# FIT MODEL WITH EPOCHS 12 and using CALLBACKS--------------------------------------------------------------------------
results = model.fit(train_ds, epochs=500, verbose=1,
                    validation_data=val_ds)



# summarize history for accuracy
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model
model.save("custom_NN_noweights")

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)

# int8 quantization
def representative_dataset():
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory, img))
        img_array = np.asarray(img_array).reshape((1, 160, 120, 3))
        yield [img_array.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
tflite_model_size = open('model_quant.tflite', "wb").write(tflite_quant_model)  
