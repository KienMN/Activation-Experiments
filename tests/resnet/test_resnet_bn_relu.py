import re
import os
import pandas as pd
import tensorflow as tf
from activation_layers.utils import insert_layer_nonseq
from tensorflow.keras.models import load_model
from activation_layers import DPReLU

# Directory
work_dir = os.path.abspath(os.path.dirname(__file__))

# Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
IMG_SHAPE = [32, 32, 3]
train_images = train_images.astype('float32')
train_labels = train_labels.ravel()
test_images = test_images.astype('float32')
test_labels = test_labels.ravel()

train_images /= 255.
test_images /= 255.

BATCH_SIZE = 128
EPOCHS = 200

# Prepare model
base_model = tf.keras.applications.ResNet50(
  input_shape=IMG_SHAPE,
  include_top=False,
  weights='imagenet',
  pooling='avg'
)
# base_model.summary()

print(base_model.summary())

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_images, test_labels))

metrics = {
  'loss': history.history['loss'],
  'val_loss': history.history['val_loss'],
  'accuracy': history.history['accuracy'],
  'val_accuracy': history.history['val_accuracy']
}

metrics = pd.DataFrame(metrics)
print(metrics)

# Save results
metrics.to_csv(work_dir + '/bn_relu_metrics.csv', index=False)
base_model.save(work_dir + '/resnet_bn_relu_on_cifar10.h5')