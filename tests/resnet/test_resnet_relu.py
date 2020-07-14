import re
import os
import pandas as pd
import tensorflow as tf
from activation_layers.utils import insert_layer_nonseq
from tensorflow.keras.models import load_model
from activation_layers import DPReLU

# Directory
work_dir = os.path.abspath(os.path.dirname(__file__))
print(work_dir)

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
  pooling='avg',
  classes=10
)
# base_model.summary()

def normal_layer_factory():
  return tf.keras.layers.Layer(name='nl')

# Skip batch normalization layer
base_model = insert_layer_nonseq(base_model, '.*bn', normal_layer_factory, position='replace')
# Fix possible problems with new model
base_model.save(work_dir + '/temp1.h5')
base_model = load_model(work_dir + '/temp1.h5')

print(base_model.summary())

base_learning_rate = 0.0001
base_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

history = base_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_images, test_labels))

metrics = {
  'loss': history.history['loss'],
  'val_loss': history.history['val_loss'],
  'accuracy': history.history['accuracy'],
  'val_accuracy': history.history['val_accuracy']
}

metrics = pd.DataFrame(metrics)
print(metrics)

# Save results
metrics.to_csv(work_dir + '/relu_metrics.csv', index=False)
base_model.save(work_dir + '/resnet_relu_on_cifar10.h5')