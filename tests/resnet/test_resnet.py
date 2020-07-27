from __future__ import print_function, division, absolute_import

import os
import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from activation_layers import DPReLU, FReLU
from activation_layers.utils import insert_layer_nonseq
from tensorflow.keras.models import load_model

parser = ArgumentParser()
parser.add_argument('--activation', '-a', type=str, choices=['relu', 'prelu', 'dprelu', 'frelu'], help='Type of activation', required=True)
parser.add_argument('--batch_normalization', '-bn', type=bool, default=False, help='Whether to apply batch normalization or not')
parser.add_argument('--dataset', '-d', type=str, help='Name of dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs', default=50)
parser.add_argument('--batch_size', '-b', type=int, help='Batch size', default=32)
parser.add_argument('--optimizer', '-op', type=str, choices=['adam'], default='adam', help='Name of optimizer')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='Learning rate')

# Arguments
args = parser.parse_args()
print('Arguments', args)

# Directory
work_dir = os.path.abspath(os.path.dirname(__file__))
model_name = 'resnet50_{}_on_{}'.format('bn_' + args.activation if args.batch_normalization else args.activation, args.dataset)
print('Model name:', model_name)

# Load dataset
if args.dataset == 'cifar10':
  IMG_SHAPE = [32, 32, 3]
  N_CLASSES = 10
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_images = train_images.astype('float32')
  train_labels = train_labels.ravel()
  test_images = test_images.astype('float32')
  test_labels = test_labels.ravel()

  train_images /= 255.
  test_images /= 255.
  
elif args.dataset == 'cifar100':
  IMG_SHAPE = [32, 32, 3]
  N_CLASSES = 100
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
  train_images = train_images.astype('float32')
  train_labels = train_labels.ravel()
  test_images = test_images.astype('float32')
  test_labels = test_labels.ravel()

  train_images /= 255.
  test_images /= 255.

# Load model
def dprelu_layer_factory():
  return DPReLU(shared_axes=[1, 2, 3], name='dprelu')

def prelu_layer_factory():
  return tf.keras.layers.PReLU(shared_axes=[1, 2, 3], name='prelu')

def frelu_layer_factory():
  return FReLU(shared_axes=[1, 2, 3], name='frelu')

def normal_layer_factory():
  return tf.keras.layers.Layer(name='nl')

# Prepare the model
base_model = tf.keras.applications.ResNet50(
  input_shape=IMG_SHAPE,
  include_top=False,
  weights='imagenet',
  pooling='avg'
)

# Replace ReLU activation layer
if args.activation == 'dprelu':
  base_model = insert_layer_nonseq(base_model, '.*relu.*', dprelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

  base_model = insert_layer_nonseq(base_model, '.*out.*', dprelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

elif args.activation == 'prelu':
  base_model = insert_layer_nonseq(base_model, '.*relu.*', prelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

  base_model = insert_layer_nonseq(base_model, '.*out.*', prelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

elif args.activation == 'frelu':
  base_model = insert_layer_nonseq(base_model, '.*relu.*', frelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

  base_model = insert_layer_nonseq(base_model, '.*out.*', frelu_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

# Skip batch normalization layer
if not args.batch_normalization:
  base_model = insert_layer_nonseq(base_model, '.*bn', normal_layer_factory, position='replace')
  # Fix possible problems with new model
  base_model.save(work_dir + '/temp.h5')
  base_model = load_model(work_dir + '/temp.h5', custom_objects={'DPReLU': DPReLU})

print(base_model.summary())

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Dense(N_CLASSES)
])

print(model.summary())

# Hyper-parameters
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

# Optimizer
if args.optimizer == 'adam':
  optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

# Callbacks
model_filepath = work_dir+'/weights/'+model_name+'.{epoch:02d}-{val_accuracy:.4f}.hdf5'
callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=model_filepath,
  monitor='val_accuracy',
  mode='max',
  save_best_only=True
)
callbacks = [callback]

# Compile and train the model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_images, test_labels), callbacks=callbacks)

metrics = {
  'loss': history.history['loss'],
  'val_loss': history.history['val_loss'],
  'accuracy': history.history['accuracy'],
  'val_accuracy': history.history['val_accuracy']
}

metrics = pd.DataFrame(metrics)
print(metrics)

# Save results
metrics.to_csv(work_dir + '/' + model_name + '_metrics.csv', index=False)
model.save(work_dir + '/' + model_name + '.h5')