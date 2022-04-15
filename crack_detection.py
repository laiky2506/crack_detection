# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:16:11 2022

@author: laiky

Data Source: https://data.mendeley.com/datasets/5y9wdsg2zt/2
"""

# Import packages and set numpy random seed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

FILE_PATH = r'data'
DATA_DIR = Path(FILE_PATH)
BATCH_SIZE = 32
IMG_SIZE = (180,180)
SEED = 12345

train_dataset = tf.keras.utils.image_dataset_from_directory(DATA_DIR,validation_split=0.2,subset="training",seed=SEED,image_size=IMG_SIZE,batch_size=BATCH_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(DATA_DIR,validation_split=0.2,subset="validation",seed=SEED,image_size=IMG_SIZE,batch_size=BATCH_SIZE)

# Store labels of dataset
class_names = train_dataset.class_names

#%% Check image
# Print the first several training images, along with the labels
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
#%%
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

#%%
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#Image Augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip(),
  tf.keras.layers.RandomRotation(0.2)
])

#%%
#Show a sample of augmented image
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  ax = plt.subplot(3, 3, 1)
  plt.imshow(first_image / 255)
  plt.axis('off')
  for i in range(8):
    ax = plt.subplot(3, 3, i + 2)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
    
#%%
preprocess_input = tf.keras.applications.vgg19.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#%%
#Freeze the base model
base_model.trainable = False

#%%
# (a,b,c,d) = (Batch, wideth, height, channel)
base_model.summary()

#%%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layers = tf.keras.layers.Dense(len(class_names),activation='softmax')

#%%
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layers(x)

model = tf.keras.Model(inputs,outputs)
model.summary()
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
)

#%%
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

#%%
#model performance before training
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
#%%
log_path = r'C:\Users\laiky\Documents\TensorFlow\log\crackdetection_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(patience=3)
initial_epochs = 10
history = model.fit(train_dataset,epochs=initial_epochs, validation_data=validation_dataset, callbacks=[tb_callback,es_callback])

#%%
#Freeze the base model
base_model.trainable = False

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer = rmsprop, loss=loss, metrics=['accuracy'])
model.summary()

#%%
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history = model.fit(train_dataset,epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset, callbacks=[tb_callback])

#%%
#Deploy model to make prediction

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict(image_batch)
predictions = np.argmax(predictions, axis=1)

#compare predictions and labels
print(f'Prediction: {predictions}')
print(f'labels: {label_batch}')


#%%
plt.figure(figsize=(12,26))

for i in range(32):
    ax = plt.subplot(8,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"Actual: {class_names[label_batch[i]]} \n Prediction: {class_names[int(predictions[i])]}")
    plt.axis("off")