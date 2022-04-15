# Crack Detection with VGG19 model
Transfer Learning with VGG19 for Crack Detection

## Introduction
This is a project assigned during the Deep Learning with Python MUP-AI05 course, one of the course for AI05 bootcamp organized by Selangor Human Resource Development Centre (SHRDC) on 13 April 2022. The data used in this project is obtained from Concrete Crack Images for Classification (link: [https://data.mendeley.com/datasets/5y9wdsg2zt/2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)). The dataset contain 40,000 images seperated into 2 classes: Negative and Positive.

## Methodology
In this project, the neural network model is built with TensorFlow Keras functional API framework. Modules used in this project include:
* numpy
* pandas
* tensorflow
* matplotlib.pyplot
* pathlib.Path
* datetime

### STEP 1: Load and store of data
Images are load, resized to (180px,180px) and stored as variable with tensorflow.keras.utils.image_dataset_from_directory() method. The data are splitted into train and validation dataset.

### STEP 2: Preparation of data before training
The images are tuned with tensorflow.data.AUTOTUNE() method.

### STEP 3: Model Design
VGG19 is imported and used for the model to do transfer learning. The images are then augmented using RandomFlip and RandomRotate method from tensorflow.keras.layers before input into the model. The model trained for 10 epochs with a learning rate of 0.001 then fine tune with a learning rate of 0.00001 for another 10 epochs.

#### The Model flow diagram:
![Model!](/reference/model.png "Model")

### STEP 4: Evaluation of the model
The accuracy and loss of the model are as following diagram
![Epoch Accuracy!](/reference/epoch_accuracy.png "Epoch Accuracy")
![Epoch Loss!](/reference/epoch_loss.png "Epoch Loss")

## Result
#### The actual value of test data vs prediction value:
![Actual vs Prediction!](/reference/actual_vs_prediction.png "Actual vs Prediction")
