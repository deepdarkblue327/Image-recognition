#python
# coding: utf-8

### Required Imports ###
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import pandas as pd
import numpy as np
import os, time, glob

import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[23]:

print("Starting")
#Directory locations for the code
image_path = "Data\\img_align_celeba"

#Cropped Output
output_path = "Data\\cropped_faces"
if not os.path.exists(output_path):
        os.mkdir(output_path)
        
output_labels = "Data\\glasses.csv"

### place to save the model ###
MODEL_DIR = "tmp/mnist_convnet_model"


def import_data(start,end):
    #Eyeglasses label
    feature = pd.read_csv(output_labels)
    eye_glasses = feature[["Images","Eyeglasses"]]

    ### Changing -1 to 0 in the labels ###
    Y_labels = list(feature.Eyeglasses)
    Y_labels = np.array([i if i == 1 else 0 for i in Y_labels])[start:end]

    #Importing images as numpy objects
    SCALE = "L"
    X_REZ = 28
    Y_REZ = 28

    ### Import and resize image ###
    def resizer(path,scale="L",resize_x=28,resize_y=28):
        return np.array(Image.open(path).convert(scale).resize((resize_x,resize_y), Image.ANTIALIAS)).ravel().tolist()

    ### List of path to cropped images
    dirs_list = glob.glob(output_path+"\\*.jpg")[start:end]

    ### Import all images in a directory ###
    train_img = np.array([resizer(i,SCALE,X_REZ,Y_REZ) for i in dirs_list],dtype='float32')/255.0

    ### dictionary of images and their labels ###
    data = {}
    data["x"] = train_img
    data["y"] = Y_labels

    return data


# In[25]:


#### Range of images to be imported to train ####
data_length = 2000

data_start_index = 0
data_end_index = data_start_index + data_length


# In[34]:


### importing training data ###
data = import_data(data_start_index,data_end_index)
print(data)


# In[35]:


### Splitting the data into training and testing ###
train_data = {}
test_data = {}
train_data["x"], test_data["x"], train_data["y"], test_data["y"] = train_test_split(data["x"], data["y"], test_size=0.2)


# In[37]:


### Reference: https://www.tensorflow.org/tutorials/layers ###

### Model function that trains and evaluates the model ###
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, INPUT_LAYER_SHAPE_X, INPUT_LAYER_SHAPE_Y, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=LAYER_1_FILTERS,
      kernel_size=KERNEL_SIZE,
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=POOLING_SIZE, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=LAYER_2_FILTERS,
      kernel_size=KERNEL_SIZE,
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=POOLING_SIZE, strides=2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * LAYER_2_FILTERS])
    dense = tf.layers.dense(inputs=pool2_flat, units=DENSE_LAYER_UNITS, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=DROPOUT, training=mode == tf.estimator.ModeKeys.TRAIN)
 
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# In[41]:


##### HYPERPARAMETERS #####

INPUT_LAYER_SHAPE_X = 28
INPUT_LAYER_SHAPE_Y = 28
KERNEL_SIZE = [5, 5]
POOLING_SIZE = [2, 2]
LAYER_1_FILTERS = 32
LAYER_2_FILTERS = 64

DENSE_LAYER_UNITS = 1024
LEARNING_RATE = 0.001
DROPOUT = 0.4
BATCH_SIZE = 100

### Specifies the number of steps the model will take. Can exceed the number of images to train ###
TRAIN_STEPS = 200

### Specifies the number of runs through the training data ###
### None implies that the model will train till the number of steps specified ###
NUM_EPOCHS = None


# In[ ]:


mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)
tensors_to_log = {"probabilities": "softmax_tensor"}

### Logging to save progress. Checkpointing to restart from whenever necessary ###
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

### Feeding data to the model for running ###
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data["x"]},
    y=train_data["y"],
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    shuffle=True)

### Training starts ###
start_time = time.time()
mnist_classifier.train(input_fn=train_input_fn,steps=TRAIN_STEPS,hooks=[logging_hook])
print("Time taken to train =", float((time.time() - start_time)/60.0), "minutes")


# In[12]:


##### Training complete #####


# In[19]:


### Testing the trained model ###
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data["x"]},
    y=test_data["y"],
    num_epochs=10,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(" ")

##Printing accuracy
print("Accuracy", eval_results["accuracy"])

## printing loss
print("Loss",eval_results["loss"])

