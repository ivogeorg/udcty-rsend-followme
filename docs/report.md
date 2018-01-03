# Udacity RSEND Follow Me Project Report

**Author:** Ivo Georgiev

**Date:** Jan 3, 2018

_Notes:_

_1. The notebook files are under `code`._

_2. The model and weights are under `data/weights`. The model is in HDF5 format even though it has no `.h5` extension._
  
_3. This report is under `docs`._

## Overview

The project involves training a model for the recognition of human targets from a (simulated) flying quadcopter. There is one target person who has to be recognized by themselves and in a crowd. The target's absence from the scene should also be learned.

The project uses a fully-convolutional network (FCN) to do semantic segmentation, thus providing pixel-level object recognition. 

## Model Architecture

The FCN is consists of an Encoder and a Decoder subnetworks, connected by a single 1-by-1 convolutional layer. Below are shown the shapes of the tensors flowing through the model.
```
inputs has shape (?, 128, 128, 3)
x1 has shape (?, 64, 64, 32)
x2 has shape (?, 32, 32, 64)
x3 has shape (?, 16, 16, 128)
x4 has shape (?, 8, 8, 256)
x11 has shape (?, 8, 8, 512)
x5 has shape (?, 16, 16, 256)
x6 has shape (?, 32, 32, 128)
x7 has shape (?, 64, 64, 64)
x8 has shape (?, 128, 128, 32)
```

### Encoder

The Encoder consists of 4 encoder layers, `x1`, `x2`, `x3`, and `x4`. The first layer starts with 32 filters, and they are doubled for each subsequent layer, including the 1-by-1 convolutional layer `x11`.

#### Encoder layer

An encoder layer consists of a single separable convolutional layer with batch regularization, kernel size 3, and stride of 2, SAME padding, and ReLU activation.

### Decoder

The Decoder consists of 4 decoder layers, `x5`, `x6`, `x7`, and `x8`. The number is meant to match the number of encoders. This symmetry helps proper segmentation.

Layers `x5` and `x6` receive skip connections from layers `x3` and `x2`, respectively, by concatenation. Concatenation requires _matching height and width_. Skip connections help the model retain more detail after downsampling and upsampling.

#### Decoder layer

The decoder layer consists of 4 layers:
  1. A bilinear upsampling layer with a factor of 2.
  2. (optional) A concatenation layer for skip connections.
  3. 2 separable convolutional layers for learning details from earlier layers after concatenation. _Note: The `filters` parameter of the

### Separable convolutions

Separable convolutional layers significantly reduce the number of model parameters, and thus make traing easier. They also provide some level of regularization, by virtue of reducing the number of parameters. In this respect, they act like _dropout_.

## Hyperparameters

The hyperparamers used are shown below.
```
learning_rate = 0.01
batch_size = 64
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

The _learning rate_ is relatively high but the model trained okay with it in only 20 epochs. As can be seen from the notebook loss curves, the model tended to _overfit_ on certain batches. Further experimentation could potentially get rid of this effect.

The _batch size_ was chosen to be low enough to run one epoch fast enough on a laptop. For comparable results and performance with AWS, it was retained for the live run.

The _number of epochs_ was chosen based on the one-epoch local run results. 

_Note: The model beat the targer 40% on the first run with these parameters. Exhaustive parameter space search was not required._

## Network Layers

### Fully connected layers

Fully connected layers are best used as the final 2-4 layers of a convolutional network which is tasked with classification. The network recognizes whether there is or isn't an object of particular type in a scene/image, but, having lost the spatial information in the fully-connected layers, it cannot localize it.

### 1-by-1 convolutions

As the title of the Follow Me project suggests, both object recognition and localization is required, so the quadcopter can use this information to properly navigate and follow the target. The usual CNN fully-connected layer is substituted with a 1-by-1 convolution, which preserves spatial information and passes it intact from the Encoder acorss to the Decoder. The Decoder can then use _upsampling_ and _skip connections_ to do pixel-level segmentation and recognition.

## Image Manipulation

An FCN requires accurate _ground truth_ to train, therefore the image sets for the project contain pairs of (image, mask). The mask provides pixel-level ground truth for each image, training the model to recognize its targets, _irrespective of their orientation, distance, pose, and overlap_.

The images are fed into the model as [batch_size, height, width, 3] tensors of pixel RGB values. The mask contains regions of 3 distinct values:
  * red for background (to be ignored, or masked out)
  * blue for the target person
  * green for confounders (objects of the same type as the target)
  
The set-based metric Intersection-over-Union (IoU) is particularly well suited for this kind of data, providing an accurate aggregate measure of misclassified pixels.

## Limitations

The model does worst (IoU = 0.1921) when it has to recognize the target from afar. Most of the distinctive characteristics of the target object fade out at a distnace. This task would require significantly more data to train on.

The model as it is right now is sufficient, given enough data, for recognizing a single type of object. A much deeper model might need to be created to do multi-class object recognition, with a correspondingly larger training set.

### Recognition of non-human targets

While the model is trained to recognize human (silhouettes) targets, there is no reason why it cannot be trained to recognize targets of other types (e.g. dogs, cars, etc.). There is nothing specific in the model for recognizing humans. 

Of course, some objects might be more difficult to learn to recognize and track than others. A small dog in a crowd would be **very hard**, while a mid-size icecream truck would be **easier**. However, the model will need to be trained on images that contain target objects and masks that provide the respective ground truth.

## Future Enhancements

The model is not making use of all the video data that can be collected and processed quickly. There are many more features that can be extracted for tracking of multi-class environments.

For one, the architecture could employ some recurrent connections to provide continuing across frames. This would significantly enhance the copter's ability to track its target, _even at a distance, in a crowd, or both_. 

Information about the attitude of the copter might also be used to help train the copter to track a target _even when its own motion is jerky_, for example if it has to veer away quickly to avoid an obstacle. This would enhance the real-world utility of a UAV as they would have to meet stringent requirements for safety.
