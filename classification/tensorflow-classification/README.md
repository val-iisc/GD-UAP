# tensorflow-classification

Different neural network architechtures implemented in tensorflow for image classification. Weights converted from caffemodels. Some weights were converted using `misc/convert.py` others using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow). The weights can be downloaded from [here](https://www.dropbox.com/sh/qpuqj03gv00ba85/AAApqsIe4SqSOrsfpwrYjOema?dl=0). Tested with Tensorflow 1.0. Weights for inception-V3 taken from Keras implementation provided [here](https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py). Contributions are welcome!

## Features

* A single call program to classify images using different architechtures (vgg-f, caffenet, vgg-16, vgg-19, googlenet, resnet-50, resnet-152, inception-V3)
* Returns networks as a dictionary of layers, so accessing activations at intermediate layers is easy
* Functions to classify single image or evaluate on whole validation set

## Usage

* For classification of a single image, `python classify.py --network 'resnet152' --img_path 'misc/sample.jpg'`
* For evaluation over whole ilsvrc validation set `python classify.py --network 'resnet152' --img_list '<list with image names>' --gt_labels '<list with gt labels corresponding to images>'`
* Currently the `--network` argument can take vggf, caffenet, vgg16, vgg19, googlenet, resnet50, resnet152, inceptionv3.

## Performance
These converted models have the following performance on the ilsvrc validation set, with each image resized to 224x224 (227 or 299 depending on architechture), and per channel mean subtraction.


| Network        | Top-1 Accuracy           | Top-5 Accuracy  |
| ------------- |:-------------:| :-----:|
| VGG-F      | 58.33% | 80.75% |
| CaffeNet      | 56.77% | 79.98% |
| VGG-16      | 70.93%      |   89.82% |
| VGG-19      | 71.02%      |   89.85% |
| GoogLeNet | 68.69%      |    89.01% |
| ResNet-50 | 74.71% |    92.00% |
| ResNet-152 | 76.33% |    92.93% |
| Inception-V3 | 76.85% |    93.39% |
