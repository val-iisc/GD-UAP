# Semantic Segmentation Metrics on Pytorch


## Metrics used:

* Pixel Accuracy
* mean Accuracy(of per-class pixel accuracy)
* mean IOU(of per-class Mean IOU)
* Frequency weighted IOU

For more information, kindly refer [Fully Convolutional Networks for Semantic Segmentation
](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

## Functions

### Convert .mat files to .png files
Use 

` python demo.py convert_prediction predict_loc id_file`

### Convert Pascal VOC Validation filesfrom 3d-color to 2d-class-id .png format
Use

`python demo.py convert_gt gt_loc id_file`

### Calculate the metrics
Use

`python demo.py find_metrics predict_path gt_path id_file [--options]`

## Files

* `utils.py`: contains functions.
* `demo.py`: contains a brief demo of how to use the functions.[use demo.py -h]

## Acknowledgement


A few parts have been adopted from the code present in [martinkersner/py_img_seg_eval](https://github.com/martinkersner/py_img_seg_eval/tree/c0bf9787ebbe3e5e2c7833efe78b5b2d392afaf1). Although the formulations are slightly wrong, it was very helpful.

Also, a big thanks to [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/).

