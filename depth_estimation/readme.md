
![depth example](depth_example.png)

# GD-UAP for Depth Estimation

Code for the paper [Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations]()

Mopuri Konda Reddy,Aditya Ganeshan, R. Venkatesh Babu 

This section contains code to craft and evaluate GD-UAP on the task of depth-estimation using Tensorflow.

## Precomputed Perturbations

Perturbations crafted using the proposed algorithm are provided in this [link](https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0). After extracting them, and placing them in the respective folders (In each task), you can use the evaluation code provided in each task for evaluation.

## Usage Instructions

### 1) Preparation

For training and evaluation, the following steps must be followed: 

1) Download the KITTI dataset, for testing and training, and  PLACES-205 dataset for the validation. Kindly look [here](http://www.cvlibs.net/datasets/kitti/raw_data.php) and [here](http://places.csail.mit.edu/downloadData.html) for these datasets respectively.

2) The files for training images, testing images has been provided by [monodepth](https://github.com/mrharicot/monodepth), and are saved in `monodepth_files/filenames/`. For making a validation set using PLACES-205 dataset, the file `utils/places_205.txt` has been provided, which contains the list of Places-205 Files (Used for creating the validation dataset).

3) Use scripts `make_gaussian_noise.py`, `make_preprocessed_data.py` provided in `data/` to make the gaussian sample (for range prior training) and the validation dataset.

4) Use the script `weights/download_weights.sh` to download the weights for `Monodepth Resnet50 Eigen` and `Monodepth VGG Eigen`. 

5) ** For only evaluation: ** Follow steps 1,2 and 4. Download and save the precomputed perturbations from this [link](https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0), and save the depth-estimation perturbations in the `perturbations/` folder.

### 2) Training

To train a perturbation you can use the `train.py` script. For example to train/craft a perturbation for `Monodepth VGG Eigen`, with data priors, use the following command:

```
python train.py --encoder vgg --prior_type with_data --img_list monodepth_files/filenames/eigen_train_files.txt --batch_size 10 --checkpoint_file weights/model_eigen
```

This will run an optimization process proposed in the paper to craft a UAP for `Monodepth VGG Eigen` with data-priors. The resultant perturbation is saved in `perturbations/`.


### 3) Testing

Evaluating the performance of a perturbation on the KITTI Eigen split test dataset is a two step process. First, use the `get_outputs.py` script.This will save the disparity maps for normal and perturbed images as `.npy` files. For example, to get the output of the perturbation `perturbations/monodepth_vgg_with_data.npy` on `Monodepth VGG Eigen` architecture, use the following command:

```
# To save output of normal Input
python get_outputs.py --network dl_vgg16 --adv_im perturbations/dl_vgg_with_data.npy --img_list utils/pascal_test.txt --save_path output/ --gpu 0 
# To save output of Perturbed Input
python get_outputs.py --network dl_vgg16 --adv_im perturbations/dl_vgg_with_data.npy --img_list utils/pascal_test.txt --save_path output/ --gpu 0 
```

This command will save the predicted disparity maps for normal and perturbed images at `output/normal_predictions/` and `output/perturbed_predictions/` respectively.

To find the Fooling rate and other metrics from the output, we use modified version of scripts provided by [monodepth](https://github.com/mrharicot/monodepth). After the previous step, to find the various metrics of perturbed prediction w.r.t. normal prediction (which can then be used to find the fooling rate), use the `monodepth_files/eval_test.py` script as follows:

```
T.B.A.
```
 
To find the various metrics of perturbed prediction w.r.t. ground truth, use the `monodepth_files/evaluate_kitty.py` script as follows:


```
T.B.A.
```
 

## Results

These are some results taken from the paper:

### Quantitative Results

T.B.A.


## Acknowledgement

T.B.A.

## Notice

The code has been refactored from an earlier version and it still has to be thoroughly tested. This will be done soon.

