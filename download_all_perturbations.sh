wget https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0 -O perturbations.tar.gz
tar -xzf perturbations.tar.gz
rm -rf perturbations.tar.gz
mv perturbation_segmentation/* segmentation/perturbations/
mv perturbation_classification/* classification/perturbations/
mv perturbation_depth/* depth_estimation/perturbations/
rm -rf perturbation_depth
rm -rf perturbation_segmentation
rm -rf perturbation_classification

