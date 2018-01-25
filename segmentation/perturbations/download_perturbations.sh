# Download perturbations from this link:
wget https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0
tar -xzf perturbations.tar.gz?dl=0
rm -rf perturbations.tar.gz?dl=0
rm -rf perturbation_classification
rm -rf perturbation_depth
mv perturbation_segmentation/* ./
rm -rf perturbation_segmentation
# Save the segmentation perturbation files in this directory.
