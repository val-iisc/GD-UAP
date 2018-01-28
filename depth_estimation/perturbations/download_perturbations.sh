wget https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0 -O perturbations.tar.gz
tar -xzf perturbations.tar.gz
rm -rf perturbations.tar.gz
rm -rf perturbation_classification
rm -rf perturbation_segmentation
mv perturbation_depth/* ./
rm -rf perturbation_depth
# Now we need to create the zero_perturbation.npy
python get0.py
