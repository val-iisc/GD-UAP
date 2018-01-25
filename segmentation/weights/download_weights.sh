# For Deeplab Multiscale Resnet-101. From https://github.com/isht7/pytorch-deeplab-resnet
# Download weight by visiting https://drive.google.com/open?id=0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg 
# For the other use the following link
wget https://www.dropbox.com/s/hjmdi9k3skyjfjb/additional_weights.tar.gz?dl=0 -O additional_weights.tar.gz
tar -xzf additional_weights.tar.gz
rm additional_weights.tar.gz
mv additional_weights/* ./
rm -rf additional_weights
# Save the weight files(.pth)  in this directory. 
