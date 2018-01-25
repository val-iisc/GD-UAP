# For classification

#download weights for vgg-f
wget https://www.dropbox.com/s/b96tfz2hnthfegm/vgg_f.npy

#download weights for caffenet
wget https://www.dropbox.com/s/8evo8ydl1jl03z4/caffenet.npy

#download weights for vgg-16
wget https://www.dropbox.com/s/zpeufcwesimhvua/vgg16.npy

#download weights for vgg-19
wget https://www.dropbox.com/s/e1v93adr4igwuct/vgg19.npy

#download weights for googlenet
wget https://www.dropbox.com/s/kzlgksuginkatb5/googlenet.npy

#download weights for resnet-50
wget https://www.dropbox.com/s/o8ay4gjdnu1ktds/resnet50.npy

#download weights for resnet-152
wget https://www.dropbox.com/s/8bzt5bfvkmz8xr9/resnet152.npy

mv *.npy classification/weights/

# For segmentation
wget https://www.dropbox.com/s/hjmdi9k3skyjfjb/additional_weights.tar.gz?dl=0 -O additional_weights.tar.gz
tar -xzf additional_weights.tar.gz
rm additional_weights.tar.gz
mv additional_weights/* segmentation/weights/
rm -rf additional_weights

# For depth estimation:

