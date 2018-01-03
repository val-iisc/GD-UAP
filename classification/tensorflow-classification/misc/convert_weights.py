#sample code for conversion of weights from caffe to tensorflow

import caffe
import numpy as np

net = caffe.Classifier('deploy.prototxt','VGG_CNN_F.caffemodel', caffe.TEST)

weights = {}

for i in net.params.keys():
    temp = []
    if i == 'fc6':
        t = np.reshape(net.params[i][0].data,[4096,256,6,6])
        t = np.transpose(t,[2,3,1,0])
        temp.append(np.reshape(t,[-1,4096]))
    elif 'fc' in i:
        temp.append(np.transpose(net.params[i][0].data,[1,0]))
    else:
        temp.append(np.transpose(net.params[i][0].data,[2,3,1,0]))
    temp.append(net.params[i][1].data)
    weights[i] = temp

np.save('vgg_f',weights)

