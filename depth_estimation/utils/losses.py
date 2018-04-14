# definition of different custom loss functions
import tensorflow as tf

# to maximise activations


def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        # total blob activations
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j])))
            except:
                # total blob activations
                loss += tf.log(tf.reduce_mean(tf.abs(network[i])))
    return loss


def l2_outputs(layers):
    loss = 0
    for layer in layers:
        loss += tf.log(tf.nn.l2_loss(layer))
    return loss


def l2_all_resnet(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j in ['branch_1', 'branch_2', 'branch_3', 'branch_4']:
                        print(i, j)
                        loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss


def l2_all(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss


def l2_all_vgg(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                if not 'r' in i:
                    print i
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss


def l2_all_conv(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if not 'branch' in j:
                        print i, j
                        loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                if not 'relu' in i:
                    print i
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss
