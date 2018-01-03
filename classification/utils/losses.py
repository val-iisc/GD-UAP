#definition of different custom loss functions
import tensorflow as tf

# Loss as defined in Fast-feature-fool
def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j]))) #total blob activations
            except:
                loss += tf.log(tf.reduce_mean(tf.abs(network[i]))) #total blob activations
    return loss

# Loss as defined for DG_UAP
def l2_all(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i,j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j]))) 
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss

