import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Iterator

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16(object):
    """ VGG net"""

    def __init__(self, X):
        self.X = X
        build()

    def build(self):
        """Build the network architecture"""
        fc1 = fc_layer(self.X, "fc1", X.shape[1], 4096)
        fc2 = fc_layer(fc1, "fc2", 4096, 10,relu=False)
        #conv1_1 = conv(image,
        # ...

#    def load_weights():
#        """Load external weights from previous run"""


def fc_layer(x, name, input_size, layer_size, relu=True):
    with tf.variable_scope(name) as scope:
        #variable for storing the weights+biases of this layer.
        #'trainable' adds it to collection GraphKeys.TRAINABLE_VARIABLES
        weights = tf.get_variable('weights', shape=[input_size, layer_size], trainable=True)
        biases = tf.get_variable('biases', shape=[input_size], trainable=True)
        h = tf.nn.xw_plus_b(x, weights, biases)
        if relu == True:
            h = tf.nn.relu(h)
        return h


if __name__ == '__main__':
    train_img_list_txt = '/home/meyn_ol/data/cache/cl_0123_ps_64_dm_55_sam_799_0.txt'
    batch_size = 128

    #Why explicitly on CPU ???
    #with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_img_list_txt,
                                 mode='training',
                                 batch_size,
                                 num_classes,
                                 shuffle=True)

    #register the TF Iterator with the TF Dataset
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data.data)

    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])

    # continue here, now I need the tf.placeholder
    # Ok, why do I need dataset and iterator? is there no way plugging the data
    # directly into the placeholder? Is this really that slow? well, now go for the "optimal" dataset solution
    # First exclude VGG net all together and just run the input routine.
    # There is a MWE in ~/tmp/tf_include.py
    # But hurry, concetrate on the actual model and not on the input routine.
    #vgg = Vgg16()

