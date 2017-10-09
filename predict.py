"""
Load trained AlexNet and predict the class scores for a given image.
"""
import os
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

#Class names from Imagenet
from imagenet_classes import class_names

test_id = 1
batch_size = 128

# Path to pretrained weights
alexnet_weights_path = '/home/meyn_ol/data/weights/bvlc_alexnet.npy'
# Path to a list of test images.
img_list_txt = '/home/meyn_ol/data/cache/img_list.txt'

num_classes = 1000

# Path for tf.summary.FileWriter
filewriter_root = '/tmp/alexnet/tensorboard'
filewriter_path = os.path.join(filewriter_root, 'run_' + str(test_id))

with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(img_list_txt,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)
    next_batch = iterator.get_next()
    
# Ops for initializing the two different iterators
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])

# Initialize model, with pretrained weights
model = AlexNet(x, num_classes, weights_path=alexnet_weights_path)

# Compute softmax probabilities from scores
score = model.fc8 # Top layer
softmax_probs = tf.nn.softmax(score)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Add the learned weights of conv1 to the summary
with tf.variable_scope('conv1', reuse=True):
    conv1_w = tf.get_variable('weights')

print(conv1_w.shape)
conv1_w = tf.transpose(conv1_w, perm=[3, 0, 1, 2], name='permutation')
print(conv1_w.shape)

#conv1_w_1,conv1_w_2 = tf.split(conv1_w, [1,95],3)
#print(conv1_w_1.shape)
#print(conv1_w_2.shape)
#conv1_w = tf.reshape(conv1_w,[-1,11,11,3]) 
#a0,a1,a2,a3,a4 conv1_w = tf.split(conv1_w, num_or_size_splits=5, axis=0)
conv_1_w_summary = tf.summary.image('conv1_weights',conv1_w, max_outputs=1000)

image_summary = tf.summary.image('input image', x,max_outputs=batch_size)

# Merge all summaries together
merged_summary_op = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Get the number of steps per epoch
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_root))

    # classify all images in the test dataset.
    print("{} Starting test".format(datetime.now()))
    sess.run(test_init_op)

    img_batch, label_batch = sess.run(next_batch)
    probs, acc = sess.run([softmax_probs, accuracy], feed_dict={x: img_batch,
                                        y: label_batch})

    #run the merged_summary_op for later visualization in TBoard
    merged_summary = sess.run(merged_summary_op, feed_dict={x: img_batch,
                                        y: label_batch})

    #Write the log files to disk
    # TODO
    # Is it possible to w/o merging the summary and call image_summary directly?
    writer.add_summary(merged_summary)

    # Print top 5 predicted categories of each sample.
    # Get idx with argsort and reverse order *column*-wise with [:,::-1]
    top_x = (np.argsort(probs)[:,::-1])[:,0:5]
    for row_idx, row in enumerate(top_x):
        el_names = [class_names[c] for c in row]
        el_probs = [probs[row_idx,c] for c in row]
        print('####\nTest sample {}\n####'.format(row_idx))
        print(list(zip(el_names,el_probs)))

    print("{} Test stopped. Accuracy: {:.4f}".format(datetime.now(), acc))

