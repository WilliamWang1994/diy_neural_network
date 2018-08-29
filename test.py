import os
import numpy as np
import cv2
import numpy
import tensorflow as tf
# import tensorflow.contrib.slim
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("../../tf_Test/MNIST_data", one_hot=True


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1, name='variable')
    return tf.Variable(init)


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name='conv_op')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(x):
    return tf.layers.batch_normalization(x, training=False)


def network(x):
    w_conv1 = weight_variable([3, 3, 3, 1])
    b_conv1 = bias_variable([1])
    h_conv1 = conv_2d(x, w_conv1) + b_conv1
    y = tf.nn.relu(batch_norm(h_conv1))
    return y, h_conv1, [w_conv1, b_conv1]


def train():
    x = tf.placeholder(tf.float32, [2, 112, 112, 3], name="inputs")  # shape=[1, 28, 28, 1],
    test_img = numpy.zeros((2, 112, 112, 3))
    test_img[0, :] = cv2.resize(cv2.imread('4.bmp'), (112, 112))
    test_img[1, :] = cv2.resize(cv2.imread('15.bmp'), (112, 112))
    # test_img[0, :] = np.ones((5, 5, 3))
    y, h_conv1, variables = network(x)
    variables += tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
    bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
    variables += bn_moving_vars
    saver = tf.train.Saver(variables)  # var_list=tf.trainable_variables()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(y, feed_dict={x: test_img})
        checkpoint_path = os.path.join('test_ckpt/', "test.ckpt")
        saver.save(sess, checkpoint_path)


def mnist_network(x, keep_prob):
    def conv2d(con_x, W, padding):
        return tf.nn.conv2d(con_x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(pool_x):
        return tf.nn.max_pool(pool_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batch_norm(xx, name):
        return tf.layers.batch_normalization(xx, training=False, name=name)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weight_variable([7, 7, 1, 32], "conv_1")
    b_conv1 = bias_variable([32], "bias_1")
    h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1, "VALID") + b_conv1, "bn_1"))
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], "conv_2")
    b_conv2 = bias_variable([64], "bias_2")
    h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2, "SAME") + b_conv2, "bn_2"))
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 128], "conv_3")
    b_conv3 = bias_variable([128], "bias_3")
    h_conv3 = tf.nn.relu(batch_norm(conv2d(h_pool2, W_conv3, "SAME") + b_conv3, "bn_3"))
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([3, 3, 128, 128], "conv_4")
    b_conv4 = bias_variable([128], "bias_4")
    h_conv4 = tf.nn.relu(batch_norm(conv2d(h_pool3, W_conv4, "SAME") + b_conv4, "bn_4"))

    W_conv5 = weight_variable([3, 3, 128, 10], "conv_5")
    b_conv5 = bias_variable([10], "bias_5")
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, "VALID") + b_conv5)
    sq_out = tf.squeeze(h_conv5, [1, 2])
    y = tf.nn.softmax(sq_out)
    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_conv5, b_conv5]


def restore():
    x = tf.placeholder(tf.float32, [2, 112, 112, 3], name="inputs")
    test_array = np.zeros((2, 112, 112, 3))
    test_array[0, :] = np.array(cv2.resize(cv2.imread('15.bmp'), (112, 112)))
    test_array[1, :] = np.array(cv2.resize(cv2.imread('4.bmp'), (112, 112)))
    sess = tf.Session()
    y, h_conv1, variables = network(x)
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file('test_ckpt/test.ckpt', tensor_name='', all_tensors=True)
    saver = tf.train.Saver()
    saver.restore(sess, "test_ckpt/test.ckpt")
    y, conv = sess.run([y, h_conv1], feed_dict={x: test_array})
    print("---------------------------------------")
    return y
# train()
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file('test_ckpt/test.ckpt', tensor_name='', all_tensors=True)
# a = 1+1
# begin = time.time()
restore()
# print(time.time() - begin)