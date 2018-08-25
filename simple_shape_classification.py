import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


dir_path = 'E:\\datasets\\simple shapes\\'
shape_dict = {0: 'circle', 1: 'square', 2: 'star', 3: 'triangle'}

total_num = 3720
train_num = 3000
test_num = total_num - train_num

image_width = 100
image_height = 100

train_images = []
train_labels = []


def get_label_array(index):
    arr = [0, 0, 0, 0]
    arr[index] = 1
    return arr


def get_image_array(file_path):
    print(file_path)
    image = Image.open(file_path, 'r').resize((100, 100))
    img_arr = np.array(image)
    img_arr = img_arr.reshape([image_width * image_height])
    img_arr = np.where(img_arr == 0, 0, 1)
    return img_arr


def init_train_sets():
    for i in range(train_num):
        for shape_index in range(len(shape_dict)):
            train_labels.append(get_label_array(shape_index))
            train_images.append(get_image_array(dir_path + shape_dict[shape_index] + '\\' + str(i) + '.png'))


def add_layer(inputs, input_size, output_size, activation_fun=None):
    Weights = tf.Variable(tf.zeros([input_size, output_size]), name='W')
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.001, name='b')
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fun is None:
        output = Wx_plus_b
    else:
        output = activation_fun(Wx_plus_b)
    return output


init_train_sets()

x_inputs = tf.placeholder(tf.float32, [None, image_width * image_height])
y_inputs = tf.placeholder(tf.float32, [None, 4])

y = add_layer(x_inputs, image_width * image_height, 4, tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_inputs * tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train, feed_dict={x_inputs: train_images, y_inputs: train_labels})
        if i % 50 == 0:
            print(i, sess.run(cross_entropy, feed_dict={x_inputs: train_images, y_inputs: train_labels}))

    saver = tf.train.Saver(max_to_keep=4)
    saver.save(sess, 'model/simple_shape_classification_model')

    test_image_arr = []
    test_image_arr.append(get_image_array(dir_path + 'star\\3020.png'))
    test_image_arr.append(get_image_array(dir_path + 'square\\3020.png'))
    test_image_arr.append(get_image_array(dir_path + 'triangle\\3020.png'))
    test_image_arr.append(get_image_array(dir_path + 'circle\\3020.png'))
    test_image_arr.append(get_image_array(dir_path + 'star\\3021.png'))
    test_image_arr.append(get_image_array(dir_path + 'triangle\\3022.png'))
    test_image_arr.append(get_image_array(dir_path + 'circle\\3021.png'))
    test_image_arr.append(get_image_array(dir_path + 'star\\3022.png'))
    y_pre = sess.run(y, feed_dict={x_inputs: test_image_arr})
    pre_result = tf.argmax(y_pre, 1)
    print(sess.run(pre_result))

#   true result [2 1 3 0 2 3 0 2]

# with tf.Session() as sess:
#
#     saver = tf.train.import_meta_graph('model/simple_shape_classification_model.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('model/'))
#
#     test_image_arr = [get_image_array('E:\\datasets\\simple shapes\\square\\3020.png'),
#                       get_image_array('E:\\datasets\\simple shapes\\star\\3026.png'),
#                       get_image_array('E:\\datasets\\simple shapes\\triangle\\3020.png')]
#
#     y_pre = sess.run(y, feed_dict={x_inputs: test_image_arr})
#     pre_result = tf.argmax(y_pre, 1)
#     print(sess.run(pre_result))

