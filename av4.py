import time,os
import tensorflow as tf
from av4_input import image_and_label_queue

# telling tensorflow how we want to randomly initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def conv_layer(layer_name, input_tensor, filter_size, strides=[1, 1, 1, 1, 1], padding='SAME'):
  """makes a simple convolutional layer"""
  input_depth = filter_size[3]
  output_depth = filter_size[4]
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      W_conv = weight_variable(filter_size)
      variable_summaries(W_conv, layer_name + '/weights')
    with tf.name_scope('biases'):
      b_conv = bias_variable([output_depth])
      variable_summaries(b_conv, layer_name + '/biases')
    h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) + b_conv
    tf.histogram_summary(layer_name + '/pooling_output', h_conv)
    print layer_name,"output dimensions:", h_conv.get_shape()
    return h_conv

def relu_layer(layer_name,input_tensor,act=tf.nn.relu):
  """makes a simple relu layer"""
  with tf.name_scope(layer_name):
    h_relu = act(input_tensor, name='activation')
    tf.histogram_summary(layer_name + '/relu_output', h_relu)
    tf.scalar_summary(layer_name + '/sparsity', tf.nn.zero_fraction(h_relu))

  print layer_name, "output dimensions:", h_relu.get_shape()
  return h_relu

def pool_layer(layer_name,input_tensor,ksize,strides=[1, 1, 1, 1, 1],padding='SAME'):
  """makes a simple pooling layer"""
  with tf.name_scope(layer_name):
    h_pool = tf.nn.max_pool3d(input_tensor,ksize=ksize,strides=strides,padding=padding)
    tf.histogram_summary(layer_name + '/pooling_output', h_pool)
    print layer_name, "output dimensions:", h_pool.get_shape()
    return h_pool

def fc_layer(layer_name,input_tensor,output_dim):
  """makes a simple fully connected layer"""
  input_dim = int((input_tensor.get_shape())[1])

  with tf.name_scope(layer_name):
    weights = weight_variable([input_dim, output_dim])
    variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      h_fc = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/fc_output', h_fc)
    print layer_name, "output dimensions:", h_fc.get_shape()
    return h_fc


def max_net(x_image_batch,keep_prob):
    "making a simple network that can receive 20x20x20 input images. And output 2 classes"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        print "image batch dimensions", x_image_batch.get_shape()
        # formally adding one depth dimension to the input
        x_image_with_depth = tf.reshape(x_image_batch, [-1, 20, 20, 20, 1])
        print "input to the first layer dimensions", x_image_with_depth.get_shape()

    h_conv1 = conv_layer(layer_name='conv1_5x5x5', input_tensor=x_image_with_depth, filter_size=[5, 5, 5, 1, 20])
    h_relu1 = relu_layer(layer_name='relu1', input_tensor=h_conv1)
    h_pool1 = pool_layer(layer_name='pool1_2x2x2', input_tensor=h_relu1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv2 = conv_layer(layer_name="conv2_3x3x3", input_tensor=h_pool1, filter_size=[3, 3, 3, 20, 30])
    h_relu2 = relu_layer(layer_name="relu2", input_tensor=h_conv2)
    h_pool2 = pool_layer(layer_name="pool2_2x2x2", input_tensor=h_relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv3 = conv_layer(layer_name="conv3_2x2x2", input_tensor=h_pool2, filter_size=[2, 2, 2, 30, 40])
    h_relu3 = relu_layer(layer_name="relu3", input_tensor=h_conv3)
    h_pool3 = pool_layer(layer_name="pool3_2x2x2", input_tensor=h_relu3, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv4 = conv_layer(layer_name="conv4_2x2x2", input_tensor=h_pool3, filter_size=[2, 2, 2, 40, 50])
    h_relu4 = relu_layer(layer_name="relu4", input_tensor=h_conv4)
    h_pool4 = pool_layer(layer_name="pool4_2x2x2", input_tensor=h_relu4, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv5 = conv_layer(layer_name="conv5_2x2x2", input_tensor=h_pool4, filter_size=[2, 2, 2, 50, 60])
    h_relu5 = relu_layer(layer_name="relu5", input_tensor=h_conv5)
    h_pool5 = pool_layer(layer_name="pool5_2x2x2", input_tensor=h_relu5, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    with tf.name_scope("flatten_layer"):
        h_pool2_flat = tf.reshape(h_pool5, [-1, 5 * 5 * 5 * 60])

    h_fc1 = fc_layer(layer_name="fc1", input_tensor=h_pool2_flat, output_dim=1024)
    h_fc1_relu = relu_layer(layer_name="fc1_relu", input_tensor=h_fc1)

    with tf.name_scope("dropout"):
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)

    return y_conv




def train():
    "train a network"

    # create session since everything is happening in one
    sess = tf.Session()
    # TODO: write atoms in layers of depth

    _,y_,x_image_batch = image_and_label_queue(sess=sess,batch_size=FLAGS.batch_size,
                                                pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                num_threads=FLAGS.num_threads,database_path=FLAGS.database_path)




    float_image_batch = tf.cast(x_image_batch,tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(float_image_batch,keep_prob)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        tf.scalar_summary('weighted cross entropy mean', cross_entropy_mean)
        train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # re-initialize all variables (two thread veriables were initialized before)
    sess.run(tf.initialize_all_variables())


    batch_num = 0
    while True:
        start = time.time()

        #sess.run([y_, x_image_batch], feed_dict={keep_prob: 0.5})
        training_error = sess.run([train_step_run], feed_dict={keep_prob: 0.5})
        print "training error:",training_error
        print "examples per second:", "%.2f" % (50 / (time.time() - start))

        batch_num+=1

class FLAGS:

    # important model parameters
    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 1
    # size of the box around the ligand in pixels
    side_pixels = 20
    # weights for each class for the scoring function
    # number of times each example in the dataset will be read
    num_epochs = 20
    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 50
    # number of background processes to fill the queue with images
    num_threads = 16
    # data directories
    # path to the csv file with names of images selected for training
<<<<<<< HEAD:av3.py
    train_set_file_path = '/pylon1/ci4s8bp/msun4/labeled_npy/train_set.csv'
    # path to the csv file with names of the images selected for testing
    test_set_file_path = '/pylon1/ci4s8bp/msun4/unlabeled_npy/database_index.csv'
=======
    database_path = "../datasets/labeled_pdb_av4/**/"
>>>>>>> 5f9b69cb057e1d72ea3d48d508833156857fd9cd:av4.py
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = 0





def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start"""
    summaries_dir = os.path.join(FLAGS.summaries_dir)
    # FLAGS.run_index defines when
    FLAGS.run_index = 1
    while ((tf.gfile.Exists(summaries_dir + "/"+ str(FLAGS.run_index) +'_train' ) or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_test' ))
           or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate') or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_logs')) and FLAGS.run_index < 1000:
        FLAGS.run_index += 1
    else:
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_train' )
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_test')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_netstate')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_logs')
    train()

if __name__ == '__main__':
    tf.app.run()
