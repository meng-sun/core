import time,os
import tensorflow as tf
import numpy as np
from av4_input import image_and_label_queue

# telling tensorflow how we want to randomly initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """attaches a lot of summaries to a tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

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
            tf.summary.histogram(layer_name + '/pooling_output', h_conv)
    print layer_name,"output dimensions:", h_conv.get_shape()
    return h_conv

def relu_layer(layer_name,input_tensor,act=tf.nn.relu):
    """makes a simple relu layer"""
    with tf.name_scope(layer_name):
        h_relu = act(input_tensor, name='activation')
        tf.summary.histogram(layer_name + '/relu_output', h_relu)
        tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(h_relu))

    print layer_name, "output dimensions:", h_relu.get_shape()
    return h_relu

def pool_layer(layer_name,input_tensor,ksize,strides=[1, 1, 1, 1, 1],padding='SAME'):
    """makes a simple pooling layer"""
    with tf.name_scope(layer_name):
        h_pool = tf.nn.max_pool3d(input_tensor,ksize=ksize,strides=strides,padding=padding)
        tf.summary.histogram(layer_name + '/pooling_output', h_pool)
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
        tf.summary.histogram(layer_name + '/fc_output', h_fc)
    print layer_name, "output dimensions:", h_fc.get_shape()
    return h_fc


def max_net(x_image_batch,keep_prob):
    "makes a simple network that can receive 20x20x20 input images. And output 2 classes"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        print "image batch dimensions", x_image_batch.get_shape()
        # formally adding one depth dimension to the input
        x_image_with_depth = tf.reshape(x_image_batch, [-1, 20, 20, 20, 14])
        print "input to the first layer dimensions", x_image_with_depth.get_shape()

    h_conv1 = conv_layer(layer_name='conv1_5x5x5', input_tensor=x_image_with_depth, filter_size=[5, 5, 5, 14, 20])
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
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)

    return y_conv


def train():
    "train a network"
    # with the current setup all of the TF's operations are happening in one session
    sess = tf.Session()
    current_epoch,label_batch,image_batch = image_and_label_queue(sess=sess,batch_size=FLAGS.batch_size,
                                                pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                num_threads=FLAGS.num_threads,database_path=FLAGS.database_path,
                                                                  num_epochs=FLAGS.num_epochs)
    
    dense_image_batch = tf.sparse_tensor_to_dense(image_batch,validate_indices=False)
    
    # floating is temporary
    float_image_batch = tf.cast(dense_image_batch,tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    predicted_labels= max_net(float_image_batch,keep_prob)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_labels,labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy mean', cross_entropy_mean)

    # randomly shuffle along the batch dimension and calculate an error
    shuffled_labels = tf.random_shuffle(label_batch)
    shuffled_cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predicted_labels,shuffled_labels))
    tf.summary.scalar('shuffled cross entropy mean', shuffled_cross_entropy_mean)

    # Adam optimizer is a very heart of the network
    train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # merge all summaries and create a file writer object
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)
    
    # create saver to save and load the network state
    saver = tf.train.Saver()
    if not FLAGS.saved_session is None:
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess, FLAGS.saved_session)
    # initialize all variables (two thread variables should have been initialized in av4_input already)
    
    sess.run(tf.global_variables_initializer())

    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # TODO change after testing
    batch_num = 0
    while True:
        start = time.time()

        epo,c_entropy_mean,_ = sess.run([current_epoch[0],cross_entropy_mean,train_step_run], feed_dict={keep_prob: 0.5})
        print "epoch:",epo,"global step:", batch_num, "\tcross entropy mean:", c_entropy_mean
        #test = sess.run(train_step_run, feed_dict={keep_prob:0.5})
        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        if (batch_num % 100 == 99):
            # once in a while save the network state and write variable summaries to disk
            c_entropy_mean,sc_entropy_mean,summaries = sess.run(
                [cross_entropy_mean, shuffled_cross_entropy_mean, merged_summaries], feed_dict={keep_prob: 1})
            print "cross entropy mean:",c_entropy_mean, "shuffled cross entropy mean:", sc_entropy_mean
            train_writer.add_summary(summaries, batch_num)
            saver.save(sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)

        batch_num += 1
    assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'


class FLAGS:
    """important model parameters"""

    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 0.5
    # size of the box around the ligand in pixels
    side_pixels = 20
    # weights for each class for the scoring function
    # number of times each example in the dataset will be read
    num_epochs = 5000 # epochs are counted based on the number of the protein examples
    # usually the dataset would have multiples frames of ligand binding to the same protein
    # av4_input also has an oversampling algorithm.
    # Example: if the dataset has 50 frames with 0 labels and 1 frame with 1 label, and we want to run it for 50 epochs,
    # 50 * 2(oversampling) * 50(negative samples) = 50 * 100 = 5000
    num_classes = 2
    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 40
    # number of background processes to fill the queue with images
    num_threads = 512
    # data directories
    database_path = "/home/ubuntu/maksym/labeled_pdb_av4/**/"
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None


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
