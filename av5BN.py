import time,os
import tensorflow as tf
import numpy as np
from av3_input import launch_enqueue_workers


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

def conv_layer_withBN(layer_name, input_tensor, filter_size, strides=[1, 1, 1, 1, 1], padding='SAME'):
  """makes a simple convolutional layer"""
  input_depth = filter_size[3]
  output_depth = filter_size[4]
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      W_conv = weight_variable(filter_size)
      variable_summaries(W_conv, layer_name + '/weights')
    with tf.name_scope('biases'):
      b_conv = tf.zeros(output_depth)
      variable_summaries(b_conv, layer_name + '/biases')

    constant = tf.ones(output_depth)
    h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) 

    batch_mean, batch_variance = tf.nn.moments(h_conv,[0])
    h_conv_withBN = (constant * (h_conv - batch_mean) / tf.sqrt(batch_variance + 1e-8)) + b_conv

    tf.histogram_summary(layer_name + '/pooling_output', h_conv_withBN)
    print layer_name,"output dimensions:", h_conv_withBN.get_shape()
    return h_conv_withBN

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

def fc_layer_withBN(layer_name,input_tensor,output_dim):
  """makes a simple fully connected layer"""
  input_dim = int((input_tensor.get_shape())[1])

  with tf.name_scope(layer_name):
    weights = weight_variable([input_dim, output_dim])
    variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = tf.zeros(output_dim)
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_withBN'):
      h_fc = tf.matmul(input_tensor, weights) 

      constant = tf.ones(output_dim)
      batch_mean, batch_variance = tf.nn.moments(h_fc,[0])
      h_fc_withBN = (constant * (h_fc - batch_mean) / tf.sqrt(batch_variance + 1e-8)) + biases
      tf.histogram_summary(layer_name + '/fc_output', h_fc_withBN)
    
    print layer_name, "output dimensions:", h_fc.get_shape()
    return h_fc


def max_net(x_image_batch,keep_prob):
    "making a simple network that can receive 20x20x20 input images. And output 2 classes"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        print "image batch dimensions", x_image_batch.get_shape()
        # formally adding one depth dimension to the input
        x_image_with_depth = tf.reshape(x_image_batch, [-1, 40, 40, 40, 1])
        print "input to the first layer dimensions", x_image_with_depth.get_shape()

    h_conv1 = conv_layer_withBN(layer_name='conv1_5x5x5', input_tensor=x_image_with_depth, filter_size=[5, 5, 5, 1, 20])
    h_relu1 = relu_layer(layer_name='relu1', input_tensor=h_conv1)
    h_pool1 = pool_layer(layer_name='pool1_2x2x2', input_tensor=h_relu1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv2 = conv_layer_withBN(layer_name="conv2_3x3x3", input_tensor=h_pool1, filter_size=[3, 3, 3, 20, 30])
    h_relu2 = relu_layer(layer_name="relu2", input_tensor=h_conv2)
    h_pool2 = pool_layer(layer_name="pool2_2x2x2", input_tensor=h_relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv3 = conv_layer_withBN(layer_name="conv3_2x2x2", input_tensor=h_pool2, filter_size=[2, 2, 2, 30, 40])
    h_relu3 = relu_layer(layer_name="relu3", input_tensor=h_conv3)
    h_pool3 = pool_layer(layer_name="pool3_2x2x2", input_tensor=h_relu3, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv4 = conv_layer_withBN(layer_name="conv4_2x2x2", input_tensor=h_pool3, filter_size=[2, 2, 2, 40, 50])
    h_relu4 = relu_layer(layer_name="relu4", input_tensor=h_conv4)
    h_pool4 = pool_layer(layer_name="pool4_2x2x2", input_tensor=h_relu4, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv5 = conv_layer_withBN(layer_name="conv5_2x2x2", input_tensor=h_pool4, filter_size=[2, 2, 2, 50, 60])
    h_relu5 = relu_layer(layer_name="relu5", input_tensor=h_conv5)
    h_pool5 = pool_layer(layer_name="pool5_2x2x2", input_tensor=h_relu5, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    with tf.name_scope("flatten_layer"):
        h_pool2_flat = tf.reshape(h_pool5, [-1, 10 * 10 * 10 * 60])

    h_fc1 = fc_layer_withBN(layer_name="fc1", input_tensor=h_pool2_flat, output_dim=1024)
    h_fc1_relu = relu_layer(layer_name="fc1_relu", input_tensor=h_fc1)

    with tf.name_scope("dropout"):
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer_withBN(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer_withBN(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)

    return y_conv

def with_memory(logits, labels, func, prev_cost,alpha=0.125):
    unprocessed_cost = func(logits, labels)
    return (alpha*unprocessed_cost)+((1-alpha)*prev_cost)



"""def weighted_cross_entropy_mean_with_labels(logits,labels,pos_weight=1):
    computes weighted cross entropy mean for a multi class classification.
    Applies tf.nn.weighted_cross_entropy_with_logits
    accepts "labels" instead of "targets" as in
    tf.nn.sparse_softmax_cross_entropy_with_logits

    Note: classes are independent

    with tf.name_scope('weighted_cross_entropy_mean'):
        # convert labels to targets first
        # example for 3 classes
        # labels [1, 1, 1, 3, 2]
        # targets [1 0 0]
        # [1 0 0]
        # [1 0 0]
        # [0 0 1]
        # [0 1 0]

        batch_size = int(logits.get_shape()[0])
        num_classes = int(logits.get_shape()[1])
        labels = tf.cast(labels, dtype=tf.int32)
        indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
        sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size, dtype=tf.float32),shape=[batch_size, num_classes])
        targets = tf.sparse_tensor_to_dense(sparse_targets)

        # weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits,targets, pos_weight=pos_weight, name=None)
        weighted_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,targets)

        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy, name='cross_entropy')
        return cross_entropy_mean"""


"""def weighted_sparse_softmax_cross_entropy_with_logits(logits,labels,class_weights=[1,1]):


    # convert labels to targets first
    # example for 3 classes
    # labels [1, 1, 1, 3, 2]
    # targets [1 0 0]
    # [1 0 0]
    # [1 0 0]
    # [0 0 1]
    # [0 1 0]
    batch_size = int(logits.get_shape()[0])
    num_classes = int(logits.get_shape()[1])

    labels = tf.cast(labels,dtype=tf.int32)

    indices = tf.cast(tf.pack((tf.range(0,batch_size),labels),axis=1),dtype=tf.int64)

    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size,dtype=tf.float32),shape=[batch_size,num_classes])
    targets = tf.sparse_tensor_to_dense(sparse_targets)
    # now we have targets instead of labels

    # formula:
    # ent = targets * -log(softmax(logits)) = targets * -log(softmax(x))
    # ent = targets * -log(e**x/sum(e**x))
    # ent = targets * -(x - log(sum(e**x))
    # ent = targets * -(x - soft_maximum_x)

    # soft maximum formula is: soft_maximum_x = log(sum e**x))
    # is numerically unstable because: e**x easily goes to infinity and causes overflow
    # stable way to compute
    # if we shift by some constant K
    # log(e**x1 + e**x2 + e**x3...) = K + log((e**x1)/K + (e**x2)/K + (e**x3)/K)
    # K + log (e**(x1-K) + e**(x2-K) + e**(x3-K)+...)
    # now is we substitute K to the maximum of all terms
    # if K = max(x), there is no overflow since all (x-K) are negative
    # and e** of negative number is small and will not overflow
    # log(sum(e**x)) = max + log(e**(x1-max) + e**(x2-max) + e**..)

    # find maximum for every row and make it K
    max_logits = tf.reduce_max(logits, reduction_indices=1)
    # compute soft maximum
    soft_maximum_x = max_logits + tf.log(tf.reduce_sum(tf.exp(logits - tf.tile(tf.reshape(max_logits,shape=[batch_size,1]),multiples=[1,num_classes])),1))
    # compute simple cross entropy entropy
    simple_entropy = targets * -(logits - tf.tile(tf.reshape(soft_maximum_x,shape=[batch_size,1]),multiples=[1,num_classes]))
    # multiply one of the columns by a certain number to make misclassification of that class more expensive
    weighted_entropy_mean = tf.reduce_mean(class_weights * simple_entropy)

    return weighted_entropy_mean"""

'''def perceptron_loss(logits, targets):
    num_classes = int(logits.get_shape()[1])
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    min_cost = tf.argmin([cost, tf.nn.sparse_softmax_cross_entropy_with_logits(logits, -targets+tf.ones(num_classes))],0)
    return cost - min_cost'''

'''def multiclass_hinge_loss(logits, labels):
    max_logits = tf.argmax(logits, 1)
    batch_size = int(logits.get_shape()[0])
    num_classes = int(logits.get_shape()[1])
    labels = tf.cast(labels, dtype=tf.int32)
    indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size, dtype=tf.float32),shape=[batch_size, num_classes])
    targets = tf.sparse_tensor_to_dense(sparse_targets)
    return tf.reduce_mean(tf.maximum(0., tf.float(1 + max_logits - targets)))
    #tf.contrib.losses.smoothed_hinge_loss(y_conv, y_)
'''
def multiclass_rmse(logits, labels):
    labels = tf.cast(labels, dtype=tf.int32)
    batch_size = int(logits.get_shape()[0])
    num_classes = int(logits.get_shape()[1])
    indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size, dtype=tf.float32),shape=[batch_size, num_classes])
    targets = tf.sparse_tensor_to_dense(sparse_targets)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.reduce_sum(logits - targets, 1))))

def sigmoid_cross_entropy(logits, labels):
    labels = tf.cast(labels, dtype=tf.int32)
    batch_size = int(logits.get_shape()[0])
    num_classes = int(logits.get_shape()[1])
    indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size, dtype=tf.float32),shape=[batch_size, num_classes])
    targets = tf.sparse_tensor_to_dense(sparse_targets)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, targets))

def train():
    "train a network"

    # create session since everything is happening in one
    sess = tf.Session()
    train_image_queue,filename_coordinator = launch_enqueue_workers(sess=sess, pixel_size=FLAGS.pixel_size, side_pixels=FLAGS.side_pixels, num_workers=FLAGS.num_workers, batch_size=FLAGS.batch_size,
                                         database_index_file_path=FLAGS.train_set_file_path, num_epochs=FLAGS.num_epochs)

    y_, x_image_batch,_,_ = train_image_queue.dequeue_many(FLAGS.batch_size)
    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(x_image_batch, keep_prob)

    #Tensorflow
    #loss = sigmoid_cross_entropy(y_conv, y_)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y_)
    #loss = multiclass_hinge_loss(y_conv, y_)
    #loss = tf.contrib.losses.mean_squared_error(y_conv, y_)
    #loss = tf.Variable(0, dtype=tf.float32)
    #loss = with_memory(y_conv, y_, tf.nn.sparse_softmax_cross_entropy_with_logits, loss)
#tanh?
#add weights to each
#w memory
#change label
#clipping
#train with min of one cost function (at first?) then another?
    #loss = perceptron_loss(y_conv, y_)
    cross_entropy_mean = tf.reduce_mean(loss)

    with tf.name_scope('train'):
        tf.scalar_summary('weighted cross entropy mean', cross_entropy_mean)
        train_step_run = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope('evaluate_predictions'):
        # first: evaluate error when labels are randomly shuffled
        #  randomly shuffle along one of the dimensions:
        shuffled_y_ = tf.random_shuffle(y_)
        shuffled_cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,shuffled_y_))
	#shuffled_cross_entropy_mean = tf.Variable(0, dtype=tf.float32)
    	#shuffled_cross_entropy_mean  = with_memory(y_conv, y_, tf.nn.sparse_softmax_cross_entropy_with_logits, shuffled_cross_entropy_mean )

    # many small subroutines that are needed to save network state,logs, etc.
    if (FLAGS.saved_session !=0):
        # Note: new saved states go to a new folder for clarity
        # restore variables from sleep
	print " restoring variables from sleep"
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.saved_session)
    else:
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())




    # merge all summaries
    merged_summaries = tf.merge_all_summaries()
    # create a _log writer object
    train_writer = tf.train.SummaryWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)
    # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test' + str(FLAGS.run_index))


    # initialize all variables
    sess.run(tf.initialize_all_variables())

    batch_num = 0
    while not filename_coordinator.stop:
        start = time.time()
        training_error,_ = sess.run([cross_entropy_mean,train_step_run],feed_dict={keep_prob:0.5})

        print "step:", batch_num, "run error:", training_error,\
            "examples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        # once in a thousand batches calculate correct predictions
        if (batch_num % 1000 == 999):
            # evaluate and print a few things
            print "eval:-------------------------------------------------------------------------------------"
            shuffled_training_error,training_error,train_summary = sess.run([shuffled_cross_entropy_mean,cross_entropy_mean,merged_summaries],feed_dict={keep_prob:1})
            print "step:", batch_num, "run error:",training_error, "shuffled run error:", shuffled_training_error
            train_writer.add_summary(train_summary, batch_num)

            saver.save(sess,FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)

        # exit the loop in case there is something wrong with the setup and model diverged into inf
        assert not np.isnan(training_error), 'Model diverged with loss = NaN'
        batch_num+=1




class FLAGS:

    # important model parameters
    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 0.5
    # size of the box around the ligand in pixels
    side_pixels = 40
    # weights for each class for the scoring function
    # weights of [1 10] would mean that errors in a positive class are weighted 10 times more
    class_weights = [1,1]
    # number of times each example in the dataset will be read
    num_epochs = 20

    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 50
    # number of background processes to fill the queue with images
    num_workers = 32

    # data directories
    # path to the csv file with names of images selected for training
    train_set_file_path = '/home/ubuntu/common/data/kaggle/jan_01/labeled_npy/train_set.csv'
    # path to the csv file with names of the images selected for testing
    test_set_file_path = '/home/ubuntu/common/data/kaggle/dec_20/unlabeled_npy/database_index.csv'
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = 0#'./summaries/99_netstate/saved_state-15999'





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
