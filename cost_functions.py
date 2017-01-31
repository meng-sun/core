import tensorflow as tf
class STATIC:
    batch_size=100
    num_classes=2
#cost functions 

#also try different weights
#try with different labels
#try training with different orders of functions
def get_targets(labels):
    """helper function"""
    batch_size = STATIC.batch_size
    num_classes = STATIC.num_classes
    labels = tf.cast(labels, dtype=tf.int32)
    indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size, dtype=tf.float32),shape=[batch_size, num_classes])
    return tf.sparse_tensor_to_dense(sparse_targets)


def absolute_diff(logits,labels):
    return tf.contrib.losses.absolute_difference(logits, get_targets(labels))
    # may not converge?


def hinge_loss(logits,labels):
    return tf.contrib.losses.hinge_loss(logits, get_targets(labels))


def mce(logits, labels):
    flat_labels = tf.reshape(get_targets(labels),[-1])
    flat_logits = tf.reshape(logits,[-1])
    if_true = tf.equal(1., flat_labels)
    true_logits = tf.boolean_mask(flat_logits,if_true)
    if_false = tf.equal(0., flat_labels) # TODO a better way of doing this
    false_logits = tf.boolean_mask(flat_logits, if_false)
    return tf.pow(1+tf.exp(false_logits-true_logits),-1)
    # essentially just 1-softmax


def lvq2(logits,labels, min_cost):
    cost = hinge_loss(logits, labels)
    return tf.minimum(tf.constant(min_cost,dtype=tf.float32), cost)


def square_square(logits,labels):
    flat_labels = tf.reshape(get_targets(labels),[-1])
    flat_logits = tf.reshape(logits,[-1])
    if_true = tf.equal(1., flat_labels)
    true_logits = tf.boolean_mask(flat_logits,if_true)
    if_false = tf.equal(0., flat_labels) # TODO a better way of doing this
    false_logits = tf.boolean_mask(flat_logits, if_false)
    return tf.pow(true_logits,2)-tf.pow(tf.maximum(tf.zeros(STATIC.batch_size), 1-false_logits),2)


def square_exp(logits,labels,alpha=0.5):
    flat_labels = tf.reshape(get_targets(labels),[-1])
    flat_logits = tf.reshape(logits,[-1])
    if_true = tf.equal(1., flat_labels)
    true_logits = tf.boolean_mask(flat_logits,if_true)
    if_false = tf.equal(0., flat_labels) # TODO a better way of doing this
    false_logits = tf.boolean_mask(flat_logits, if_false)
    return tf.pow(true_logits,2)+alpha*(tf.exp(-false_logits))


def nll(logits, labels):
    return 0    


def mee(logits,labels):
    return 0


def modified_cosine_difference(logits, labels, alpha=0.1):
    #otherwise with regular equation cosine_difference is too similar to MCE
    targets = get_targets(labels)
    targets = ((1-alpha)*targets) + alpha*tf.ones(STATIC.num_classes)

    targets_magnitude = tf.sqrt(tf.reduce_sum(tf.pow(targets,2),1))
    logits_magnitude = tf.sqrt(tf.reduce_sum(tf.pow(logits,2),1))
    total_magnitude = targets_magnitude*logits_magnitude
    return (tf.reduce_sum(tf.multiply(logits, targets),1))/total_magnitude


def softmax_cross_entropy(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)


#same as cross entropy for us
def log_loss_explicit(logits,labels):
    batch_size = STATIC.batch_size
    num_classes = STATIC.num_classes
    labels = tf.cast(labels, dtype=tf.int32)
    indices = tf.cast(tf.pack((tf.range(0, batch_size), labels), axis=1), dtype=tf.int64)
    sparse_logits_correct_class = tf.SparseTensor(indices=indices, values=logits,shape=[batch_size, num_classes])
    logits_correct_class = tf.reduce_sum(tf.sparse_tensor_to_dense(sparse_logits_correct_class),1)
    opp_indices = - indices + [1,1]
    sparse_logits_incorrect_class = tf.SparseTensor(indices=opp_indices, values=logits, shape=[batch_size, num_classes])
    logits_incorrect_class = tf.reduce_sum(tf.sparse_tensor_to_dense(sparse_logits_incorrect_class),1)
    return tf.log(1+tf.reduce_sum(tf.exp(logits_correct_class - logits_incorrect_class),axis=1))


def log_loss(logits,labels):
    return tf.contrib.losses.log_loss(logits, get_targets(labels))


def with_memory(parameters_array, func, prev_cost, alpha=0.125):
    '''prev_cost should be initialized to tf.Variable(0)'''
    unprocessed_cost = func(parameters_array)
    return (alpha*unprocessed_cost)+((1-alpha)*prev_cost)


def rmse(logits, labels):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.reduce_sum(logits - get_targets(labels), 1))))


def mse(logits,labels):
    return tf.contrib.losses.mean_pairwise_squared_error(logits,get_targets(labels))
    # not a good idea?


def perceptron_loss(logits, labels):
    return 0.5*tf.pow(tf.reduce_sum(logits-get_targets(labels),1),2)
    # outdated?

#unfinished
def difference_cost(parameters_array, parameters_array2, func1, func2):
    return 0


def hinge_loss_explicit(logits, labels):
    targets = get_targets(labels)
    all_ones = tf.ones([batch_size, num_classes], dtype=tf.float32)
    targets = (2*targets) - all_ones
    return tf.maximum(tf.zeros([batch_size,num_classes]), all_ones-0.5*(targets*logits))


def sigmoid_cross_entropy(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, get_targets(labels))
