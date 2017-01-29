import tensorflow as tf
from av3 import *

def run(x_image_batch,keep_prob):
    "making a simple network"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        # reshaping 8000 vector into 20*20*20 image
        x_image = tf.reshape(x_image_batch, [-1, 20, 20, 20, 1])

    inp_tensor = x_image
    max_no_layers = 20
    no_layers = 0
    inp_size = 1
    out_size = 30
    current_matrix_edge_size = 20
    while no_layers < max_no_layers:
	h_conv = conv_layer(layer_name='conv'+str(no_layers)+'_2x2x2', input_tensor=inp_tensor, filter_size=[2, 2, 2, inp_size, out_size])
	inp_tensor = relu_layer(layer_name='relu'+str(no_layers), input_tensor=h_conv)
	if no_layers < 3:
		inp_tensor = pool_layer(layer_name='pool'+str(no_layers)+'_2x2x2', input_tensor=inp_tensor, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])
		current_matrix_edge_size = max(int(current_matrix_edge_size+1)/2, 1)
	inp_size = out_size
	out_size = out_size + 15
	no_layers += 1

    with tf.name_scope("flatten_layer"):
        h_pool4_flat = tf.reshape(inp_tensor, [-1, current_matrix_edge_size*current_matrix_edge_size*current_matrix_edge_size * inp_size])

    h_fc1 = fc_layer(layer_name="fc1", input_tensor=h_pool4_flat, output_dim=1024)
    h_fc1_relu = relu_layer(layer_name="fc1_relu", input_tensor=h_fc1)

    with tf.name_scope("dropout"):
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)

    return y_conv
