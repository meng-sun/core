import tensorflow as tf
import numpy as np

# TODO:
# generate one very large tensor
# take slices from that tensor
# use slices to find good affine transform
# generate a smaller tensor, use the whole one,
# regenerate it every epoch.

def random_transition_matrix():
    """returns a random transition matrix
    rotation range - determines random rotations along any of X,Y,Z axis
    shift_range determines allowed shifts along any of X,Y,Z axis """
    # shift range is hard coded to 10A because that's how the proteins look like
    # rotation range is hardcoded to 360 degrees

    shift_range = tf.constant(10,dtype=tf.float32) # FIXME
    rotation_range = tf.cast(tf.convert_to_tensor(np.pi*2),dtype=tf.float32)

    # randomly shift along X,Y,Z
    x_shift = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32)* shift_range
    y_shift = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32)* shift_range
    z_shift = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32)* shift_range

    # [1, 0, 0, random_x_shift],
    # [0, 1, 0, random_y_shift],
    # [0, 0, 1, random_z_shift],
    # [0, 0, 0, 1]])

    xyz_shift_matrix = tf.concat(0,[[tf.concat(0,[[1.0],[0.0],[0.0],[x_shift]])],
                         [tf.concat(0,[[0.0],[1.0],[0.0],[y_shift]])],
                         [tf.concat(0,[[0.0],[0.0],[1.0],[z_shift]])],
                         [tf.concat(0,[[0.0],[0.0],[0.0],[1.0]])]
                         ])


    # randomly rotate along X
    x_rot = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32, seed=None, name=None)*rotation_range


    # [[1, 0, 0, 0],
    # [0, cos(x_rot),-sin(x_rot),0],
    # [0, sin(x_rot),cos(x_rot),0],
    # [0, 0, 0, 1]],dtype=tf.float32)


    x_rot_matrix = tf.concat(0,[[tf.concat(0,[[1.0],[0.0],[0.0],[0.0]])],
                         [tf.concat(0,[[0.0],[tf.cos(x_rot)],[-tf.sin(x_rot)],[0.0]])],
                         [tf.concat(0,[[0.0],[tf.sin(x_rot)],[tf.cos(x_rot)],[0.0]])],
                         [tf.concat(0,[[0.0],[0.0],[0.0],[1.0]])]
                         ])


    # randomly rotate along Y
    y_rot = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32, seed=None, name=None) * rotation_range

    # [cos(y_rot), 0,sin(y_rot), 0],
    # [0, 1, 0, 0],
    # [-sin(y_rot), 0,cos(y_rot), 0],
    # [0, 0 ,0 ,1]])

    y_rot_matrix = tf.concat(0,[[tf.concat(0,[[tf.cos(y_rot)],[0.0],[tf.sin(y_rot)],[0.0]])],
                         [tf.concat(0,[[0.0],[1.0],[0.0],[0.0]])],
                         [tf.concat(0,[[-tf.sin(y_rot)],[0.0],[tf.cos(y_rot)],[0.0]])],
                         [tf.concat(0,[[0.0],[0.0],[0.0],[1.0]])]
                         ])

    z_rot = tf.random_uniform([], minval=-1, maxval=1, dtype=tf.float32, seed=None, name=None) * rotation_range


    # [[cos(z_rot), -sin(z_rot), 0, 0],
    # [sin(z_rot), cos(z_rot), 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]])


    z_rot_matrix = tf.concat(0,[[tf.concat(0,[[tf.cos(z_rot)],[-tf.sin(z_rot)],[0.0],[0.0]])],
                         [tf.concat(0,[[tf.sin(z_rot)],[tf.cos(z_rot)],[0.0],[0.0]])],
                         [tf.concat(0,[[0.0],[0.0],[1.0],[0.0]])],
                         [tf.concat(0,[[0.0],[0.0],[0.0],[1.0]])]
                         ])

    random_affine_transform_matrix = tf.matmul(tf.matmul(tf.matmul(xyz_shift_matrix,x_rot_matrix),y_rot_matrix),z_rot_matrix)


    return random_affine_transform_matrix


def affine_transform(coordinates,transition_matrix=random_transition_matrix()):
    """applies affine transform to the array of coordinates. By default generates a random affine transform matrix."""
    coordinates_with_ones = tf.concat(1, [coordinates, tf.cast(tf.ones([tf.shape(coordinates)[0],1]),tf.float32)])
    transformed_coords = tf.matmul(coordinates_with_ones,tf.transpose(transition_matrix))[0:,:-1]

    return transformed_coords,transition_matrix