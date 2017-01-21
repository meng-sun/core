import tensorflow as tf
import numpy as np
import time

# generate one very large tensor
# take slices from that tensor
# use slices to find good affine transform
# generate a smaller tensor, use the whole one,
# regenerate it every epoch.

def generate_deep_affine_transform(num_frames):
    """Generates a very big batch of affine transform matrices in 3D. The first dimension is batch, the other two
    describe typical affine transform matrices. Deep affine transform can be generated once in the beginning
    of training, and later slices can be taken from it randomly to speed up the computation."""

    # shift range is hard coded to 10A because that's how the proteins look like
    # rotation range is hardcoded to 360 degrees

    shift_range = tf.constant(10, dtype=tf.float32)  # FIXME
    rotation_range = tf.cast(tf.convert_to_tensor(np.pi * 2), dtype=tf.float32)

    # randomly shift along X,Y,Z
    x_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range
    y_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range
    z_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range

    # [1, 0, 0, random_x_shift],
    # [0, 1, 0, random_y_shift],
    # [0, 0, 1, random_z_shift],
    # [0, 0, 0, 1]])

    # try to do the following:
    # generate nine tensors for each of them
    # concatenate and reshape sixteen tensors

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = x_shift

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = y_shift

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = z_shift

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    xyz_shift_stick = tf.pack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    xyz_shift_matrix = tf.transpose(tf.reshape(xyz_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along X
    x_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [[1, 0, 0, 0],
    # [0, cos(x_rot),-sin(x_rot),0],
    # [0, sin(x_rot),cos(x_rot),0],
    # [0, 0, 0, 1]],dtype=tf.float32)

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.cos(x_rot)
    afn1_2 = -tf.sin(x_rot)
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.sin(x_rot)
    afn2_2 = tf.cos(x_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    x_rot_stick = tf.pack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    x_rot_matrix = tf.transpose(tf.reshape(x_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Y
    y_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [cos(y_rot), 0,sin(y_rot), 0],
    # [0, 1, 0, 0],
    # [-sin(y_rot), 0,cos(y_rot), 0],
    # [0, 0 ,0 ,1]])

    afn0_0 = tf.cos(y_rot)
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.sin(y_rot)
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = -tf.sin(y_rot)
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.cos(y_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    y_rot_stick = tf.pack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    y_rot_matrix = tf.transpose(tf.reshape(y_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Z
    z_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [[cos(z_rot), -sin(z_rot), 0, 0],
    # [sin(z_rot), cos(z_rot), 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]])

    afn0_0 = tf.cos(z_rot)
    afn0_1 = -tf.sin(z_rot)
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.sin(z_rot)
    afn1_1 = tf.cos(z_rot)
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    z_rot_stick = tf.pack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    z_rot_matrix = tf.transpose(tf.reshape(z_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    xyz_shift_xyz_rot = tf.matmul(tf.matmul(tf.matmul(xyz_shift_matrix, x_rot_matrix), y_rot_matrix), z_rot_matrix)

    return xyz_shift_xyz_rot

def affine_transform(coordinates,transition_matrix):
    """applies affine transform to the array of coordinates. By default generates a random affine transform matrix."""
    coordinates_with_ones = tf.concat(1, [coordinates, tf.cast(tf.ones([tf.shape(coordinates)[0],1]),tf.float32)])
    transformed_coords = tf.matmul(coordinates_with_ones,tf.transpose(transition_matrix))[0:,:-1]

    return transformed_coords,transition_matrix



























"""
def random_transition_matrix():
    returns a random transition matrix
    rotation range - determines random rotations along any of X,Y,Z axis
    shift_range determines allowed shifts along any of X,Y,Z axis
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


# create a tensor of 1000 matrices, concatenate them all
# try to take a slice of 1 every time


idx = tf.random_uniform([], minval=0, maxval=100, dtype=tf.int32)
many_affine = tf.Variable(tf.pack([random_transition_matrix() for i in range(100)]))

sess = tf.Session()
#one_tensor = tf.gather(many_affine,idx)
#one_pix = tf.reduce_max(one_tensor)
#multithread_batch = tf.train.batch([one_pix],10,num_threads=1,capacity=40)
init_op = tf.initialize_all_variables()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord)
print sess.run(many_affine)

#while True:
#    start = time.time()
#    sess.run(multithread_batch) #sess.run(many_affine)
#    print "exps:", 10/(time.time()-start)

#
# print "done"

# 25/s per regular slice with one thread


def affine_transform(coordinates,transition_matrix=random_transition_matrix()):
    applies affine transform to the array of coordinates. By default generates a random affine transform matrix.
    coordinates_with_ones = tf.concat(1, [coordinates, tf.cast(tf.ones([tf.shape(coordinates)[0],1]),tf.float32)])
    transformed_coords = tf.matmul(coordinates_with_ones,tf.transpose(transition_matrix))[0:,:-1]

    return transformed_coords,transition_matrix"""