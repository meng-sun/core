import tensorflow as tf
from glob import glob
import os,time
from av4_utils import generate_deep_affine_transform,affine_transform

def image_walk(coords, elements, dense_shape):
    """returns a sparse tensor of shape [num, dense_shape[0], dense_shape[1]...] containing only the positions 
    corresponding to the occurrences of the atom num in elements at each num slice"""
    #dense_shape = [14,dense_shape[0],dense_shape[1],dense_shape[2]]
    #coords = tf.concat(1, [tf.reshape(tf.cast(elements-1, dtype=tf.int64), [-1,1]), coords])
    dense_shape = [dense_shape[0],dense_shape[1],dense_shape[2],14]
    coords = tf.concat(1, [coords,tf.reshape(tf.cast(elements-1, dtype=tf.int64), [-1,1])])
    #num_atoms = tf.shape(coords)[0]
    #coords_transpose = tf.transpose(coords)
    #_, ind_sort_z = tf.nn.top_k(-coords_transpose, k = num_atoms)
    #coords_sort_z = tf.gather(coords, ind_sort_z[3])
    #coords_transpose_z = tf.transpose(coords_sort_z)
    #_, ind_sort_y = tf.nn.top_k(-coords_transpose_z, k = num_atoms)
    #coords_sort_y = tf.gather(coords_sort_z, ind_sort_y[2])
    #coords_transpose_y = tf.transpose(coords_sort_y)
    #_, ind_sort_x = tf.nn.top_k(-coords_transpose_y, k=num_atoms)
    #coords_sort_x = tf.gather(coords_sort_y, ind_sort_x[1])
    #coords_transpose_x = tf.transpose(coords_sort_x)
    #_, ind_sort_elements = tf.nn.top_k(-coords_transpose_x, k=num_atoms)
    #coords_sort_elements = tf.gather(coords_sort_x, ind_sort_elements[0])
    #coords_transpose_elements = tf.transpose(coords_sort_elements)
    #sparse_tensors_by_element = tf.SparseTensor(indices=coords_sort_elements, values=coords_transpose_elements[0], shape=dense_shape)
    sparse_tensors_by_element = tf.SparseTensor(indices=coords, values=elements, shape=dense_shape)
    # TODO reshape by reference without having to write out explicitly 
    #sparse_tensors_by_element = tf.sparse_reshape(sparse_tensors_by_element17, [14,dense_shape[0], dense_shape[1], dense_shape[2]])
    return sparse_tensors_by_element

def index_the_database_into_queue(database_path,shuffle):
    """Indexes av4 database and returns two lists of filesystem path: ligand files, and protein files.
    Ligands are assumed to end with _ligand.av4, proteins should be in the same folders with ligands.
    Each protein should have its own folder named similarly to the protein name (in the PDB)."""
    # TODO controls epochs here
    ligand_file_list = []
    receptor_file_list = []
    for ligand_file in glob(os.path.join(database_path+'/**/', "*_ligand.av4")):
        receptor_file = "/".join(ligand_file.split("/")[:-1]) + "/" + ligand_file.split("/")[-1][:4] + '.av4'
        if os.path.exists(receptor_file):
            ligand_file_list.append(ligand_file)
            receptor_file_list.append(receptor_file)

    index_list = range(len(ligand_file_list))
    examples_in_database = len(index_list)

    if examples_in_database ==0:
        raise Exception('av4_input: No files found in the database path:',database_path)
    print "Indexed ligand-protein pairs in the database:",examples_in_database

    # create a filename queue (tensor) with the names of the ligand and receptors
    index_tensor = tf.convert_to_tensor(index_list,dtype=tf.int32)
    ligand_files = tf.convert_to_tensor(ligand_file_list,dtype=tf.string)
    receptor_files = tf.convert_to_tensor(receptor_file_list,dtype=tf.string)

    filename_queue = tf.train.slice_input_producer([index_tensor,ligand_files,receptor_files],num_epochs=None,shuffle=shuffle)
    return filename_queue,examples_in_database


def read_receptor_and_ligand(filename_queue,num_epochs,examples_in_database):
    """Reads ligand and protein raw bytes based on the names in the filename queue. Returns tensors with coordinates
    and atoms of ligand and protein for future processing.
    Important: by default it does oversampling of the positive examples based on training epoch."""

    def decode_av4(serialized_record):
        # decode everything into int32
        tmp_decoded_record = tf.decode_raw(serialized_record, tf.int32)
        # first four bytes describe the number of frames in a record
        number_of_frames = tf.slice(tmp_decoded_record, [0], [1])
        # labels are saved as int32 * number of frames in the record
        labels = tf.slice(tmp_decoded_record, [1], number_of_frames)
        # elements are saved as int32 and their number is == to the number of atoms
        number_of_atoms = ((tf.shape(tmp_decoded_record) - number_of_frames - 1) / (3 * number_of_frames + 1))
        elements = tf.slice(tmp_decoded_record, number_of_frames + 1, number_of_atoms)

        # coordinates are saved as a stack of X,Y,Z where the first(vertical) dimension
        # corresponds to the number of atoms
        # second (horizontal dimension) is x,y,z coordinate of every atom and is always 3
        # third (depth) dimension corresponds to the number of frames

        coords_shape = tf.concat(0, [number_of_atoms, [3], number_of_frames])
        tmp_coords = tf.slice(tmp_decoded_record, number_of_frames + number_of_atoms + 1,
                              tf.shape(tmp_decoded_record) - number_of_frames - number_of_atoms - 1)
        multiframe_coords = tf.bitcast(tf.reshape(tmp_coords, coords_shape), type=tf.float32)

        return labels,elements,multiframe_coords

    # read raw bytes of the ligand and receptor
    idx = filename_queue[0]
    ligand_file = filename_queue[1]
    serialized_ligand = tf.read_file(ligand_file)
    serialized_receptor = tf.read_file(filename_queue[2])
    
    # for epoch testing
    global epoch_counter
    # create an epoch counter
    examples_processed = tf.Variable(0,tf.int64)
    examples_processed = examples_processed.count_up_to(num_epochs*examples_in_database)
    epoch_counter = tf.div(examples_processed,examples_in_database)

    # decode bytes into meaningful tensors
    ligand_labels, ligand_elements, multiframe_ligand_coords = decode_av4(serialized_ligand)
    receptor_labels, receptor_elements, multiframe_receptor_coords = decode_av4(serialized_receptor)

    def count_frame_from_epoch(epoch_counter,ligand_labels):
        """Some simple arithmetics is used to sample all of the available frames
        if the index of the examle is even, positive label is taken every even epoch
        if the index of the example is odd, positive label is taken every odd epoch
        current negative example increments once every two epochs, and slides along all of the negative examples"""

        def select_pos_frame(): return tf.constant(0)
        def select_neg_frame(): return tf.mod(tf.div(1+epoch_counter,2), tf.shape(ligand_labels) - 1) +1
        current_frame = tf.cond(tf.equal(tf.mod(epoch_counter+idx+1,2),1),select_pos_frame,select_neg_frame)
        return current_frame

    current_frame = count_frame_from_epoch(epoch_counter,ligand_labels)
    ligand_coords = tf.gather(tf.transpose(multiframe_ligand_coords, perm=[2, 0, 1]),current_frame)
    label = tf.gather(ligand_labels,current_frame)

    return ligand_file,tf.squeeze(epoch_counter),tf.squeeze(label),ligand_elements,tf.squeeze(ligand_coords),receptor_elements,tf.squeeze(multiframe_receptor_coords)

def image_walk(coords, elements, dense_shape):
    """returns a sparse tensor of shape [num, dense_shape[0], dense_shape[1]...] containing only the positions 
    corresponding to the occurrences of the atom num in elements at each num slice"""
    # using the simpler method since atomdict has distinct values 0-7 inclusive right now
    
    if_merge_1 = tf.equal(1, elements) 
    partitioned_coords_1 = tf.boolean_mask(coords, if_merge_1)
    sparse_tensors_by_element1 = tf.SparseTensor(indices=partitioned_coords_1,values=tf.tile([1],[tf.shape(partitioned_coords_1)[0]]), shape=dense_shape)
    
    if_merge_2 = tf.equal(2, elements)
    partitioned_coords_2 = tf.boolean_mask(coords, if_merge_2)
    sparse_tensors_by_element2 = tf.sparse_concat(0, [sparse_tensors_by_element1,
    tf.SparseTensor(indices=partitioned_coords_2,values=tf.tile([2],
    [tf.shape(partitioned_coords_2)[0]]), shape=dense_shape)])
    
    if_merge_3 = tf.equal(3, elements)
    partitioned_coords_3 = tf.boolean_mask(coords, if_merge_3)
    sparse_tensors_by_element3 = tf.sparse_concat(0,[sparse_tensors_by_element2,
    tf.SparseTensor(indices=partitioned_coords_3,
    values=tf.tile([3],[tf.shape(partitioned_coords_3)[0]]), shape=dense_shape)])

    if_merge_4 = tf.equal(4, elements)
    partitioned_coords_4 = tf.boolean_mask(coords, if_merge_4)
    sparse_tensors_by_element4 = tf.sparse_concat(0,[sparse_tensors_by_element3,
    tf.SparseTensor(indices=partitioned_coords_4,
    values=tf.tile([4],[tf.shape(partitioned_coords_4)[0]]), shape=dense_shape)])

    if_merge_5 = tf.equal(5, elements)
    partitioned_coords_5 = tf.boolean_mask(coords, if_merge_5)
    sparse_tensors_by_element5 = tf.sparse_concat(0, [sparse_tensors_by_element4,
    tf.SparseTensor(indices=partitioned_coords_5,
    values=tf.tile([5],[tf.shape(partitioned_coords_5)[0]]), shape=dense_shape)])

    if_merge_6 = tf.equal(6, elements)
    partitioned_coords_6 = tf.boolean_mask(coords, if_merge_6)
    sparse_tensors_by_element6 = tf.sparse_concat(0,[sparse_tensors_by_element5,
    tf.SparseTensor(indices=partitioned_coords_6,
    values=tf.tile([6],[tf.shape(partitioned_coords_6)[0]]), shape=dense_shape)])

    if_merge_7 = tf.equal(7, elements)
    partitioned_coords_7 = tf.boolean_mask(coords, if_merge_7)
    sparse_tensors_by_element7 = tf.sparse_concat(0,[sparse_tensors_by_element6,
    tf.SparseTensor(indices=partitioned_coords_7,
    values=tf.tile([7],[tf.shape(partitioned_coords_7)[0]]), shape=dense_shape)])
    
    # TODO reshape by reference without having to write out explicitly 
    sparse_tensors_by_element = tf.sparse_reshape(sparse_tensors_by_element7, [7,dense_shape[0], dense_shape[1], dense_shape[2]])
    return sparse_tensors_by_element
     
def convert_protein_and_ligand_to_image(ligand_elements,ligand_coords,receptor_elements,receptor_coords,side_pixels,pixel_size):
    """Take coordinates and elements of protein and ligand and convert them into an image.
    Return image with one dimension so far."""

    # FIXME abandon ligand when it does not fit into the box (it's kept now)
    # TODO check if indeed it breaks in the last iteration cycle when a good affine transform is found
    # max_num_attempts - maximum number of affine transforms for the ligand to be tried
    max_num_attemts = 1000
    # affine_transform_pool_size is the first(batch) dimension of tensor of transition matrices to be returned
    # affine tranform pool is only generated once in the beginning of training and randomly sampled afterwards
    affine_transform_pool_size = 10000

    # transform center ligand around zero
    ligand_center_of_mass = tf.reduce_mean(ligand_coords, reduction_indices=0)
    centered_ligand_coords = ligand_coords - ligand_center_of_mass
    centered_receptor_coords = receptor_coords - ligand_center_of_mass

    # use TF while loop to find such an affine transform matrix that can fit the ligand so that no atoms are outside
    box_size = (tf.cast(side_pixels, tf.float32) * pixel_size)

    def generate_transition_matrix(attempt,transition_matrix,batch_of_transition_matrices):
        """Takes initial coordinates of the ligand, generates a random affine transform matrix and transforms coordinates."""
        transition_matrix= tf.gather(batch_of_transition_matrices,tf.random_uniform([], minval=0, maxval=affine_transform_pool_size, dtype=tf.int32))
        attempt += 1
        return attempt, transition_matrix,batch_of_transition_matrices

    def not_all_in_the_box(attempt, transition_matrix,batch_of_transition_matrices,ligand_coords=centered_ligand_coords,box_size=box_size,max_num_attempts=max_num_attemts):
        """Takes affine transform matrix and box dimensions, performs the transformation, and checks if all atoms
        are in the box."""
        transformed_coords, transition_matrix = affine_transform(ligand_coords, transition_matrix)
        not_all = tf.cast(tf.reduce_max(tf.cast(tf.square(box_size*0.5) - tf.square(transformed_coords) < 0,tf.int32)),tf.bool)
        within_iteration_limit = tf.cast(tf.reduce_sum(tf.cast(attempt < max_num_attemts, tf.float32)), tf.bool)
        return tf.logical_and(within_iteration_limit, not_all)


    attempt = tf.Variable(tf.constant(0, shape=[1]))
    batch_of_transition_matrices = tf.Variable(generate_deep_affine_transform(affine_transform_pool_size))
    transition_matrix = tf.gather(batch_of_transition_matrices, tf.random_uniform([], minval=0, maxval=affine_transform_pool_size, dtype=tf.int64))

    last_attempt,final_transition_matrix,_ = tf.while_loop(not_all_in_the_box, generate_transition_matrix, [attempt, transition_matrix,batch_of_transition_matrices],parallel_iterations=1)

    # rotate receptor and ligand using an affine transform matrix found
    rotatated_ligand_coords,_ = affine_transform(centered_ligand_coords,final_transition_matrix)
    rotated_receptor_coords,_ = affine_transform(centered_receptor_coords,final_transition_matrix)

    # check if all of the atoms are in the box, if not set the ligand to 0, but do not raise an error
    def set_elements_coords_zero(): return tf.constant([0],dtype=tf.int32),tf.constant([[0,0,0]],dtype=tf.float32)
    def keep_elements_coords(): return ligand_elements,rotatated_ligand_coords
    not_all = tf.cast(tf.reduce_max(tf.cast(tf.square(box_size * 0.5) - tf.square(rotatated_ligand_coords) < 0, tf.int32)),tf.bool)
    ligand_elements,rotatated_ligand_coords = tf.case({tf.equal(not_all,tf.constant(True)): set_elements_coords_zero},keep_elements_coords)

    # move coordinates of a complex to an integer number so as to put every atom on a grid
    # ceiled coords is an integer number out of real coordinates that corresponds to the index on the cell
    ceiled_ligand_coords = tf.cast(tf.round(-0.5 + (tf.cast(side_pixels,tf.float32)*0.5) + (rotatated_ligand_coords/pixel_size)),tf.int64)
    ceiled_receptor_coords = tf.cast(tf.round(-0.5 + (tf.cast(side_pixels, tf.float32) * 0.5) + (rotated_receptor_coords/pixel_size)),tf.int64)

    # crop atoms of the protein that do not fit inside the box
    top_filter = tf.reduce_max(ceiled_receptor_coords,reduction_indices=1)<side_pixels
    bottom_filter = tf.reduce_min(ceiled_receptor_coords,reduction_indices=1)>0
    retain_atoms = tf.logical_and(top_filter,bottom_filter)
    cropped_receptor_coords = tf.boolean_mask(ceiled_receptor_coords,retain_atoms)
    cropped_receptor_elements = tf.boolean_mask(receptor_elements,retain_atoms)

    # merge protein and ligand together. In this case an arbitrary value of 10 is added to the ligand
    complex_coords = tf.concat(0,[ceiled_ligand_coords,cropped_receptor_coords])

    complex_elements = tf.concat(0,[ligand_elements+7,cropped_receptor_elements])

    # in coordinates of a protein rounded to the nearest integer can be represented as indices of a sparse 3D tensor
    # values from the atom dictionary can be represented as values of a sparse tensor
    # in this case TF's sparse_tensor_to_dense can be used to generate an image out of rounded coordinates
    #sparse_complex = tf.SparseTensor(indices=complex_coords, values=complex_elements,shape=[side_pixels,side_pixels,side_pixels])
    #dense_complex = tf.sparse_tensor_to_dense(sparse_complex, validate_indices=False)
    sparse_complex = image_walk(complex_coords, complex_elements, [side_pixels, side_pixels, side_pixels])

    # FIXME: sparse_tensor_to_dense has not been properly tested.
    # FIXME: I may need to sort indices according to TF's manual on the function
    # FIXME: try to save an image and see how it looks like

    return sparse_complex,ligand_center_of_mass,final_transition_matrix


def image_and_label_shuffle_queue(sess,batch_size,pixel_size,side_pixels,num_threads,database_path,num_epochs):
    """Creates shuffle queue for training the network"""

    filename_queue,examples_in_database = index_the_database_into_queue(database_path,shuffle=True)

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    _,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,num_epochs,examples_in_database)

    # convert coordinates of ligand and protein into an image
    sparse_complex,_,_ = convert_protein_and_ligand_to_image(ligand_elements,ligand_coords,receptor_elements,receptor_coords,side_pixels,pixel_size)

    # selectively initialize some of the variables
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    sess.run(init_new_vars_op)

    # create a batch of proteins and ligands to read them together
    multithread_batch = tf.train.batch([current_epoch, label, sparse_complex], batch_size, num_threads=num_threads,capacity=batch_size * 3,
                                       dynamic_pad=True, shapes=[[], [], [None]])

    return multithread_batch


def single_dense_image_example(sess,batch_size,pixel_size,side_pixels,num_threads,database_path,num_epochs):
    """Reads the database and returns a single dense image example for visualization and other purposes."""

    filename_queue, examples_in_database = index_the_database_into_queue(database_path,shuffle=False)

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch #TODO read sequentially
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,num_epochs,index_list[-1])

    # convert coordinates of ligand and protein into an image
    sparse_complex,ligand_center_of_mass,final_transition_matrix = convert_protein_and_ligand_to_image(ligand_elements,ligand_coords,receptor_elements,receptor_coords,side_pixels,pixel_size)

    # selectively initialize some of the variables
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    sess.run(init_new_vars_op)

    # return all the information about a single image
    return ligand_file,ligand_center_of_mass,final_transition_matrix,current_epoch,label,sparse_complex

"""def image_and_label_stable_queue(sess,batch_size,pixel_size,side_pixels,num_threads,database_path,num_epochs):
    Creates shuffle queue for training the network


    # create a list of files in the database
    index_list,ligand_file_list,receptor_file_list = index_the_database(database_path)

    # create a filename queue (tensor) with the names of the ligand and receptors
    index_tensor = tf.convert_to_tensor(index_list,dtype=tf.int32)
    ligand_files = tf.convert_to_tensor(ligand_file_list,dtype=tf.string)
    receptor_files = tf.convert_to_tensor(receptor_file_list,dtype=tf.string)

    filename_queue = tf.train.slice_input_producer([index_tensor,ligand_files,receptor_files],num_epochs=None,shuffle=True)

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,num_epochs,index_list[-1])

    # convert coordinates of ligand and protein into an image
    dense_complex,ligand_center_of_mass,final_transition_matrix = convert_protein_and_ligand_to_image(ligand_elements,ligand_coords,receptor_elements,receptor_coords,side_pixels,pixel_size)

    # selectively initialize some of the variables
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    sess.run(tf.local_variables_initializer())
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    sess.run(init_new_vars_op)
    
    # create a batch of proteins and ligands to read them together
    multithread_batch = tf.train.batch([current_epoch, label, dense_complex], batch_size, num_threads=num_threads,capacity=batch_size * 3,shapes=[[], [], [side_pixels, side_pixels, side_pixels]])

    return multithread_batch"""
