import random,time,threading
import numpy as np
import tensorflow as tf


def generate_random_transition_matrix(shift_range=10,rotation_range=np.pi*2):
    """returns a random transition matrix
    rotation range - determines random rotations along any of X,Y,Z axis
    shift_range determines allowed shifts along any of X,Y,Z axis """
    # REMARK: shift range is hard coded to 10A because that's how the proteins look like

    # randomly shift along X,Y,Z
    xyz_shift_matrix = np.matrix([[1 ,0 , 0, random.random()*shift_range],
                                 [0, 1, 0, random.random()*shift_range],
                                 [0, 0, 1, random.random()*shift_range],
                                 [0, 0, 0, 1]])

    ## randomly rotate along X
    rand_x = random.random()*rotation_range
    x_rotate_matrix = np.matrix([[1, 0, 0, 0],
                               [0, np.cos(rand_x), - np.sin(rand_x), 0],
                               [0, np.sin(rand_x), np.cos(rand_x), 0],
                                [0, 0, 0, 1]])

    ## randomly rotate along YX
    rand_y = random.random()*rotation_range
    y_rotate_matrix = np.matrix([[np.cos(rand_y), 0, np.sin(rand_y), 0],
                                [0, 1, 0, 0],
                                [-np.sin(rand_y), 0, np.cos(rand_y), 0],
                                [0, 0 ,0 ,1]])

    ## randomly rotate along Z
    rand_z = random.random()*rotation_range
    z_rotate_matrix = np.matrix([[np.cos(rand_z), -np.sin(rand_z), 0, 0],
                               [np.sin(rand_z), np.cos(rand_z), 0, 0],
                               [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    # combining all matrices into one by multiplication
    # nice property of the matrix
    xyz_rotate_shift_matrix =  xyz_shift_matrix * x_rotate_matrix * y_rotate_matrix * z_rotate_matrix

    return xyz_rotate_shift_matrix




def affine_transform(coordinate_array,transition_matrix):
    """applies transition to every point of the array"""
    matrix_coord = np.matrix(coordinate_array)    # np.matrix has different defaults compared to np.array
    raw_of_ones = np.matrix(np.ones(len(matrix_coord[:,0])))
    transformed_coord = (transition_matrix * np.vstack((matrix_coord.transpose(),raw_of_ones)))[0:3,:].transpose()
    return np.array(transformed_coord)

#def affine_transform_old(coordinate_array,transition_matrix):
#    """applies transition to every point in array"""
#
#    def transform_one_point(one_point,transition_matrix=transition_matrix):
#        return np.sum(transition_matrix * np.matrix(np.append(one_point,1)).transpose(),axis=1)[0:3]
#
#    transformed_coordinates = np.array(map(transform_one_point,coordinate_array[:,0:3])).reshape(-1,3)
#    return transformed_coordinate


def launch_enqueue_workers(sess,pixel_size,side_pixels,num_workers,batch_size,database_index_file_path,num_epochs):
    """1) loads a whole database file into RAM.
       2) creates coordinator and picks protein-ligand-label trios in random order
       3) creates many workers that collectively pick data from coordinator
       4) each worker randomly rotates and shifts the box around the ligand center (data augmentation)
       5) each worker converts coordinates(sparse tensor) to image(dense tensor) and enqueues it"""
    class molecule_class():
        """simple class to store the information about the molecule"""
        coords = np.array([[]])
        atom_tags = np.array([[]])

        def load_npy(self, file_path):
            raw_array = np.load(file_path)
            self.coords = raw_array[:, 0:3]
            self.atom_tags = raw_array[:, 3]

    class filename_coordinator_class():
        """helps many background processes to read and enqueue images
        1) holds database index
        2) holds information about which images have been read"""
                                                                                    # TODO add randomization of readings
                                                                                    # TODO make it read an ~exact amount of entries

        labels = []
        ligand_filenames = []
        receptor_filenames = []
        # counter counts the index of the record of numpy array to return
        counter = 0
        index_file_size = 0
        # "stop" to stop the threads
        stop = False
        # number of epochs to train for
        num_epochs = 1
        # current epoch number
        epoch = 1
        # because multiple threads are reading from coordinator at the same time
        # it is possible that thread 2 will increment while thread 1 got value1[i],value2[i],but not value3[i]
        # thread 1 would read value1[i],value2[i],value3[i+1]
        lock = threading.Lock()

        def load_database_index_file(self, database_index_file_path):
            raw_database_array = np.genfromtxt(database_index_file_path, delimiter=',', dtype=str)
            self.labels = np.asarray(raw_database_array[:, 0], dtype=np.float32)
            self.ligand_filenames = raw_database_array[:, 1]
            self.receptor_filenames = raw_database_array[:, 2]
            self.index_file_size = len(self.labels)
            print "database index file has been loaded"

        def iterate_one(self,lock=lock):
            with lock:
                label = self.labels[self.counter]
                ligand_filename = self.ligand_filenames[self.counter]
                receptor_filename = self.receptor_filenames[self.counter]
                if (self.index_file_size - 1) > self.counter:
                    self.counter += 1
                else:
                    # when end of file list if reached see if it's the last epoch
                    if num_epochs > self.epoch:
                        self.epoch +=1
                        self.counter=0
                        print "training epoch:",self.epoch
                    else:
                        # stop the workers

                        # gracefully stop
                        self.stop = True
                        self.counter=0


            return label, ligand_filename, receptor_filename


    def if_in_the_box(atom_coord_array, box_edge=pixel_size * side_pixels):
        """returns a boolean array of size equal to height of atom_coord_array
        True - when in the box, False - when outside of the box"""

        pos_filter = -np.asarray(np.asarray((np.amax(atom_coord_array, axis=1)) / (0.5 * box_edge), dtype=int),
                                dtype=bool)
        neg_filter = -np.asarray(np.asarray((- np.amin(atom_coord_array, axis=1)) / (0.5 * box_edge), dtype=int),
                                dtype=bool)
        filter = pos_filter * neg_filter

        return filter


    # a small tensorgraph for image conversion
    # it's important that it stays outside of any loops and is only called once
    ceiled_coords = tf.placeholder(tf.int64, [None, 3])
    atom_tags = tf.placeholder(tf.float32)
    final_filter = tf.placeholder(tf.bool)

    sparse_prot = tf.SparseTensor(indices=ceiled_coords, values=atom_tags,
                                  shape=[side_pixels, side_pixels, side_pixels])
    filtered_prot = tf.sparse_retain(sparse_prot, final_filter)
    feed_image = tf.sparse_tensor_to_dense(filtered_prot, validate_indices=False)                    # TODO when two atoms go into one cell

    # label should be fed separately
    feed_label = tf.placeholder(tf.int64)

    # ligand and filename are fed for universality to be used in evaluations
    feed_ligand_filename = tf.placeholder(tf.string)
    feed_receptor_filename = tf.placeholder(tf.string)

    # a simple custom queue that will be used to store images
    image_queue = tf.FIFOQueue(100 + num_workers * batch_size, [tf.int64,tf.float32,tf.string,tf.string], shapes=[[],[side_pixels,side_pixels,side_pixels],[],[]])

    enqueue_image_op = image_queue.enqueue([feed_label,feed_image,feed_ligand_filename,feed_receptor_filename])


    def image_queue_worker(filename_coordinator,pixel_size=pixel_size, side_pixels=side_pixels,num_attempts=1000):
        """1) makes random affine transform matrix
           2) applies affine transform to the ligand
           3) checks if all atoms are in the box, if not, repeats (1)
           4) sends an image to the queue"""

        while True:

            label, ligand_filename, receptor_filename = filename_coordinator.iterate_one()

            # loading ligand molecule
            ligand_molecule = molecule_class()
            ligand_molecule.load_npy(ligand_filename)
            # shifting to the new center of mass
            ligand_center_of_mass = np.average(ligand_molecule.coords,axis=0)
            ligand_molecule.coords = ligand_molecule.coords - ligand_center_of_mass

            # find transition matrix that could fit all the ligand atoms
            final_transition_matrix = False
            for attempt in range(num_attempts):
                # make a random transition matrix
                random_transition_matrix = generate_random_transition_matrix()
                # affine transform
                transformed_ligand_coord = affine_transform(ligand_molecule.coords, random_transition_matrix)

                # see if all ligand atoms are in the box

                if all(if_in_the_box(transformed_ligand_coord)):
                    final_transition_matrix = random_transition_matrix
                    break

            # check if attempt to generate the transition matrix was successful
            # if not, exit procedure
            if isinstance(final_transition_matrix, bool):
                print "reached maximum number of attempts and failed", num_attempts                                     #TODO write this into log
                # "pass" rewinds loop to the beginning without enqueing image
                pass

            # if the script goes past this point it means that transition matrix was successfully generated
            # transform ligand to the new coordinate system (only do it once)
            ligand_molecule.coords = affine_transform(ligand_molecule.coords,final_transition_matrix)

            # parse the protein (receptor)
            receptor_molecule = molecule_class()
            receptor_molecule.load_npy(receptor_filename)
            # transform the protein in the same way as the ligand
            receptor_molecule.coords = receptor_molecule.coords - ligand_center_of_mass
            receptor_molecule.coords = affine_transform(receptor_molecule.coords,final_transition_matrix)
            # now merge the molecule coordinates and atom names
            molecule_complex = molecule_class()
            molecule_complex.coords = np.vstack((ligand_molecule.coords,receptor_molecule.coords))
            molecule_complex.atom_tags = np.hstack((ligand_molecule.atom_tags,receptor_molecule.atom_tags))

            # moving coordinates of a complex to an integer number so as to put every atom on a grid
            # ceiled coords is an integer number out of real coordinates that corresponds to the index on the cell
            molecule_complex.ceiled_coords = np.asarray(np.round(molecule_complex.coords / pixel_size - 0.5 + side_pixels / 2), dtype=int)
            molecule_complex.filter = if_in_the_box(atom_coord_array=molecule_complex.coords)

            sess.run([enqueue_image_op], feed_dict={feed_label:label, ceiled_coords: molecule_complex.ceiled_coords, atom_tags: molecule_complex.atom_tags,
                                     final_filter: molecule_complex.filter,feed_ligand_filename:ligand_filename,feed_receptor_filename:receptor_filename})



    # load the database index file
    filename_coordinator = filename_coordinator_class()
    filename_coordinator.load_database_index_file(database_index_file_path)
    filename_coordinator.num_epochs = num_epochs


    # launch many enqueue workers to fill the queue at the same time
    image_enqueue_threads = [threading.Thread(target=image_queue_worker,args=(filename_coordinator,)) for worker_name in xrange(num_workers)]
    for t in image_enqueue_threads:
        print "thread started:",t
        t.start()
        # sleep 0.3 seconds every time before launching next thread to avoid the situation
        # when too many threads are requesting memory access at the same time
        time.sleep(0.3)

    return image_queue,filename_coordinator
