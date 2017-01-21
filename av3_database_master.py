import numpy as np
import os,re, random,prody,time
from itertools import chain
from av2_atomdict import atom_dictionary

# database master
# 1) crawls directories and writes a human-readable database_index.csv file
# 2) splits the data into training and testing sets

# 3) parses proteins and writes useful data in them into .npy formatted arrays
# 3) later it's much faster to read from npy arrays

# todo there is a problem with multiframe PDBs that is supposed to be fixed with Xiao's update of the database
# todo some error handling and loggig
# todo add time.time
# FIXME write .npy files for pdb now

def preprocess_PDB_to_npy(database_path):
    """crawls the folder (receptors in this case) and saves every PDB it finds
    into .npy array with 1) coordinates 2) mapped to the atom name number """
    class statistics:
        docked_ligands_overlap = 0
        docked_ligands_not_overlap = 0

    def preprocess_ligands(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if re.search('.pdb$', filename):
                    ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                    try:
                        prody_ligand = prody.parsePDB(ligand_file_path)
                    except Exception:
                        pass

                    def ligand_atom_to_number(atomname):
                        atomic_tag_number = atom_dictionary.LIG[atomname.lower()]
                        return atomic_tag_number

                    try:

                        atom_numbers = map(ligand_atom_to_number, prody_ligand.getElements())
                        coordinates_and_atoms = np.hstack(
                            (prody_ligand.getCoords(), np.reshape(atom_numbers, (-1, 1))))
                        np.save(re.sub('.pdb$', '', ligand_file_path), coordinates_and_atoms)
                    except Exception:
                        pass


    def preprocess_receptors(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if re.search('.pdb$', filename):
                    receptor_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                    prody_receptor = prody.parsePDB(receptor_file_path)

                    def receptor_atom_to_number(atomname):
                        atomic_tag_number = atom_dictionary.REC[atomname.lower()]
                        return atomic_tag_number
                    try:
                        atom_numbers = map(receptor_atom_to_number, prody_receptor.getElements())
                        coordinates_and_atoms = np.hstack((prody_receptor.getCoords(), np.reshape(atom_numbers, (-1, 1))))
                        np.save(re.sub('.pdb$', '', receptor_file_path), coordinates_and_atoms)
                    except Exception:
                        pass

    preprocess_ligands(database_path + "/docked_ligands/")
    preprocess_ligands(database_path + "/crystal_ligands/")
    #preprocess_receptors(database_path + "/receptors/")




def assign_label_from_path(ligand_file_path):                                                                         # todo
    """assigns label to be used for training."""
    # 1 for the crystal structure
    # 0 for fast docked structure
    label = None
    if re.search('/crystal_ligands/', ligand_file_path):
        if label == None:
            label = 1
        else:
            raise Exception('can not assign two labels to one example:')

    if re.search('/docked_ligands/',ligand_file_path):
        if label == None:
            label = 0
        else:
           raise Exception('can not assign two labels to one example:')

    if re.search('/ligands/',ligand_file_path):
        if label == None:
            label = random.randint(0,1)
        else:
            raise Exception('can not assign two labels to one example:')

    if label == None:
        raise Exception("can not assign any labels to:",ligand_file_path)
    else:
        return label



def write_database_index_file(database_path,database_index_path,lig_dirs,label_gen_func=assign_label_from_path,rec_dir="receptors"):
    """crowls the database of the structure we are using and indexes it"""
    database_index_file = open(database_index_path + "/database_index.csv", "w")
    step = 0

    # walking across the ligand folders
    #for dirpath, dirnames, filenames in chain(os.walk(database_path + "/" + lig_dirs[0]),os.walk(database_path + "/" + lig_dirs[1])):
    for lig_dir in lig_dirs:
        for dirpath, dirnames, filenames in os.walk(database_index_path + "/" + lig_dir):

            for filename in filenames:
                # first look if what we found is a ligand
                pdb_name = dirpath.split("/")[-1]

                if re.search('.npy$', filename):
                    # second see if we can find a corresponding protein
                    ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                    label = label_gen_func(ligand_file_path)
                    if label is not None:
                        print "added to database", step, "label:", label
                        step += 1
                        receptor_file_path = os.path.abspath(str(database_path + "/" + rec_dir + "/" + pdb_name + ".npy"))

                        if os.path.exists(receptor_file_path):
                            database_index_file.write(
                                str(label) + "," + ligand_file_path + "," + receptor_file_path + "\n")
                        else:
                            print "WARNING: missing receptor"


    database_index_file.close()


def split_into_train_and_test_sets(database_index_path,train_set_div,replicate_positives=1):
    """splits the database textfile into training and testing sets.
    can replicate positive examples to do oversampling"""
    # create a unique pdb list, and randomly assign every pdb to the training or to the testing set
    database_index_lines = open(database_index_path + "/database_index.csv").readlines()
    def return_pdb(string):
        return string.split("/")[-1].strip("\n")

    pdb_names = np.array(map(return_pdb,database_index_lines))
    unique_pdb_names = np.unique(pdb_names)

    # now split according to the group
    test_set = []
    train_set = []


    def randomly_assign_and_retrieve_example(unique_pdb_name,pdb_names=pdb_names,train_set_div=train_set_div,database_index_lines=database_index_lines):
        # Given a pdb name randomly assign it either to the training or to the testing set keeping the probability.
        # Retrieve all of the examples with this pdb name from the database and write into training/testing file
        # TODO swap "random" to test only
        if (random.random() > train_set_div):
            for index in np.where(pdb_names == unique_pdb_name)[0]:
                test_set.append(database_index_lines[index])
        else:
            for index in np.where(pdb_names == unique_pdb_name)[0]:
                label = float(database_index_lines[index].split(",")[0])
                if (label == 1.0):
                    [train_set.append(database_index_lines[index]) for i in range(replicate_positives)]
                else:
                    train_set.append(database_index_lines[index])

    map(randomly_assign_and_retrieve_example,unique_pdb_names)

    if not (train_set_div==1):
        random.shuffle(test_set)
        open(database_index_path + "/test_set.csv", "w").writelines(test_set)
    if not (train_set_div==0):
        random.shuffle(train_set)
        open(database_index_path + "/train_set.csv", "w").writelines(train_set)


def docked_ligand_overlaps_with_crystal(database_path,docked_lig_path,tanimoto_cutoff=0.75,clash_cutoff_A=4,clash_size_cutoff=0.3):
    """This script takes docked ligand, and searches crystal ligands to find if doked position has a chance to be correct
    (overlaps). And in that case returns one."""

    def tanimoto_similarity(cryst_lig_path, docked_lig_path):
        command = os.popen('babel -d {} {} -ofpt -xfFP4'.format(cryst_lig_path, docked_lig_path))
        ls = command.read()
        tanimoto_similarity = re.split('=|\n', ls)[2]
        return tanimoto_similarity

    def calculate_clash(cryst_lig_path, docked_lig_path,clash_cutoff_A=clash_cutoff_A):
        """calculates the ratio of atoms between two molecules that appear within van Der Waals cutoff of 4A"""
        # calculate atom to atom distances
        prody_docked_ligand = prody.parsePDB(docked_lig_path).getCoords()
        prody_cryst_ligand = prody.parsePDB(cryst_lig_path).getCoords()

        def atom_clashing(index):
            # calculate distance to all atoms in the other molecule and check if any of them falls below cutoff
            if any(np.sum((prody_cryst_ligand[index,:] - prody_docked_ligand)**2,axis=1)**0.5 < clash_cutoff_A):
                return True
            else:
                return False

        clash_mask = map(atom_clashing,np.arange(len(prody_cryst_ligand[:, 0])))
        return np.average(clash_mask)

    clash_found = False
    # walk through /crystal/ligands and see if any of them are similar
    # there should be at least one
    for dirpath, dirnames, filenames in os.walk(database_path + "/crystal_ligands/" + docked_lig_path.split("/")[-2]):
        for filename in filenames:
            if re.search('.pdb$', filename):
                cryst_lig_path = str(os.path.abspath(dirpath) + "/" + filename)
                if tanimoto_similarity(cryst_lig_path,docked_lig_path) > tanimoto_cutoff:
                    # if ligand in crystal is similar to the docked ligand, calculate clash
                    if (calculate_clash(cryst_lig_path,docked_lig_path) > clash_size_cutoff):
                        clash_found = True
    return clash_found




def prepare_labeled_pdb():
    preprocess_PDB_to_npy(database_path='../datasets/test_hydro')
    write_database_index_file(database_path='../datasets/test_hydro',database_index_path='../datasets/test_hydro',lig_dirs=["crystal_ligands","docked_ligands"])
    split_into_train_and_test_sets(database_index_path='../datasets/test_hydro',train_set_div=0.95,replicate_positives=20)

def prepare_labeled_npy():
    write_database_index_file(database_path='../datasets/labeled_npy',database_index_path='../datasets/labeled_npy',lig_dirs=["crystal_ligands","docked_ligands"])
    split_into_train_and_test_sets(database_index_path='../datasets/labeled_npy',train_set_div=1,replicate_positives=20)

def prepare_unlabeled_npy():
    write_database_index_file(database_path='/home/ubuntu/common/data/kaggle/jan_01/unlabeled_npy',database_index_path='/home/ubuntu/common/data/kaggle/jan_01/unlabeled_npy',lig_dirs=["ligands"])
    split_into_train_and_test_sets(database_index_path='/home/ubuntu/common/data/kaggle/jan_01/unlabeled_npy',train_set_div=0)

#prepare_labeled_pdb()
#prepare_labeled_npy()
prepare_unlabeled_npy()

