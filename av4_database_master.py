import tensorflow as tf
import numpy as np
import time,os,re,prody
from av4_atomdict import atom_dictionary

class stats:
    start = time.time()
    ligands_parsed = 0
    ligands_failed = 0

# crawls the database with crystal ligands
# looks which of them have docked ligands - if can acquire (?) and have pdb(?) - if can acquire
# moves all of them in a single folder with protein together
# writes statistics
# TODO make exceptions more informative

def convert_database_to_av4(database_path,positives_folder,decoys_folder,receptors_folder):
    """Crawls the folder (receptors in this case) and saves every PDB it finds
    into .npy array with 1) coordinates 2) mapped to the atom name number """

    # make a directory where the av4 form of the output will be written
    output_path = str(database_path+'_av4')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save_av4(filepath,labels,elements,multiframe_coords):
        labels = np.asarray(labels,dtype=np.int32)
        elements = np.asarray(elements,dtype=np.int32)
        multiframe_coords = np.asarray(multiframe_coords,dtype=np.float32)

        if not (int(len(multiframe_coords[:,0]) == int(len(elements)))):
            raise Exception('Number of atom elements is not equal to the number of coordinates')

        if multiframe_coords.ndim==2:
            if not int(len(labels))==1:
                raise Exception ('Number labels is not equal to the number of coordinate frames')
        else:
            if not (int(len(multiframe_coords[0, 0, :]) == int(len(labels)))):
                raise Exception('Number labels is not equal to the number of coordinate frames')

        number_of_examples = np.array([len(labels)], dtype=np.int32)
        av4_record = number_of_examples.tobytes()
        av4_record += labels.tobytes()
        av4_record += elements.tobytes()
        av4_record += multiframe_coords.tobytes()
        f = open(filepath + ".av4", 'w')
        f.write(av4_record)
        f.close()


    for dirpath,dirnames,filenames in os.walk(database_path +"/"+ positives_folder):
        for filename in filenames:
            if re.search('.pdb$', filename):
                path_to_positive = str(os.path.abspath(dirpath) + "/" + filename)
                
                # if output file already generated, skip the conversion
                path_to_pdb_subfolder = output_path + "/" + str(os.path.abspath(dirpath)).split("/")[-1]
                ligand_output_file = path_to_pdb_subfolder + "/" + path_to_positive.split("/")[-1].split(".")[0]+'.av4'
                if os.path.exists(ligand_output_file):
                    continue

                # find path to folder with decoys
                path_to_decoys = re.sub(positives_folder,decoys_folder,str(os.path.abspath(dirpath)))
                # find path to the file with receptor
                path_to_receptor = re.sub(positives_folder,receptors_folder,str(os.path.abspath(dirpath))+'.pdb')

                try:
                    prody_receptor = prody.parsePDB(path_to_receptor)
                    prody_positive = prody.parsePDB(path_to_positive)

                    # ligand_cords will store multiple frames
                    multiframe_ligand_coords = prody_positive.getCoords()
                    labels = np.array([1])

                    # find all decoys
                    if os.path.exists(path_to_decoys):
                        for decoyname in os.listdir(path_to_decoys):
                            if re.match(re.sub('.pdb', '', path_to_positive.split("/")[-1]),re.sub('.pdb', '', decoyname)):
                                prody_decoy = prody.parsePDB(path_to_decoys + "/" + decoyname)

                                # see if decoy is same as the initial ligand
                                if not all(np.asarray(prody_decoy.getElements()) == np.asarray(prody_positive.getElements())):
                                    raise Exception('attempting to add ligand with different order of atoms')

                                multiframe_ligand_coords = np.dstack((multiframe_ligand_coords,prody_decoy.getCoords()))


                                labels = np.concatenate((labels,[0]))


                    if multiframe_ligand_coords.ndim==2:
                        raise Exception('no decoy molecules found')

                except Exception as e:
                    print e
                    stats.ligands_failed+=1
                    print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed
                    continue

                stats.ligands_parsed += 1
                print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed

                # create an output path to write binaries for protein and ligands
                path_to_pdb_subfolder = output_path + "/" + str(os.path.abspath(dirpath)).split("/")[-1]

                if not os.path.exists(path_to_pdb_subfolder):
                    os.makedirs(path_to_pdb_subfolder)

                # convert atomnames to tags and write the data to disk
                def atom_to_number(atomname):
                    atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
                    return atomic_tag_number

                receptor_elements = map(atom_to_number,prody_receptor.getElements())
                ligand_elements = map(atom_to_number,prody_positive.getElements())

                receptor_output_path = path_to_pdb_subfolder + "/" +str(os.path.abspath(dirpath)).split("/")[-1]
                save_av4(receptor_output_path,[0],receptor_elements,prody_receptor.getCoords())
                ligand_output_path = path_to_pdb_subfolder + "/" + path_to_positive.split("/")[-1].split(".")[0]
                save_av4(ligand_output_path,labels,ligand_elements,multiframe_ligand_coords)




convert_database_to_av4(database_path="../datasets/labeled_pdb",positives_folder="crystal_ligands",decoys_folder="docked_ligands",receptors_folder='receptors')
