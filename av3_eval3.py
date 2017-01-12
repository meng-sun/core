import time,re
import tensorflow as tf
import numpy as np
from av5BN import FLAGS,max_net, with_memory, sigmoid_cross_entropy#,weighted_cross_entropy_mean_with_labels
from av3_input import launch_enqueue_workers

# set up global parameters
FLAGS.saved_session = '../summaries//saved_state-59999'

FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)


# todo log AUC
# todo cross entropy
# change predictions to all_prediction to avoid confusion
# todo better epoch counter that can stop


class store_predictions:
    """stores all of the predictions
    unique and sorted by protein-ligand pairs"""
    pl_pairs =np.array([],dtype=str)
    # collects all of the predictions (for the same pair) into an array of objects


    predictions = np.array([])
    # collects labels
    labels = np.array([])

    # formatting options in the text file
    def usd_format(self,number_array):
        """converts numpy array into user defined format"""
        return np.char.array(np.around(number_array,decimals=3),itemsize=5)

    def top_100_score(self,predictions,labels):
        """ takes sorted in descending order by predictions list of labels
        if the dataset had n=100 drugs, how many drugs are in top 100 of the list
        top_100_score = (TP in head -n)/n, where n is a number of Positives in a whole dataset
        takes list of predictions, corresponding labels, returns float -percent that is correct"""

        # sort the array by predictions
        order = np.flipud(predictions.argsort())
        labels = labels[order]

        # take the top n
        num_positives = np.sum(np.asarray(labels,dtype=bool))

        # if no positive labels return nan
        if num_positives == 0:
            return float('nan')
        else:
            return np.sum(labels[0:num_positives]) / num_positives * 100

    def auc(self,predictions,labels):
        """calculates area under the curve AUC for binary predictions/labels needs
        sorted in descending order predictions"""

        # sort the array by predictions in descending order in case it has not been done
        order = np.flipud(predictions.argsort())
        labels = labels[order]

        # clean labels, calculate the number of positive labels
        labeled_true = (np.asarray(labels,dtype=bool) == True)
        num_positives = np.sum(labeled_true)
        num_predictions = len(labeled_true)

        # If no positive result return nan
        if num_positives == 0:
            return float('nan')

        # slide from top to the bottom;
        # each time slide the threshold so as to predict one more label as positive
        roc_curve = np.array([0.0,0.0])
        TP_above_threshold = 0
        for predict_as_positive in range(num_predictions):
            if labeled_true[predict_as_positive] == True:
                TP_above_threshold +=1
            # calculate True Positives Rate
            # TPR = TP / num_real_positives
            TPR = TP_above_threshold / float(num_positives)
            
            # FPR = FP / num_real_negatives
            FPR = (predict_as_positive +1 - TP_above_threshold) / (num_predictions - float(num_positives))
            
            roc_curve = np.vstack((roc_curve,[FPR,TPR]))

        roc_curve = np.vstack((roc_curve,[1.0,1.0]))

        
        # reduce into TP and FP rate, integrate with trapezoid to calculate AUC
        auc = np.trapz(roc_curve[:,1], x=roc_curve[:,0])


        return auc

    def confusion_matrix(self,predictions,labels):
        """calculaets and returns the confusion matrix"""
        TP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == True))
        FP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == False))
        FN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == True))
        TN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == False))

        return np.array([[TP,FP],[FN,TN]])



    def add_batch(self,ligand_file_path,receptor_file_path,batch_predictions,batch_labels):
        """maintains sorted by protein-ligand pairs lists of various predictions
        splits the batch into "new" and "known" protein-ligand pairs
        adds predictions of "known" ligand pairs with "," appends new protein-ligand pairs as they are"""


        # extract meaningful names from the file path
        def extract_file_from_path(file_path):
            return file_path.split("/")[-1]

        ligand_filenames = np.char.array(map(extract_file_from_path,ligand_file_path))
        receptor_filenames = np.char.array(map(extract_file_from_path,receptor_file_path))
        batch_pl_pairs = ligand_filenames + "," + receptor_filenames

        # sort the batch by protein-ligand pairs
        order = batch_pl_pairs.argsort()
        batch_pl_pairs = batch_pl_pairs[order]
        batch_predictions = batch_predictions[order]
        batch_labels = batch_labels[order]

        # check if all of the entries in the batch are unique
        if not np.array_equal(batch_pl_pairs,np.unique(batch_pl_pairs)):
            raise Exception("batch has duplicate entries")

        # get binmask with True for each non-unique protein-ligand pair, False for unique protein-ligand pair
        binmask_self = (np.searchsorted(batch_pl_pairs,self.pl_pairs, 'right') - np.searchsorted(batch_pl_pairs,self.pl_pairs,'left')) == 1
        binmask_batch = (np.searchsorted(self.pl_pairs,batch_pl_pairs, 'right') - np.searchsorted(self.pl_pairs,batch_pl_pairs,'left')) == 1

        # check if the entries appended to each other have similar names
        if not np.array_equal(batch_pl_pairs[binmask_batch],self.pl_pairs[binmask_self]):
            raise Exception('Error while merging arrays. Names do not match')

        # check if labels are similar
        if not np.array_equal(batch_labels[binmask_batch],self.labels[binmask_self]):
            raise Exception('Error while merging arrays. Labels for the same example should be similar')

        # split into overlapping and not overlapping entries
        overlap_pl_pairs = batch_pl_pairs[binmask_batch]
        overlap_predictions = np.char.array(self.predictions[binmask_self])
        batch_overlap_predictions = self.usd_format(batch_predictions[binmask_batch])

        # for known entries join all of the predictions together
        overlap_predictions = overlap_predictions + "," + batch_overlap_predictions
        overlap_labels = batch_labels[binmask_batch]

        # merge unique and not unique predictions
        self.pl_pairs = np.hstack((self.pl_pairs[-binmask_self],batch_pl_pairs[-binmask_batch],overlap_pl_pairs))
        self.predictions = np.hstack((self.predictions[-binmask_self],self.usd_format(batch_predictions[-binmask_batch]),overlap_predictions))
        self.labels = np.hstack((self.labels[-binmask_self],batch_labels[-binmask_batch],overlap_labels))

        # now sort everything by the first column
        order = self.pl_pairs.argsort()
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]


    def save_predictions(self,file_path):
        """sorts in descending order of confidence,computes average predictions,formats and writes into file"""
        # compute average of predictions
        num_examples = len(self.labels)

        if num_examples == 0:
            raise Exception ("nothing to save")

        def string_to_average(string):
            return np.average(np.array(string.split(","),dtype=float))
        prediction_averages = np.around(map(string_to_average,self.predictions),decimals=3)

        # sort by prediction averages
        order = np.flipud(prediction_averages.argsort())
        prediction_averages = prediction_averages[order]
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]
        # write all of the predictions to the file
        f = open(file_path + "_predictions.txt", 'w')

        for i in range(num_examples):
            f.write((str(prediction_averages[i]) + " "*10)[:10]
                    + (str(self.labels[i]) + " "*50)[:10]
                    + str(self.pl_pairs[i] + " "*50)[:50]
                    + str(self.predictions[i] + " "*50)[:50]
                    + "\n")

        f.close()
        # write and save some metadata

        f = open(file_path + "_scores.txt", 'w')
        f.write("top 100 score: ")
        f.write(str(self.top_100_score(self.predictions,self.labels)))
        f.write("\nAUC: ")
        f.write(str(self.auc(prediction_averages,self.labels)))
        f.write("\nconfusion matrix: ")
        f.write(str(self.confusion_matrix(prediction_averages,self.labels)))
        f.close()

        # write a file in Kaggle MAP{K} submision format
        # the form is:
        # Protein1, Ligand3 Ligand4 Ligand2
        # Protein2, Ligand5 Ligand9 Ligand7

        raw_database_array = np.genfromtxt(FLAGS.test_set_file_path, delimiter=',', dtype=str)
        receptor_set = raw_database_array[:,2]
        receptor_set = list(set(map(lambda x:x.split('.')[0].split('/')[-1],receptor_set)))
        submission = {}
        for i in range(num_examples):
            # get the name of the ligand and protein
            ligand,receptor = self.pl_pairs[i].split(',')
            ligand = ligand.split('/')[-1].split('.')[0]
            receptor = receptor.split('/')[-1].split('.')[0]
            # add all protein-ligand pairs to submission
            if not receptor in submission.keys():
                submission[receptor] = {}
                submission[receptor]['ligands'] = [ligand]
                submission[receptor]['score'] = [prediction_averages[i]]
            else:
                submission[receptor]['ligands'].append(ligand)
                submission[receptor]['score'].append(prediction_averages[i])
        
        # write and save submisison to file
        # if failed to predict any liagnd for a receptor
        # use placeholder 'L' as predict result
        # e.g. P1234,L
        with open(file_path+'_submission.csv','w') as f:
            f.write('Id,Expected\n')
            for key in receptor_set:
                if key in submission.keys():
                    ligands = np.array(submission[key]['ligands'])
                    scores = np.array(submission[key]['score'])
                    ligands = ligands[np.flipud(scores.argsort())]
                    f.write(key+','+' '.join(ligands)+'\n')
                else:
                    f.write(key+','+'L'+'\n')

def evaluate_on_train_set():
    "train a network"

    # create session all of the evaluation happens in one
    sess = tf.Session()
    train_image_queue,filename_coordinator = launch_enqueue_workers(sess=sess,pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                                    num_workers=FLAGS.num_workers, batch_size=FLAGS.batch_size,
                                                                    database_index_file_path=FLAGS.test_set_file_path,num_epochs=2)
    y_, x_image_batch,ligand_filename,receptor_filename = train_image_queue.dequeue_many(FLAGS.batch_size)
    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(x_image_batch, keep_prob)
    
    cross_entropy_mean = tf.Variable(0, dtype=tf.float32)
    cross_entropy_mean = with_memory(y_conv, y_, sigmoid_cross_entropy, cross_entropy_mean)
    #cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y_))
    #(weighted_cross_entropy_mean_with_labels(y_conv,y_))

    # compute softmax over raw predictions
    predictions = tf.nn.softmax(y_conv)[:,1]

    # restore variables from sleep
    saver = tf.train.Saver()
    saver.restore(sess,FLAGS.saved_session)

    # create a variable to store all predictions
    all_predictions = store_predictions()
    batch_num = 0

    while not filename_coordinator.stop:
        start = time.time()
	if (batch_num % 100 == 99):
            time.sleep(1)        
	my_ligand_filename,my_receptor_filename,my_predictions,labels,my_cross_entropy = sess.run([ligand_filename,receptor_filename,predictions,y_,cross_entropy_mean],feed_dict={keep_prob:1})
	# write done the av3_score
	av3_score= np.vstack((my_ligand_filename,my_receptor_filename,my_predictions.astype(str),labels.astype(str)))
        av3_score_list = av3_score.transpose().tolist()
        with open(FLAGS.predictions_file_path+"_av3_eval_score.csv","a") as fout:
            for entry in av3_score_list:
                fout.write(','.join(entry)+'\n')
        
	#all_predictions.add_batch(my_ligand_filename,my_receptor_filename,my_predictions,labels)
        print "step:", batch_num, "test error:", my_cross_entropy, "examples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        batch_num +=1

    all_predictions.save_predictions(FLAGS.predictions_file_path)


evaluate_on_train_set()


print "all done!"
