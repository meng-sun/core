# Deep learning on protein images

First application of deep convolutional neural networks for drug-protein interaction prediction has appeared [in the paper of AtomWise](https://arxiv.org/abs/1510.02855) when the small 3D AlexNets have been trained on atoms of drug-protein complexes. We have experimented with the network structure and obtained predictions of a very high accuracy.

![alt tag](https://github.com/mitaffinity/core/blob/master/misc/alexnet.jpg)
**Fig1:** AlexNet as it was described in: [Krizhevsky et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)  

![alt tag](https://github.com/mitaffinity/core/blob/master/misc/netvision_cool.jpg)
**Fig2:** Network Vision. To represent input to the network, coordinates of atoms determined by crystallographers have to be converted to an image made of cubic pixels. Since on a small scale the world is very sparse, approximately 1 real-valued pixel in 10^7 zeros, an approximation has to be made. In our case every atom is assigned a numeric tag that fills the value of the cubic pixel with the size of 0.5A. This picture depicts the structure of an aminoglycoside nucleotidyltransferase - an enzyme important for nucleic acids metabolism with the substrate in it's binding site. Image of protein have been rendered in VMD in a form common for scientific literature. Small differently colored spheres represent an approximation of atomic coordinates that the network sees. 

![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/AlexNet3d.png)
**Fig3:** The structure of the network that performed very well on our previous Kaggle competition. Significant improvements comapred have been achieved by the smaller pixel size and very wide first layer convolutions. An interactive demo of the same network can be [seen here](http://ec2-54-201-177-210.us-west-2.compute.amazonaws.com/).

Here we provide a working example of the code that distinguishes correct docked positions of ligands from incorrect with an AUC of 0.934 after 80 epochs of training.

**Usage:**
<pre>git clone https://github.com/mitaffinity/core.git</pre>
Navigate your browser to https://inclass.kaggle.com/c/affinity4, and follow the registration steps to get the data.
Afterwards, change the path in FLAGS.database_path to the location on your local machine, and type
<pre>python av4.py</pre>
this script should automatically index folders in your database and start training.

Our old [Kaggle competition](https://inclass.kaggle.com/c/affinity), and software can be found [here](https://github.com/mitaffinity/core/releases). Enjoy!
