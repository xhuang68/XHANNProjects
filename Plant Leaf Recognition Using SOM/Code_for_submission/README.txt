Use either of the two options below:

1. In order to compute the recognition accuracy on the dataset using SOM, SVM and k-NN, it is sufficient to run som_leaf.m . This makes use of the previously extracted features stored in
   features.mat. This takes about 10 seconds to finish running and display the results. 

2. Another option is to run main_folio_features.m . This first extracts the features using the dataset provided, saves it into features.mat and then proceeds to train and test SOM, SVM
   k-NN classifiers. Note that this takes longer - about 2 minutes. 
