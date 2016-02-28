=============================================
EEE511 Final Project
Spike Sorting Algorithm Using Wavelet + GMM
=============================================
Suhas Lohit 1206451918
Rakshit Raghavan 1207751060
Xiao Huang 1206424709
=============================================
INTRO:
the code is to run our algorithm and 
reproduce the results we got for 
BOTH TESTING and TRAINING data
The code was tested on MATLAB 2014b and works well.
It takes approximately 1 minute to run the code entirely.
Some warnings may be displayed and can be ignored.

IMPORTANT
because of GMM initialization issue
this script will load the best kmeans centroids results in our experiment
as the initialization of GMM to reproduce the best results
If using a different dataset or if you need a different initialization,
uncomment line 148

=============================================
STEP:
open main.m file using Matlab
simply click the run button
no need to manually load or change anything
all parameters and data needed are loaded automatically
p.s.
because of GMM initialization issue
this script will load the best kmeans centroids results in our experiment as the initialization of GMM to reproduce the best results
=============================================
INPUT:
none
=============================================
OUTPUT:
A. 2 EXCEL files, each has 4 sheets representing for 4 datasets
(saved in the same directory)
--- testing_data_results.xlsx
--- training_data_results.xlsx

B. 8 .mat files
(saved in the same directory)
--- testing_dataset_1_results.mat
--- testing_dataset_2_results.mat
--- testing_dataset_3_results.mat
--- testing_dataset_4_results.mat
--- training_dataset_1_results.mat
--- training_dataset_2_results.mat
--- training_dataset_3_results.mat
--- training_dataset_4_results.mat

C. 8 figures
(plotted in Matlab)
--- figure1.testing_dataset_1_results
--- figure2.testing_dataset_2_results
--- figure3.testing_dataset_3_results
--- figure4.testing_dataset_4_results
--- figure5.training_dataset_1_results
--- figure6.training_dataset_2_results
--- figure7.training_dataset_3_results
--- figure8.training_dataset_4_results
=============================================