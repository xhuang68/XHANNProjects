%% IMPORTANT!!!
% EEE511 Final Project
% Spike Sorting Algorithm Using Wavelet + GMM
%
% Suhas Lohit 1206451918
% Rakshit Raghavan 1207751060
% Xiao Huang 1206424709
%
% INTRO:
% this script is to run our algorithm and 
% reproduce the results we got for BOTH TESTING and TRAINING data.
% The code was tested on MATLAB 2014b and works well. 
% Some warnings may be displayed and can be ignored. 
% It takes approximately 1 minute to run the code entirely.
%
% STEP:
% simply click the run button
% no need to manually load or change anything
% all parameters and data needed are loaded automatically
% IMPORTANT:
% because of GMM initialization issue
% this script will load the best kmeans centroids results in our experiment
% as the initialization of GMM to reproduce the best results
% If using a different dataset or if you need a different initialization,
% uncomment line 148
%
% INPUT:
% none
%
% OUTPUT:
% A. 2 EXCEL files, each has 4 sheets representing for 4 datasets
% (saved in the same directory)
% --- testing_data_results.xlsx
% --- training_data_results.xlsx
%
% B. 8 .mat files
% (saved in the same directory)
% --- testing_dataset_1_results.mat
% --- testing_dataset_2_results.mat
% --- testing_dataset_3_results.mat
% --- testing_dataset_4_results.mat
% --- training_dataset_1_results.mat
% --- training_dataset_2_results.mat
% --- training_dataset_3_results.mat
% --- training_dataset_4_results.mat
%
% C. 8 figures
% (plotted in Matlab)
% --- figure1.testing_dataset_1_results
% --- figure2.testing_dataset_2_results
% --- figure3.testing_dataset_3_results
% --- figure4.testing_dataset_4_results
% --- figure5.training_dataset_1_results
% --- figure6.training_dataset_2_results
% --- figure7.training_dataset_3_results
% --- figure8.training_dataset_4_results

clear
clc
close all
warning('off','all')

dataset_name = {'testing_data', 'training_data'};
figure_num = 1;

% Delete the result files if they already exist

if(exist('training_data_results.xlsx','file')==2)
    delete('training_data_results.xlsx')
end

if(exist('testing_data_results.xlsx','file')==2)
    delete('testing_data_results.xlsx')
end

% Load all the training and testing datasets into different variables

spikes_train_dataset_1 = load('training_data/Sampledata_1.mat');
spikes_train_dataset_2 = load('training_data/Sampledata_2.mat');
spikes_train_dataset_3 = load('training_data/Sampledata_3.mat');
spikes_train_dataset_4 = load('training_data/Sampledata_4.mat');

spikes_test_dataset_1 = load('testing_data/Sampledata_test_1.mat');
spikes_test_dataset_2 = load('testing_data/Sampledata_test_2.mat');
spikes_test_dataset_3 = load('testing_data/Sampledata_test_3.mat');
spikes_test_dataset_4 = load('testing_data/Sampledata_test_4.mat');

for iter_time = 1:2
    target = cell2mat(dataset_name(iter_time));
    for dataset = 1:4

        %% Load spikes file
        clear spikes
        if (strcmp(target, 'training_data'))
            switch(dataset)
                case 1
                    spikes = spikes_train_dataset_1.spikes;
                    disp('Working on Training Dataset 1');
                case 2
                    spikes = spikes_train_dataset_2.spikes;
                    disp('Working on Training Dataset 2');
                case 3
                    spikes = spikes_train_dataset_3.spikes;
                    disp('Working on Training Dataset 3');
                case 4
                    spikes = spikes_train_dataset_4.spikes;
                    disp('Working on Training Dataset 4');
            end
        else
            switch(dataset)
                case 1
                    spikes = spikes_test_dataset_1.spikes;
                    disp('Working on Testing Dataset 1');
                case 2
                    spikes = spikes_test_dataset_2.spikes;
                    disp('Working on Testing Dataset 2');
                case 3
                    spikes = spikes_test_dataset_3.spikes;
                    disp('Working on Testing Dataset 3');
                case 4
                    spikes = spikes_test_dataset_4.spikes;
                    disp('Working on Testing Dataset 4');
            end
        end
        
        %% Load parameters
        load parameters.mat; % load the parameters struct array
        num_features_wavelet = 48;
        num_clusters_k_means = parameters(dataset).num_clusters_k_means;
        num_components_gmm = num_clusters_k_means;
        disp('loading parameters ...');
        disp(' ');

        %% Feature extraction
        % Wavelet features
        [wavelet_features] = extract_wavelet_features(spikes,num_features_wavelet);
        wavelet_features = zscore(wavelet_features); %Normalizing

        %% Clustering
        % Step1. kmeans on wavelet features
        if (strcmp(target, 'training_data'))
            centroids_wavelet = parameters(dataset).training_centroids_wavelet;  
        else
            centroids_wavelet = parameters(dataset).testing_centroids_wavelet;
        end
        disp('loading kmeans centroids ...');
        disp(' ');
        % [idx_wavelet_kmeans,centroids_wavelet] = kmeans(wavelet_features,num_clusters_k_means);

        % Step2. GMM on wavelet features
        [px, model] = gmm(wavelet_features, centroids_wavelet);
        idx_wavelet_gmm = zeros(size(px, 1), 1);
        for j = 1 : size(px, 1)
            target_idx = px(j, :);
            idx_wavelet_gmm(j, 1) = find(target_idx == max(target_idx));
        end

        %% Generating results and save as .mat
        labels = idx_wavelet_gmm;
        if (strcmp(target, 'training_data'))
            results_file = ['training_dataset_', num2str(dataset), '_results'];
        else
            results_file = ['testing_dataset_', num2str(dataset), '_results'];
        end
        save(results_file, 'labels');
        disp([results_file, '.mat is saved in the same directory']);
        disp(' ');

        %% Plotting results 
        % Wavelet + GMM clusters
        figure
        for i = 1:length(unique(labels))
        unique_clusters = unique(labels);
        subplot(1,length(unique_clusters),i)
        plot(transpose(spikes(labels == unique_clusters(i),:)));
        title(['cluster ',num2str(i), ': # ', num2str(length(find(labels == unique_clusters(i))))]); 
        axis([0 48 -3 3]);
        set(gca,'XTick', 0:12:48);
        set(gca,'YTick', -3:1:3);
        xlabel('Samples');
        ylabel('Waveform Value');
        axis square
        end
        ha = axes('Position',[0.1 0 0.8 0.8],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
        if (strcmp(target, 'training_data'))
            figure_name = 'Training';
        else
            figure_name = 'Testing';
        end 
        text(0.5, 1,['\bf', figure_name, ' Dataset ', num2str(dataset), ' Cluster Assignments'],'HorizontalAlignment' ,'center','VerticalAlignment', 'top')
        disp(['plotting figure ', num2str(figure_num), ' ...']);
        disp(' ');
        figure_num = figure_num + 1;
        
        %% Generating excel files
        if (strcmp(target, 'training_data'))
            excel_file = 'training_data_results.xlsx';
        else
            excel_file = 'testing_data_results.xlsx';
        end
        sheet = ['dataset' num2str(dataset)];
        xlswrite(excel_file, labels, sheet);
        disp([sheet, ' is saved to ', excel_file, ' in the same directory']);
        disp(' ');
                
    end
    
    %To delete the default sheet 1 in excel file
    newExcel = actxserver('excel.application');
    newExcel.DisplayAlerts = false;
    excelWB = newExcel.Workbooks.Open(fullfile(pwd,excel_file),0,false);
    excelWB.Sheets.Item(1).Delete;
    excelWB.Save();
    excelWB.Close();
    newExcel.Quit();
    delete(newExcel);
end
warning('on','all')