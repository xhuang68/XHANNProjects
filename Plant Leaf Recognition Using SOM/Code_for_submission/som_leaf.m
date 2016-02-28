clear
som_files_path = 'somtoolbox';

load('features.mat');

cd(som_files_path);

feature_list = [2:16 27:58];
msize = [22 10];
[training_set, testing_set] = split_train_test(features);

%Training the SOM
for  i= 1:size(training_set,1)
labels_train{i} = num2str(training_set(i,1));
end
labels_train = labels_train';

sD = som_data_struct(training_set(:,feature_list),'labels',labels_train);
sD = som_normalize(sD, 'var');

sMap = som_make(sD,'msize', msize);
sMap = som_autolabel(sMap,sD,'vote');

%Predictions for the test set
[bmus,qerrs] = som_bmus(sMap, testing_set(:,feature_list));

for i = 1:size(testing_set,1)
    if(ismember(sMap.labels(bmus(i)), ''))
        predictions_testing_set(i) = -1;
    else
        predictions_testing_set(i) = cellfun(@str2num, sMap.labels(bmus(i)));
    end
end

%Testing accuracy
correct_decisions_som = 0;
for i = 1:size(testing_set,1)
    if(testing_set(i,1)==predictions_testing_set(i))
        correct_decisions_som = correct_decisions_som + 1;
    end
end

accuracy_som = correct_decisions_som/size(testing_set,1);


%% kNN
D = pdist2(testing_set(:,feature_list),training_set(:,feature_list));
[result_knn,P] = knn(D, training_set(:,1), 3);
correct_decisions_knn = 0;
for i = 1:size(testing_set,1)
if(testing_set(i,1)==result_knn(i,3))
correct_decisions_knn = correct_decisions_knn + 1;
end
end
accuracy_knn = correct_decisions_knn/size(testing_set,1);


%% Linear SVM
[result_svm] = multisvm(training_set(:,feature_list),training_set(:,1),testing_set(:,feature_list));
correct_decisions_svm = 0;
for i = 1:size(testing_set,1)
if(testing_set(i,1)==result_svm(i))
correct_decisions_svm = correct_decisions_svm + 1;
end
end
accuracy_svm = correct_decisions_svm/size(testing_set,1);


%SOM visualization
figure
h = som_plotplane('hexa', msize, sMap.codebook);
title('Code Vectors for Each SOM Node')
U = som_umat(sMap);
figure;
imagesc(U);
title('U Matrix');
colorbar
axis equal
xlim([1,2*msize(2)-1]);


fprintf('\nAccuracy obtained using SOM = %f%%',accuracy_som*100);
fprintf('\nAccuracy obtained using kNN = %f%%',accuracy_knn*100);
fprintf('\nAccuracy obtained using SVM = %f%%\n',accuracy_svm*100);