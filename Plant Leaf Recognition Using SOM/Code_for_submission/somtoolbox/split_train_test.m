function [training_set_norm, testing_set_norm] = split_train_test(features)

% load new_features_63744

%Splitting the dataset into train and test
training_set = zeros(32*17, size(features,2));
training_set_norm = zeros(32*17, size(features,2));
testing_set = zeros(637-(32*17),size(features,2));
testing_set_norm = zeros(637-(32*17),size(features,2));
count = 0;
for i = 1:32
    position = find(features(:,1)==i);
    training_set((i-1)*17+1:i*17,:) = features(position(1:17),:);
    remaining = length(position)-17;
    testing_set(count+1:count+remaining,:) = features(position(18:end),:);
    count = count+remaining;    
    clear position
end

%Normalizing the data by subtracting the mean and dividing by the standard
%deviation

training_set_norm(:,1) = training_set(:,1);
training_set_norm(:,2:end) = training_set(:,2:end) - repmat(mean(training_set(:,2:end)),[size(training_set,1),1]);
training_set_norm(:,2:end) = training_set_norm(:,2:end)./repmat(std(training_set(:,2:end)),[size(training_set,1),1]);

testing_set_norm(:,1) = testing_set(:,1);
testing_set_norm(:,2:end) = testing_set(:,2:end) - repmat(mean(training_set(:,2:end)),[size(testing_set,1),1]);
testing_set_norm(:,2:end) = testing_set_norm(:,2:end)./repmat(std(training_set(:,2:end)),[size(testing_set,1),1]);