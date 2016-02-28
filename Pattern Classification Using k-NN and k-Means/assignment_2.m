function assignment_2()
    clear
    clc

    [Gaussian_train_class1, Gaussian_train_class2, Gaussian_test_class1, Gaussian_test_class2, circle_train_class1, circle_train_class2, circle_test_class1, circle_test_class2] = generate_data;
    
   
    %% K-Means
    accuracy=zeros(5,5);
    for k1=1:5 
        for k2=1:5
            %Choosing the training and crossvalidation classes from Gaussian classes 1 & 2 
            train_class1 = Gaussian_train_class1(1:750,:);
            cv_1 = Gaussian_train_class1(751:1000,:);
            
            train_class2 = Gaussian_train_class2(1:750,:);
            cv_2 = Gaussian_train_class2(751:1000,:);
    
            cluster_center1 = k_means(train_class1,k1);
            cluster_center2 = k_means(train_class2,k2);
    
            cluster_center = [cluster_center1 ; cluster_center2];
            cv = [cv_1 ; cv_2];
            accuracy(k1,k2) = compute_accuracy(cluster_center,cv,k1,k2);
        end
    end
    
    [max_accuracy, ind]  = max(accuracy(:));
    [row,col] = ind2sub(size(accuracy),ind);
    
    cluster_center_final_1 = k_means(train_class1,row);
    cluster_center_final_2 = k_means(train_class2,col);
    cluster_center_final = [cluster_center_final_1; cluster_center_final_2];
    test_set = [Gaussian_test_class1; Gaussian_test_class2];
    [accuracy_test_Gaussian, pred] = compute_accuracy(cluster_center_final,test_set,row,col); 
    
    kmeans_Gaussian_class1=[];
    kmeans_Gaussian_class2=[];
    for i = 1:length(test_set)
        if pred(i,1) == 1
            kmeans_Gaussian_class1=[kmeans_Gaussian_class1;test_set(i,:)];
        else
            kmeans_Gaussian_class2=[kmeans_Gaussian_class2;test_set(i,:)];
        end
    end
    
    gnd_truth=[ones(size(test_set,1)/2,1);zeros(size(test_set,1)/2,1)];
    cmatrix_kmeans_Gaussian = confusion_matrix(pred,gnd_truth);
    disp('Confusion Matrix for Gaussian clusters data with K-Means algorithm:')
    cmatrix_kmeans_Gaussian

    accuracy=zeros(5,5);
    for k1=1:5 
        for k2=1:5
            %Choosing the training and cross validation classes from circle classes 1 & 2 
            train_class1 = circle_train_class1(1:750,:);
            cv_1 = circle_train_class1(751:1000,:);
            
            train_class2 = circle_train_class2(1:750,:);
            cv_2 = circle_train_class2(751:1000,:);
    
            cluster_center1 = k_means(train_class1,k1);
            cluster_center2 = k_means(train_class2,k2);
    
            cluster_center = [cluster_center1 ; cluster_center2];
            cv = [cv_1 ; cv_2];
            [accuracy(k1,k2),~] = compute_accuracy(cluster_center,cv,k1,k2);
            
        end
    end
    
    [max_accuracy, ind]  = max(accuracy(:));
    [row, col] = ind2sub(size(accuracy),ind);
    
    cluster_center_final_1 = k_means(train_class1,row);
    cluster_center_final_2 = k_means(train_class2,col);
    cluster_center_final = [cluster_center_final_1; cluster_center_final_2];
    test_set = [circle_test_class1; circle_test_class2];
    [accuracy_test_circle, pred] = compute_accuracy(cluster_center_final,test_set,row,col); 
    
    kmeans_circle_class1=[];
    kmeans_circle_class2=[];
    for i = 1:length(test_set)
        if pred(i,1) == 1
            kmeans_circle_class1=[kmeans_circle_class1;test_set(i,:)];
        else
            kmeans_circle_class2=[kmeans_circle_class2;test_set(i,:)];
        end
    end
    
    gnd_truth=[ones(size(test_set,1)/2,1);zeros(size(test_set,1)/2,1)];
    cmatrix_kmeans_circle = confusion_matrix(pred,gnd_truth);
    disp('Confusion Matrix for circular data with K-Means algorithm:')
    cmatrix_kmeans_circle
    
    
    %% kNN
    % Gaussian data classifier
    %---------- preparing train data ----------%
    % load origin data
    inorder_train_data = [Gaussian_train_class1; Gaussian_train_class2];
    inorder_train_label = [ones(length(Gaussian_train_class1), 1); zeros(length(Gaussian_train_class2), 1)];
    random = randperm(length(inorder_train_data));
    train_data = zeros(length(inorder_train_data), 2);
    train_label = -ones(length(inorder_train_data), 1);
    for i = 1:length(inorder_train_data)
        train_data(i,1) = inorder_train_data(random(i), 1);
        train_data(i,2) = inorder_train_data(random(i), 2);
        train_label(i) = inorder_train_label(random(i));
    end

    Gaussian_cv_k_range = 15; % set the range of k for cross validation
    Gaussian_cv_accuracy = zeros(Gaussian_cv_k_range, 1); % init the accuracy vector for each k

    for k = 1: Gaussian_cv_k_range
        Gaussian_cv_ave_accuracy = knn_cross_validation(5, train_data, train_label, k);
        Gaussian_cv_accuracy(k) = Gaussian_cv_ave_accuracy;
    end

    Gaussian_cv_max_accuracy = max(Gaussian_cv_accuracy); % find the max accuract
    Gaussian_optimal_k_vector = find(Gaussian_cv_accuracy == Gaussian_cv_max_accuracy); % find all the k value with max accuract
    Gaussian_optimal_k = Gaussian_optimal_k_vector(1); % find the optimal k value

    %---------- testing step ----------%
    test_data = [Gaussian_test_class1; Gaussian_test_class2];
    test_label = [ones(length(Gaussian_test_class1), 1); zeros(length(Gaussian_test_class2), 1)];

    % implemented knn function
    predict_label = knn(train_data,train_label,test_data,test_label,Gaussian_optimal_k);

    % build confusion matrix
    Gaussian_cnn_test_accuracy = length(find(predict_label == test_label)) / length(test_label);
    cmatrix_knn_Gaussian(1:2,1:2)=0; 
    knn_Gaussian_class1=[];
    knn_Gaussian_class2=[];
    for i = 1:length(test_data)
        if i <= 1000
            if predict_label(i,1) == 1
                cmatrix_knn_Gaussian(1,1) = cmatrix_knn_Gaussian(1,1) + 1;
                knn_Gaussian_class1=[knn_Gaussian_class1;test_data(i,:)];
            else
                cmatrix_knn_Gaussian(1,2) = cmatrix_knn_Gaussian(1,2) + 1;
                knn_Gaussian_class2=[knn_Gaussian_class2;test_data(i,:)];
            end
        else
            if predict_label(i,1) == 1
                cmatrix_knn_Gaussian(2,1) = cmatrix_knn_Gaussian(2,1) + 1;
                knn_Gaussian_class1=[knn_Gaussian_class1;test_data(i,:)];
            else
                cmatrix_knn_Gaussian(2,2) = cmatrix_knn_Gaussian(2,2) + 1;
                knn_Gaussian_class2=[knn_Gaussian_class2;test_data(i,:)];
            end
        end
    end
    disp('Confusion Matrix for Gaussian clusters data with kNN algorithm:')
    cmatrix_knn_Gaussian

    % plot Gaussian figure
%     load Gaussian_data_frame
    % figure: KNN Classifier Testing Data Distribution
    figure('name', 'KNN Classifier Testing Data Distribution')
    subplot(1,3,1);
    plot(Gaussian_test_class1(:,1),Gaussian_test_class1(:,2),'ro'),hold on
    plot(Gaussian_test_class2(:,1),Gaussian_test_class2(:,2),'b+'),hold on
    title('Ground Truth')
    legend('class1','class2')
    axis square
%     axis(Gaussian_data_frame)

    subplot(1,3,2);
    plot(knn_Gaussian_class1(:,1),knn_Gaussian_class1(:,2),'ro'),hold on
    plot(knn_Gaussian_class2(:,1),knn_Gaussian_class2(:,2),'b+'),hold on
    title('kNN Classifier Output')
    legend('class1','class2')
    axis square
    
    subplot(1,3,3);
    plot(kmeans_Gaussian_class1(:,1),kmeans_Gaussian_class1(:,2),'ro'),hold on
    plot(kmeans_Gaussian_class2(:,1),kmeans_Gaussian_class2(:,2),'b+'),hold on
    title('K-Means Classifier Output')
    legend('class1','class2')
    axis square
%     axis(Gaussian_data_frame)

    % circle data classifier
    inorder_train_data = [circle_train_class1; circle_train_class2];
    inorder_train_label = [ones(length(circle_train_class1), 1); zeros(length(circle_train_class2), 1)];

    % shuffle inorder data
    random = randperm(length(inorder_train_data));
    train_data = zeros(length(inorder_train_data), 2);
    train_label = -ones(length(inorder_train_data), 1);
    for i = 1:length(inorder_train_data)
        train_data(i,1) = inorder_train_data(random(i), 1);
        train_data(i,2) = inorder_train_data(random(i), 2);
        train_label(i) = inorder_train_label(random(i));
    end

    %---------- cross validation/find optimal k value ----------%
    circle_cv_k_range = 15; % set the range of k for cross validation
    circle_cv_accuracy = zeros(circle_cv_k_range, 1); % init the accuracy vector for each k

    for k = 1: circle_cv_k_range
        circle_cv_ave_accuracy = knn_cross_validation(5, train_data, train_label, k);
        circle_cv_accuracy(k) = circle_cv_ave_accuracy;
    end

    circle_cv_max_accuracy = max(circle_cv_accuracy); % find the max accuract
    circle_optimal_k_vector = find(circle_cv_accuracy == circle_cv_max_accuracy); % find all the k value with max accuract
    circle_optimal_k = circle_optimal_k_vector(1); % find the optimal k value
    

    %---------- testing step ----------%
    test_data = [circle_test_class1; circle_test_class2];
    test_label = [ones(length(circle_train_class1), 1); zeros(length(circle_train_class2), 1)];

    % implemented knn function
    predict_label = knn(train_data,train_label,test_data,test_label,circle_optimal_k);

    circle_cnn_test_accuracy = length(find(predict_label == test_label)) / length(test_label);
    cmatrix_knn_circle(1:2,1:2)=0; 
    knn_cicle_class1=[];
    knn_circle_class2=[];
    for i = 1:length(test_data)
        if i <= 1000
            if predict_label(i,1) == 1
                cmatrix_knn_circle(1,1) = cmatrix_knn_circle(1,1) + 1;
                knn_cicle_class1=[knn_cicle_class1;test_data(i,:)];
            else
                cmatrix_knn_circle(1,2) = cmatrix_knn_circle(1,2) + 1;
                knn_circle_class2=[knn_circle_class2;test_data(i,:)];
            end
        else
            if predict_label(i,1) == 1
                cmatrix_knn_circle(2,1) = cmatrix_knn_circle(2,1) + 1;
                knn_cicle_class1=[knn_cicle_class1;test_data(i,:)];
            else
                cmatrix_knn_circle(2,2) = cmatrix_knn_circle(2,2) + 1;
                knn_circle_class2=[knn_circle_class2;test_data(i,:)];
            end
        end
    end
    disp('Confusion Matrix for circular data with kNN algorithm:')
    cmatrix_knn_circle

    % plot circle figure
    % figure: KNN Classifier Testing Data Distribution
    figure('name', 'KNN Classifier Results on Circular Data')
    subplot(1,3,1);
    plot(circle_test_class1(:,1),circle_test_class1(:,2),'ro'),hold on
    plot(circle_test_class2(:,1),circle_test_class2(:,2),'b+'),hold on
    title('Ground Truth')
    legend('class1','class2')
    axis square

    subplot(1,3,2)
    plot(knn_cicle_class1(:,1),knn_cicle_class1(:,2),'ro'),hold on
    plot(knn_circle_class2(:,1),knn_circle_class2(:,2),'b+'),hold on
    title('kNN Classifier Output')
    legend('class1','class2')
    axis square
    
    subplot(1,3,3)
    plot(kmeans_circle_class1(:,1),kmeans_circle_class1(:,2),'ro'),hold on
    plot(kmeans_circle_class2(:,1),kmeans_circle_class2(:,2),'b+'),hold on
    title('K-Means Classifier Output')
    legend('class1','class2')
    axis square
    

end
 
%%
function [accuracy,pred] = compute_accuracy(cluster_center,cv,k1,k2)
    ground_truth=[ones(size(cv,1)/2,1);zeros(size(cv,1)/2,1)];
    accuracy = 0;
    pred = zeros(size(cv,1),1);
    for i = 1:size(cv,1)
        dist=10^6;
        d1=cv(i,:);
        closest_center=0;
        for j=1:(k1+k2)
            c1=cluster_center(j,:);
            dist1=sum((d1-c1).^2);
            if dist1<dist
                closest_center=j;
                dist=dist1;
            end
        end
        if closest_center<k2;
            pred(i)=1;
        else pred(i)=0;
        end

         if pred(i)==ground_truth(i)
            accuracy = accuracy + 1;
         end    
    end

    accuracy = accuracy/(size(cv,1));
end

%%
function [Gaussian_train_class1, Gaussian_train_class2, Gaussian_test_class1, Gaussian_test_class2, circle_train_class1, circle_train_class2, circle_test_class1, circle_test_class2] = generate_data()
    %% generate Gaussian cluster data
    total = 1000; % number of data

    mu1 = [0 0];
    Sigma1 = [1 0; 0 1]; % two features are uncorrelated
    mu2 = [2.5 0];
    Sigma2 = [1 0; 0 1];

    Gaussian_train_class1 = mvnrnd(mu1, Sigma1, total);
    Gaussian_train_class2 = mvnrnd(mu2, Sigma2, total);
    Gaussian_test_class1 = mvnrnd(mu1, Sigma1, total);
    Gaussian_test_class2 = mvnrnd(mu2, Sigma2, total);

    minxy = min(min(Gaussian_test_class1'), min(Gaussian_test_class2'));
    maxxy = max(max(Gaussian_test_class1'), max(Gaussian_test_class2'));

    Gaussian_data_frame = [minxy(1)-0.5 maxxy(1)+0.5 minxy(2)-0.5 maxxy(2)+0.5];

    %% generate unit circle data
    r = 1;
    circle_train_class1 = generate_inside_circle_data(r, total);
    circle_train_class2 = generate_outside_circle_data(r, total);
    circle_test_class1 = generate_inside_circle_data(r, total);
    circle_test_class2 = generate_outside_circle_data(r, total);
    circle_train_class1 = circle_train_class1';
    circle_train_class2 = circle_train_class2';
    circle_test_class1 = circle_test_class1';
    circle_test_class2 = circle_test_class2';

    %% plot Gaussian cluster data figure
    % figure1: Original Gaussian Data Distribution
    figure('name', 'Gaussian Clusters')
    % plot first cluster
    x1 = -3:.2:6; x2 = -3:.2:3;
    [X1,X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],mu1,Sigma1);
    F = reshape(F,length(x2),length(x1));
    subplot(1,3,1)
    surf(x1,x2,F);
    caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
    axis([-3 6 -3 3 0 .2])
    xlabel('x1'); ylabel('x2'); zlabel('prob. density');
    title('Class1 Distribution')
    axis square
    % plot seconde cluster
    x1 = -3:.2:6; x2 = -3:.2:3;
    [X1,X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],mu2,Sigma2);
    F = reshape(F,length(x2),length(x1));
    subplot(1,3,2)
    surf(x1,x2,F);
    caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
    axis([-3 6 -3 3 0 .2])
    xlabel('x1'); ylabel('x2'); zlabel('prob. density');
    title('Class2 Distribution')
    axis square
    % plot the training data
    subplot(1,3,3)
    plot(Gaussian_train_class1(:,1),Gaussian_train_class1(:,2),'ro');
    hold on
    plot(Gaussian_train_class2(:,1),Gaussian_train_class2(:,2),'bx');
    hold on
    title('Training Data')
    axis square
    legend('class1','class2')

    %% plot circle data figure
    % figure2: Original Unit Circle Data Distribution
    figure('name', 'Circular Data')
    t=0:0.02:2*pi; 
    plot(r*cos(t),r*sin(t),'green', 'LineWidth',3); 
    hold on 
    plot(circle_train_class1(:,1), circle_train_class1(:,2), 'rx')
    hold on
    plot(circle_train_class2(:,1), circle_train_class2(:,2), 'bo')
    axis square 
    title('Training Set')
    legend('unit circle','class1','class2')

end


function data = generate_inside_circle_data(radius, number_of_data)
    r = radius*sqrt(rand(1,number_of_data));  
    theta = 2*pi*rand(1,number_of_data); 
    x = r.*cos(theta); 
    y = r.*sin(theta); 
    data = [x; y];
end

function data = generate_outside_circle_data(radius, number_of_data)
    r = radius*rand(1,number_of_data) + radius;  
    theta = 2*pi*rand(1,number_of_data); 
    x = r.*cos(theta); 
    y = r.*sin(theta); 
    data = [x; y];
end

%%
function [cluster_centers] = k_means(X,k)
    randidx = randperm(size(X, 1));
    cluster_centers = X(randidx(1:k),:);
    cluster_centers_prev = zeros(k,2);
    diff_centers = abs(cluster_centers - cluster_centers_prev);
    change_in_centers = sum(diff_centers(:));

    while(change_in_centers)
        idx = find_closest_centers(X,cluster_centers,k);
        cluster_centers_prev = cluster_centers;
        cluster_centers =  compute_centers(X,idx,k);
        diff_centers = abs(cluster_centers - cluster_centers_prev);
        change_in_centers = sum(diff_centers(:));
    end
end

function idx = find_closest_centers(X,cluster_centers,k)
    idx = zeros(size(X,1), 1);
    for i=1:size(X,1)
        dist = 10^6;
        for j=1:k
            inp=X(i,:);
            cent = cluster_centers(j,:);
            dist1=sum((inp-cent).^2);
            if(dist1<dist)
                idx(i)=j;
                dist=dist1;
            end
        end
    end
end

function cluster_centers = compute_centers(X,idx,k)
    [m,n] = size(X);
    cluster_centers = zeros(k, n);
    num=zeros(k,1);
    for i=1:k
        for j=1:m
            if(idx(j)==i)
                cluster_centers(i,:)=cluster_centers(i,:)+X(j,:);
                num(i)=num(i)+1;
            end
        end
    end

    for i=1:k
        cluster_centers(i,:)=cluster_centers(i,:)/num(i);
    end
end

%%
function accuracy = knn_cross_validation(n_folds, data, label, k)
% this function will do the n-folds cross validation 
% for knn classifier with the specific k value
% and return the average accuracy
% input:
%      n_folds - the number pf folds
%      data - the train data (no test data at this stage)
%      label - the train label
%      k - the k value for the knn algorithm
% outpur:
%      accutacy - the average accuracy of the cross validation of knn

n = length(data); % number of data
n_infold = round(n / n_folds); % number of data in each fold
sum_accuracy = 0;
    for i = 1: n_folds
        % find the start and end index
        start_idx = 1 + n_infold * (i - 1);
        if i == n_folds
            end_idx = n;
        else
            end_idx = n_infold * i;
        end
        % prepare the cv_test data and cv_train data
        cv_test = data(start_idx:end_idx, :);
        cv_test_label = label(start_idx:end_idx, :);
        cv_train = [data(1: (start_idx - 1),:); data((end_idx + 1) : n, :)];
        cv_train_label = [label(1: (start_idx - 1),:); label((end_idx + 1) : n, :)];
        % train the cv_train data
        % implemented knn function
        % cv_predict_label = knn(cv_train,cv_train_label,cv_test,cv_test_label,k);
        % matlab built-in knn function
        Factor = ClassificationKNN.fit(cv_train, cv_train_label, 'NumNeighbors', k);
        cv_predict_label = predict(Factor, cv_test);
        accuracy = length(find(cv_predict_label == cv_test_label)) / length(cv_test);
        sum_accuracy = sum_accuracy + accuracy;
    end
accuracy = sum_accuracy / n_folds;
end

function result = knn(train_data,train_label,test_data,test_label,k,distance_mark)
    % k-nearest-neighbor classifier
    %
    % input:
    %       train_data -> N * 2, test_data -> N * 2 are training data set and test data
    %       set,respectively.
    %       train_label -> N * 1,test_label -> N * 2 are column vectors.they are labels of training
    %       data set and test data set,respectively.
    %       k is the number of nearest neighbors
    %       distance_mark:   ['Euclidean', 'L2'| 'L1' | 'Cos'] 
    %       'Cos' represents Cosine distance.
    % output:
    %       predict_label -> N * 1
    %       or
    %       rate: accuracy of knn classifier

    if nargin < 5
        error('Not enought arguments!');
    elseif nargin < 6
        distance_mark='L2';
    end

    [n dim] = size(test_data);% number of test data set
    train_num = size(train_data, 1); % number of training data set
    u = unique(train_label); % class labels
    nclasses = length(u);%number of classes
    result = zeros(n, 1);
    count = zeros(nclasses, 1);
    dist = zeros(train_num,1);

    for i = 1:n
        % compute distances between test data and all training data
        % then sort
        test=test_data(i,:);
        for j=1:train_num
            train=train_data(j,:);V=test-train;
            switch distance_mark
                case {'Euclidean', 'L2'}
                    dist(j,1)=norm(V,2); % Euclead (L2) distance
                case 'L1'
                    dist(j,1)=norm(V,1); % L1 distance
                case 'Cos'
                    dist(j,1)=acos(test*train'/(norm(test,2)*norm(train,2))); % cos distance
                otherwise
                    dist(j,1)=norm(V,2); % default distance
            end
        end

        [Dummy Inds] = sort(dist);

        % compute the class labels of the k nearest samples
        count(:) = 0;
        for j = 1:k
            ind = find(train_label(Inds(j)) == u); %find the label of the j'th nearest neighbors 
            count(ind) = count(ind) + 1; % count:the number of each class of k nearest neighbors
        end

        % determine the class of the data sample
        [dummy ind] = max(count);
        result(i)   = u(ind);
    end
end

%%
function conf_mat = confusion_matrix(prediction,y_test)
    conf_mat = zeros(2,2);
    conf_mat(1,1) = sum(not(xor(prediction(1:1000),y_test(1:1000))));
    conf_mat(1,2) = sum(xor(prediction(1:1000),y_test(1:1000)));
    conf_mat(2,1) = sum(xor(prediction(1001:2000),y_test(1001:2000)));
    conf_mat(2,2) = sum(not(xor(prediction(1001:2000),y_test(1001:2000))));
end