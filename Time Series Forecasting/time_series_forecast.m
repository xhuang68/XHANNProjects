%% IMPORTANT !!!
%% In order to call this function, the test data must already be present in the workspace and the function must be called with that variable name as the argument. Note that 'data' should be an array
%% Use this line to call the function :
%% [prediction_mlp, prediction_ensemble, MSE_mlp, MSE_mlp_percentage, MSE_ensemble, MSE_ensemble_percentage] = time_series_forecast(data);

function [prediction_mlp, prediction_ensemble, MSE_mlp, MSE_mlp_percentage, MSE_ensemble, MSE_ensemble_percentage] = time_series_forecast(data)

load train_data.mat

%Testing Data
if size(data,1)>1
    data = data';
end

test_data = {};
for i = 1:length(data)
    test_data{i} = data(i);
end

%% Multilayer perceptron

%Training

trainFcn = 'trainlm';
feedbackDelays = 1:2;
hiddenLayerSize = 10;

net = narnet(feedbackDelays,hiddenLayerSize,'open',trainFcn);

[x,xi,ai,t] = preparets(net,{},{},train_data);

net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;

[net,tr] = train(net,x,t,xi,ai);
view(net)
% Testing
[x,xi,ai,t] = preparets(net,{},{},test_data);
y = net(x,xi,ai);
y = cell2mat([xi,y]);
t = cell2mat([xi,t]);
prediction_mlp = y;
p = [40   221   672   504];
set(0, 'DefaultFigurePosition', p);
plot(y, 'b');
hold on
plot(t, 'g');
title('Time Series Forecasting for Test Data Using a Single Multilayer Perceptron');
legend('MLP Output', 'Ground Truth');

e = gsubtract(t,y);
MSE_mlp = sum(e.^2);
% fprintf('Mean square error in prediction using a single multilayer perceptron = %f\n', MSE_mlp)
MSE_mlp_percentage = MSE_mlp*100/sum(t.^2);
fprintf('Mean square error in prediction (as a percentage of ground truth) using a single multilayer perceptron = %f%%\n', MSE_mlp_percentage)
clear y
clear t

%% Ensemble Learning - Bootstrap Aggregating (Bagging) using multiple models
% of the multi-layer perceptron

num_models = 36; % Number of models in the ensemble

trainFcn1 = 'trainlm'; % Backpropagation Variant 'lm' - Levenberg-Marquardt 
trainFcn2 = 'trainbr'; % Backpropagation Variant 'br' - Bayesian Regularization

feedbackDelays1 = 1:2; % Number of past inputs used for prediction
feedbackDelays2 = 1:3; 
feedbackDelays3 = 1:4;

hiddenLayerSize1 = 10; % Number of hidden neurons
hiddenLayerSize2 = 12;

bag = zeros(1,length(train_data));
model_num = 1;
p = [1221 202 672 504];
set(0, 'DefaultFigurePosition', p);
figHandle = figure;
for i=1:2               % loop over different training functions
    for j = 1:3         % loop over different number of delays
        for k = 1:2     % loop over different hidden layer sizes
            for l = 1:3 % Training with multiple initializations of weighrs
    
                %Training
                fprintf('\n Training model number %s/36 \n', num2str(model_num));
                model_num = model_num + 1;
                
                switch i % Setting training function 
                    case 1
                        trainFcn = trainFcn1;
                    case 2
                        trainFcn = trainFcn2;
                end
                
                switch j % Setting number of delays 
                    case 1
                        feedbackDelays = feedbackDelays1;
                    case 2
                        feedbackDelays = feedbackDelays2;
                    case 3
                        feedbackDelays = feedbackDelays3;
                end
                
                switch k % Setting number of hidden neurons
                    case 1
                        hiddenLayerSize = hiddenLayerSize1;
                    case 2
                        hiddenLayerSize = hiddenLayerSize2;
                end
                
               
                net = narnet(feedbackDelays,hiddenLayerSize,'open',trainFcn);

                [x,xi,ai,t] = preparets(net,{},{},train_data);

                net.divideParam.trainRatio = 80/100;
                net.divideParam.valRatio = 20/100;
                net.divideParam.testRatio = 0/100;

                [net,tr] = train(net,x,t,xi,ai);
                
                %Testing

                [x,xi,ai,t] = preparets(net,{},{},test_data);
                y = net(x,xi,ai);
                y = [xi,y];
                t = [xi,t];
                figure(figHandle);
                hand1 = plot(cell2mat(y), 'b');
                hold on
                bag = bag + cell2mat(y);
                clear y
                clear t
            end 
        end
    end
end
[x,xi,ai,t] = preparets(net,{},{},test_data);
t = cell2mat([xi,t]);
bag = bag/num_models; %Using the mean rule for ensemble combination
prediction_ensemble = bag;
e = gsubtract(t,bag);
MSE_ensemble = sum(e.^2);
% fprintf('Mean square error in prediction using an ensemble of multilayer perceptrons = %f\n', MSE_ensemble)
MSE_ensemble_percentage = MSE_ensemble*100/sum(t.^2);
fprintf('Mean square error in prediction (as a percentage of ground truth) using an ensemble of multilayer perceptrons = %f%%\n', MSE_ensemble_percentage)
hand2 = plot(t, 'g', 'LineWidth', 2);
hand3 = plot(bag, 'r', 'LineWidth', 2);
legend([hand1, hand2, hand3],'Output of a Single Model','Ensemble Output' ,'Ground Truth');
title('Time Series Forecasting for Test Data Using an Ensemble of Multilayer Perceptrons');