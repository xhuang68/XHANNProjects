function [wavelet_features] = extract_wavelet_features(raw_features,num_features)

wavelet_coefficients = zeros(size(raw_features));
for i = 1:size(wavelet_coefficients,1)
    wavelet_coefficients(i,:) = wavedec(raw_features(i,:),4,'haar');
end

% Lillifors test
ks_values = zeros(size(wavelet_coefficients,2),1);
for i = 1:size(wavelet_coefficients,2);
    [~,~,ks_values(i),~] = lillietest(wavelet_coefficients(:,i));
    [~, indices] = sort(ks_values, 'descend');
end
wavelet_features = wavelet_coefficients(:,indices(1:num_features));