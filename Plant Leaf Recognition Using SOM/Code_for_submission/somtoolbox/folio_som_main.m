clear
clc

load folio_features

%% make the data
% numerical matrix D
D = features(:,2:size(features, 2));
% convert to data struct
sD = som_data_struct(D);

% string data labels
name = {'ashanti blood'; 'barbados cherry'; 'beaumier du perou'; 'betel';...
    'bitter orange'; 'caricature plant'; 'chinese guava'; 'chocolate tree';...
    'chrysanthemum'; 'coeur demoiselle'; 'coffee'; 'croton';...
    'duranta gold'; 'eggplant'; 'ficus'; 'fruitcitere';...
    'geranium'; 'guava'; 'hibiscus'; 'jackfruit';...
    'ketembilla'; 'lychee'; 'mulberry leaf'; 'papaya';...
    'pimento'; 'pomme jacquot'; 'rose'; 'star apple';...
    'sweet olive'; 'sweet potato'; 'thevetia'; 'vieux garcon';};
L = cell(size(features, 1), 1);
for i = 1:size(features, 1)
    L{i} = name{features(i,1)};
end
sD.labels = L;

% normalization
sD = som_normalize(sD, 'var');

%% make the SOM
sM = som_make(sD);
sM = som_autolabel(sM,sD,'vote');