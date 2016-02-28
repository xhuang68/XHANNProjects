close all;
clear;
clc;

basePath='Folio Leaf Dataset/folio/'; % the data set file - 'Folio' path
name = {'ashanti blood'; 'barbados cherry'; 'beaumier du perou'; 'betel';...
    'bitter orange'; 'caricature plant'; 'chinese guava'; 'chocolate tree';...
    'chrysanthemum'; 'coeur demoiselle'; 'coffee'; 'croton';...
    'duranta gold'; 'eggplant'; 'ficus'; 'fruitcitere';...
    'geranium'; 'guava'; 'hibiscus'; 'jackfruit';...
    'ketembilla'; 'lychee'; 'mulberry leaf'; 'papaya';...
    'pimento'; 'pomme jacquot'; 'rose'; 'star apple';...
    'sweet olive'; 'sweet potato'; 'thevetia'; 'vieux garcon';};

% compute total number of images
num_of_images = 0;
for i = 1:size(name,1)
    single_file_path = fullfile(basePath, cell2mat(name(i)));
    imgs_in_one_file = dir(fullfile(single_file_path,'*.jpg'));
    num_of_images = num_of_images + size(imgs_in_one_file, 1);
    
end

% construct the features matrix
features=zeros(num_of_images,58);
index = 1;

for i = 1:size(name,1)
    single_file_path = fullfile(basePath, cell2mat(name(i)));
    imgs_in_one_file = dir(fullfile(single_file_path,'*.jpg'));
    for j = 1:size(imgs_in_one_file, 1)
        filename=fullfile(single_file_path,imgs_in_one_file(j).name);
        im=imread(filename);
        % feature extraction
        img_features = feature_extraction_new(im, i);
        features(index,:) = img_features;
        index = index + 1;
    end
end

save('features', 'features');
som_leaf;