function img_features = feature_extraction_new(img,label)

% img=imresize(im,[256,144]);

chist = colorhist(img);

% Convert RGB image to gray
grayimg=rgb2gray(img);

%Global image threshold using Otsu's method
level = graythresh(grayimg);
BW = im2bw(grayimg,level);

%Performing Opening (Erosion and Dilution) Operations
SE=strel('diamond',4);
BW1 = imerode(BW,SE);
BW2 = imdilate(BW1,SE);

%Inverse Thresholding
BW3=1-BW2;

%Edge Detection & Filtering
BW4=edge(BW2,'roberts');
BW5 = bwareaopen(BW4,40); 
 
%Convex Hull Formation
BW6 = bwconvhull(BW5);

%% rewrite col 2-6 features
% Boundry and other calculations for original processed Leaf image
[leaf_min_x, leaf_max_x, leaf_min_y, leaf_max_y, leaf_width, leaf_length, leaf_center_x, leaf_center_y] = leaf(BW6);
img_area = size(BW6, 1) * size(BW6, 2); % image area
leaf_area = leaf_width * leaf_length; % leaf area(rectangle)
leaf_perimeter = 2 * (leaf_width + leaf_length); % leaf preimeter(rectangle)
% SS = size(find(BW6 == 1),1);
k=regionprops(BW6,'Area'); % here k is a structure
hull_area = k.Area; % hull area
hull_primeter = length(find(bwperim(BW6,4)==1)); % hull primeter
% bwperim is used to count the number of 1s around the connected graph 

%Final Features col 2-6
AR = leaf_width / leaf_length; %Aspect Ratio for Leaf
WAR = hull_area / leaf_area; %White Area Ratio for Leaf
P2A = leaf_perimeter / leaf_area; %Perimeter to Area
P2H = hull_primeter / leaf_perimeter;%Perimeter to Hull
HAR = leaf_area / hull_area; %Hull Area Ratio

%% calculate col 7,8,9
xdist_map = x_dist_map_vector(BW6, 10, leaf_length, leaf_width, leaf_min_y, leaf_max_y, leaf_min_x, leaf_max_x);
% x_dist_map = x_dist_map_vector(bw, num_of_lines, length, width, start_row_idx, end_row_idx, start_col_idx, end_col_idx)
ydist_map = y_dist_map_vector(BW6, 10, leaf_length, leaf_width, leaf_min_x, leaf_max_x, leaf_min_y, leaf_max_y);
% y_dist_map = y_dist_map_vector(bw, num_of_lines, length, width, start_column_idx, end_column_idx, start_row_idx, end_row_idx)

centroid_ratio = centroid_ratio_map(BW6, leaf_min_x, leaf_max_x, leaf_min_y, leaf_max_y, leaf_width, leaf_length, leaf_center_x, leaf_center_y);


% img_crop = img(leaf_min_y:leaf_max_y, leaf_min_x:leaf_max_x,:);
% histo = rgbhist_fast(img_crop,3,2);
% color_histogram = histo(1:end-1);

%% fill the cols
img_features = zeros(1, 58);
img_features(1) = label;
img_features(2) = AR;
img_features(3) = WAR;
img_features(4) = P2A;
img_features(5) = P2H;
img_features(6) = HAR;
img_features(7:16) = xdist_map;
img_features(17:26) = ydist_map;
img_features(27:42) = centroid_ratio;
img_features(43:58) = chist;
