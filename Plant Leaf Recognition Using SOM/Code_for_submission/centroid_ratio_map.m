function centroid_ratio = centroid_ratio_map(bw, leaf_min_x, leaf_max_x, leaf_min_y, leaf_max_y, leaf_width, leaf_length, leaf_center_x, leaf_center_y)

p1 = [leaf_min_x, leaf_min_y];
p2 = [leaf_min_x, leaf_min_y + 0.25 * leaf_length];
p3 = [leaf_min_x, leaf_min_y + 0.5 * leaf_length];
p4 = [leaf_min_x, leaf_min_y + 0.75 * leaf_length];
p5 = [leaf_min_x, leaf_max_y];
 
p6 = [leaf_min_x + 0.25 * leaf_width, leaf_min_y];
p7 = [leaf_min_x + 0.25 * leaf_width, leaf_max_y];
 
p8 = [leaf_center_x, leaf_min_y];
p9 = [leaf_center_x, leaf_max_y];
 
p10 = [leaf_min_x + 0.75 * leaf_width, leaf_min_y];
p11 = [leaf_min_x + 0.75 * leaf_width, leaf_max_y];
 
p12 = [leaf_max_x, leaf_min_y];
p13 = [leaf_max_x, leaf_min_y + 0.25 * leaf_length];
p14 = [leaf_max_x, leaf_min_y + 0.5 * leaf_length];
p15 = [leaf_max_x, leaf_min_y + 0.75 * leaf_length];
p16 = [leaf_max_x, leaf_max_y];
p = [p1;p2;p3;p4;p5;p6;p7;p8;p9;p10;p11;p12;p13;p14;p15;p16];
 
 
dist_inside = zeros(1,16);
dist_whole = zeros(1,16);
count = 1;
for i = 1:16
    center = [leaf_center_x, leaf_center_y];
    point = p(i,:);
    
    if center(1) > point(1)
        a = (center(2) - point(2)) / (center(1) - point(1));
        b = (center(1) * point(2) - point(1) * center(2)) / (center(1) - point(1));
        start_x = point(1);
        end_x = center(1);
        x = start_x;
        while x <= end_x
            y = round(a * x + b);
            if bw(y, round(x)) == 1               
                point_cross = [x, y];
                break
            end
            x = x + 1;
        end
    elseif center(1) < point(1)
        a = (center(2) - point(2)) / (center(1) - point(1));
        b = (center(1) * point(2) - point(1) * center(2)) / (center(1) - point(1));
        start_x = center(1);
        end_x = point(1);
        x = start_x;
        while x <= end_x
            y = round(a * x + b);
            if bw(y, round(x)) == 0 || round(x) == end_x
                point_cross = [x, y];
                break
            end
            x = x + 1;
        end
    else
        if point(2) < center(2)
            start_y = round(point(2));
            end_y = round(center(2));
            y = start_y;
            while y <= end_y && bw(y, round(center(1))) == 0
                y = y + 1;
            end
            d_inside = abs(y - center(2));
            dist_inside(count) = d_inside;
            d_whole = 0.5 * leaf_length;
            dist_whole(count) = d_whole;
        end
        if point(2) > center(2)
            start_y = floor(center(2));
            end_y = floor(point(2));
            y = start_y;
            while y <= end_y && bw(y, round(center(1))) == 1
                y = y + 1;
            end
            d_inside = abs(y - 1 - center(2));
            dist_inside(count) = d_inside;
            d_whole = 0.5 * leaf_length;
            dist_whole(count) = d_whole;
        end
        count = count + 1;
        continue
    end
    d_inside = sqrt((point_cross(1) - center(1))^2 + (point_cross(2) - center(2))^2);
    d_whole = sqrt((point(1) - center(1))^2 + (point(2) - center(2))^2);
    dist_inside(count) = d_inside;
    dist_whole(count) = d_whole;
    count = count + 1;
end

centroid_ratio = dist_inside ./ dist_whole;