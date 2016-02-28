function [leaf_min_x, leaf_max_x, leaf_min_y, leaf_max_y, leaf_width, leaf_length, leaf_center_x, leaf_center_y] = leaf(bw)

[length, width] = size(bw);

leaf_width_row = max(bw);
i = 1;
while i <= width && leaf_width_row(i) == 0
    i = i + 1;
end
leaf_min_x = i;
while i <= width && leaf_width_row(i) == 1
    i = i + 1;
end
leaf_max_x = i - 1;
leaf_width = leaf_max_x - leaf_min_x + 1;
leaf_center_x = (leaf_min_x + leaf_max_x) / 2;

leaf_length_col = max(bw');
i = 1;
while i <= length && leaf_length_col(i) == 0
    i = i + 1;
end
leaf_min_y = i;
while i <= length && leaf_length_col(i) == 1
    i = i + 1;
end
leaf_max_y = i - 1;
leaf_length = leaf_max_y - leaf_min_y + 1;
leaf_center_y = (leaf_min_y + leaf_max_y) / 2;