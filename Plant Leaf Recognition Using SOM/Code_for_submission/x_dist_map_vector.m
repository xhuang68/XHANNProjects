function x_dist_map = x_dist_map_vector(bw, num_of_lines, length, width, start_row_idx, end_row_idx, start_col_idx, end_col_idx)

gap = floor(length / (num_of_lines + 1));
i = start_row_idx + gap;
x_dist_map = zeros(1,num_of_lines);

count = 1;
while count <= num_of_lines
    target_row = bw(i,:);
    % i
    % width
    line_dist = size(find(target_row == 1), 2);
    x_dist_map(count) = line_dist;
    count = count + 1;
    i = i + gap;
end

x_dist_map = x_dist_map / width;

