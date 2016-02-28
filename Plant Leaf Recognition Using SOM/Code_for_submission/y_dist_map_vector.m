function y_dist_map = y_dist_map_vector(bw, num_of_lines, length, width, start_column_idx, end_column_idx, start_row_idx, end_row_idx)

gap = floor(width / (num_of_lines + 1));
i = start_column_idx + gap;
y_dist_map = zeros(1,num_of_lines);

count = 1;
while count <= num_of_lines
    target_row = bw(:,i);
    %i
    %length
    line_dist = size(find(target_row == 1), 1);
    y_dist_map(count) = line_dist;
    count = count + 1;
    i = i + gap;
end

y_dist_map = y_dist_map / length;

