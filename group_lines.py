from collections import defaultdict
import numpy as np

def group_lines_by_y(lines, threshold=10):
    groups = defaultdict(list)
    
    for line0 in lines:
        # print(line)
        # print(line[0])

        line = line0[0]

        _, y1, _, y2 = line
        y_avg = (y1 + y2) // 2

        # Group nearby lines
        found_group = False
        for key in groups:
            if abs(key - y_avg) < threshold:
                groups[key].append(line)
                found_group = True
                break
        if not found_group:
            groups[y_avg].append(line)
    return groups

horizontal_lines = [[[ 77, 171, 297, 159]],
[[114, 220, 394, 210]],
[[ 78, 130, 408, 118]],
[[ 75, 167, 274, 160]],
[[ 79, 259, 298, 248]],
[[207, 161, 456, 152]],
[[389, 244, 525, 237]],
[[167, 167, 433, 154]],
[[399, 153, 457, 151]],
[[481, 206, 519, 205]],
[[358, 158, 456, 153]]]

# print(horizontal_lines)

line_groups = group_lines_by_y(horizontal_lines, threshold=10)

# print(line_groups)

# Take top 2 groups by line count or avg Y
group_ys = sorted(line_groups.keys())
top_two_lines = []

# print(group_ys)

for y in group_ys[:2]:  # top 2 horizontal lines
    lines = line_groups[y]
    all_y = [l[1] for l in lines] + [l[3] for l in lines]
    avg_y = int(np.mean(all_y))
    top_two_lines.append(avg_y)

if len(top_two_lines) < 2:
    print("Couldn't reliably find two lines.")
    exit()

# Pixel gap
pixel_gap = abs(top_two_lines[0] - top_two_lines[1])