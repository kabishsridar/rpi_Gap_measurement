import matplotlib.pyplot as plt

# Read the file and normalize data
with open('ratios.csv', 'r') as fin:
    data = fin.read().replace('\n', ',')  # Replace newlines with commas
    values = data.strip().split(',')

# Clean values and convert to floats
values = [v.strip() for v in values if v.strip() != '']
values = [float(v) for v in values]

# Separate into mm and ratio
mm = values[0::2]     # even indices
ratio = values[1::2]  # odd indices

# Plot mm vs ratio
plt.plot(mm, ratio, marker='o', color='blue', label='Ratio vs mm')

plt.xlabel('Length (mm)')
plt.ylabel('Pixel-to-mm Ratio')
plt.title('Calibration Curve: Ratio vs mm')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
