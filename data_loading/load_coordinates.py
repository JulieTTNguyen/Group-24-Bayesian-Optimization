from load_human_data import xyz
import numpy as np

print(np.shape(xyz))  # Check the shape of the loaded data
coordinates = []

for pair in xyz:
    coordinates.append(xyz[pair,:, :2])  # Extract x and y coordinates

print(coordinates)
