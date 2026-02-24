from load_human_data import xyz
import numpy as np


coordinates = []


for pair in range(len(xyz)):    
    coordinates.append((xyz[pair,:, :2]))  # Extract x and y coordinates
print(coordinates)