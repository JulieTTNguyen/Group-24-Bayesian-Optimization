from load_human_data import xyz
import numpy as np



def load_coordinates():
    coordinates = []
    data = xyz  # Use the loaded xyz data
    for pair in range(len(data)):
        coordinates.append((data[pair,:, :2]))  # Extract x and y coordinates
    return np.array(coordinates)  # Convert to array

#save as csv 
np.savetxt("data_analysis/coordinates.csv", load_coordinates().reshape(-1, 2), delimiter=",")

