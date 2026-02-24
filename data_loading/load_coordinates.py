from load_human_data import xyz
import numpy as np

def load_coordinates():
    return xyz

coords = load_coordinates()

# Save NumPy-native (best)
np.save("data_analysis/coordinates.npy", coords)

print("Shape:", coords.shape)
print(coords[:10])   # show first samples