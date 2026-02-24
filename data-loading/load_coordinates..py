from data-loading\load_human_data - Copy import xyz, params, meta
import numpy as np


print(xyz)
print(params)
print(meta)



def get_coordinates():
    with open("coordinates.txt", "r") as file:
        coordinates = []
        for line in file:
            lat, lon = map(float, line.strip().split(","))
            coordinates.append((lat, lon))
    return coordinates