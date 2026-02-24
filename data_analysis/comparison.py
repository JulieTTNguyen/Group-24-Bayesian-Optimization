import numpy as np

coords = np.load("data_analysis/coordinates.npy")

print("Loaded shape:", coords.shape)


#loop over personerne
    #loop over gæt (start ved 4 da de første 4 er automatiske)
        #få acquisition function til at give prob of improvement for alle punkter 
        #(softmax til at gøre til sandsynligheder)
        #acquisition skal tage udgangspunkt i de punkter mennesket har valgt indtil videre.
        #sammenlign med menneskescore

