import numpy as np

coords = np.load("data_analysis/coordinates.npy")

print("Loaded shape:", coords.shape)


#loop over personerne
    #loop over gæt (start ved 4 da de første 4 er automatiske)
        #få acquisition function til at give prob of improvement for alle punkter 
        #(softmax til at gøre til sandsynligheder)
        #acquisition skal tage udgangspunkt i de punkter mennesket har valgt indtil videre.
        #sammenlign med menneskescore

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

for trial in data:
    for i,point in enumerate(trial[4:]):
        x,y = point
        mu, std, ? = GP(trial[:i+1]) #opdater med den rigtige Gaussian process funktion
        aq_vals = expected_improvement(current_best, mu, std) #kør med den relavante aquisition funktion
        beta = 5 # prøv forskellige værdier
        sm = softmax(beta * aq_vals)
        sm[x][y] # nok ikke sådan det er indexed
        #gem denne sftmax et sted