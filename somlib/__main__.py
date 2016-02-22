from somlib.som import SOM, create_u_matrix
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

def main():
    path = "data/dorothea_clean.csv"
    table = pd.read_csv(path, header=None)
    data = DataFrame.as_matrix(table.dropna(axis=0))
    
    my_som = SOM(10, 10, 2000)
    lattice = my_som.calc(data)
    u_matrix = create_u_matrix(lattice)
    plt.matshow(u_matrix.T, fignum=100, cmap='viridis')
    plt.show()
