from somlib.som import SOM, create_u_matrix
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
 
def main():
    standard_som()
    #test()

if __name__ == '__main__':
    main()
    
    
def standard_som():
   path = "data/dorothea_clean.csv"
   #path = "research/data/housing.data"
   table = pd.read_csv(path, header=None)
   table_new = table.dropna(axis=0)
   data = DataFrame.as_matrix(table_new)

   my_som = SOM(20, 20, 5000)
   lattice = my_som.calc(data)
   u_matrix = create_u_matrix(lattice)
   plt.matshow(u_matrix.T, fignum=100, cmap='viridis')
   plt.show()
   
def test():
   print ("Hello")