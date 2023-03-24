import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def load():
    
    df = pd.read_csv("datasets/ParisHousing.csv")
    x = df.iloc[: , 0:-1]
    y = df.iloc[:, -1]
    print(x.head())

    # plt.scatter(df.index.values, df['price'])
    # plt.show()

    
    X = x.to_numpy()
    Y = y.to_numpy().reshape((10000,1))


    scaler = MinMaxScaler()
    model=scaler.fit(X)
    scaled_data=model.transform(X)

   
    return scaled_data, Y

if __name__ == '__main__':load()


