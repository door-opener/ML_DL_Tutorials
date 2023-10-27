import os, sys
import pandas as pd

def main():

    df_train = pd.read_csv('mnist-pngs/train.csv')
    print(df_train.head())

    df_test = pd.read_csv('mnist-pngs/test.csv')
    print(df_test.head())

    df_train = pd.read_csv('mnist-pngs/train.csv')

    # Shuffling the dataset, frac -> fraction of the dataset to shuffle #
    df_train = df_train.sample(frac=1, random_state=123)

    loc = round(df_train.shape[0]*0.9)
 
    # 90% of dataset for training, remainder for validation #
    df_new_train = df_train.iloc[:loc]
    df_new_val = df_train.iloc[loc:]

    df_new_train.to_csv('mnist-pngs/new_train.csv', index=None)
    df_new_val.to_csv('mnist-pngs/new_val.csv', index=None)
 
    print(df_new_train.head(20))
    print(df_new_val.head(20))

if __name__ == "__main__":
   main()
