import pandas as pd
import matplotlib.pyplot as plt

def plot_train_history(path):
    hist_df: pd.DataFrame = pd.read_csv(path)
    print(hist_df)
    # TODO

def plot_hyperparameter_search(path):
    results_df: pd.DataFrame = pd.read_csv(path)
    print(results_df)
    # TODO
