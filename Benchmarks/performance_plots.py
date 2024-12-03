import pandas as pd
import matplotlib.pyplot as plt

def plot_benchmarks():
    data = pd.read_csv("benchmarks/results.csv", names=["Mode", "Runtime"])
    grouped = data.groupby("Mode")["Runtime"].mean()

    grouped.plot(kind="bar", xlabel="Mode", ylabel="Runtime (seconds)",
                 title="Performance Comparison", rot=0)
    plt.show()

if __name__ == "__main__":
    plot_benchmarks()
