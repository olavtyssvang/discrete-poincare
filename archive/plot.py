import matplotlib.pyplot as plt
import csv


data_naive = []
data_local = []

with open("output/discrete-poincare/Ned-naive.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    data_naive = [row for row in reader]

with open("output/discrete-poincare/Ned-local.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    data_local = [row for row in reader]


for i in range(1, len(data_naive)):
    plt.plot(data_naive[i])

for i in range(1, len(data_naive)):
    plt.plot(data_local[i])

plt.show()