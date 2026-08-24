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
    data_naive[i] = [x/5 for x in data_naive[i]]
    data_local[i] = [x/5 for x in data_local[i]]


with open("output/discrete-poincare/Ned-naive_cor.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_naive)

with open("output/discrete-poincare/Ned-local_cor.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_local)