from examples.demo_poisson import run_poisson_example
import numpy as np
import sys
import csv
from datetime import datetime
import os

demo_poisson_times = []

num_iters = int(sys.argv[1])
print(num_iters)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"poisson_demo_benchmarking_results_{timestamp}.csv"
filepath = os.path.join("benchmarking/benchmarking-results/", filename)

with open(filepath, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Test number", "Points per dimension (square)", "Mean execution time", "Stdev execution time"])
    for j in range(0, len(sys.argv) - 2):
        for i in range(0, num_iters):
            demo_poisson_times.append(run_poisson_example(int(sys.argv[j + 2])))
        demo_poisson_times_np = np.array(demo_poisson_times)
        demo_poisson_times.clear()
        writer.writerow([j, sys.argv[j + 1], np.mean(demo_poisson_times_np), np.std(demo_poisson_times_np)])