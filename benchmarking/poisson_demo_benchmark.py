from examples.demo_poisson import run_poisson_example
import numpy as np

demo_poisson_times = []
for i in range(0, 10):
    demo_poisson_times.append(run_poisson_example())
demo_poisson_times = np.array(demo_poisson_times)
print("Mean exeution time:", np.mean(demo_poisson_times))
print("Std dev execution time:", np.std(demo_poisson_times))