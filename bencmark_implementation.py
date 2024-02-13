import os
import numpy as np
import sys


def gen_data(k=3, dim=2, points_per_cluster=200, lim=[-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    Will also plot the points in case of 2D.
    input:
        - k: Number of Gaussian clusters
        - dim: Dimension of generated points
        - points_per_cluster: Number of points to be generated for each cluster
        - lim: Range of mean values
    output:
        - X: Generated points (points_per_cluster*k, dim)
    '''
    x = []
    mean = np.random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    for i in range(k):
        cov = np.random.rand(dim, dim+10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
    x = np.array(x)
    return x


target_dir = sys.argv[1]
file_name, out_name, means, sigma, time_fname = "benchmark_data.csv", "mixture_models", "means", "sigma", sys.argv[
    2]
n_components, D = 3, 2
num_iters = 100
os.system(f"cd {target_dir}; make --always-make;")
os.system(f"touch {time_fname} && rm {time_fname}")
os.system(f"touch {file_name} && rm {file_name}")
os.system("pwd")

for i in range(6, 15):
    # size = gen_data_to_file(n_components, D, points_per_cluster = 2 ** i, filename=file_name)
    os.system(f"touch {file_name} && rm {file_name}")
    points_per_cluster = 2 ** i
    size = 0
    with open(file_name, 'a') as f:
        while points_per_cluster > 0:
            X = gen_data(n_components, D,
                         points_per_cluster=points_per_cluster)
            np.savetxt(f, X, delimiter=",")
            points_per_cluster -= 2 ** 5
            size += X.shape[0]

    command = f"./{target_dir}/{out_name} {file_name} {n_components} {size} {D} {means} {sigma} {num_iters} {time_fname}"
    print(command)
    os.system(command)
    print(np.loadtxt(time_fname, delimiter=",")[-1])
