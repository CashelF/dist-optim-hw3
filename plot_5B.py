import numpy as np
import matplotlib.pyplot as plt
from hw3_functions import parallel_sgd, local_sgd

def plot_loss(loss_parallel, loss_local_sgds):
    plt.plot(range(len(loss_parallel)), loss_parallel, label="Parallel Loss")
    # plt.plot(range(len(loss_parallels[1])), loss_parallels[1], label="Parallel Loss 5 Comm")
    # plt.plot(range(len(loss_parallels[2])), loss_parallels[2], label="Parallel Loss 20 Comm")

    plt.plot(range(len(loss_local_sgds[0])), loss_local_sgds[0], label="Local SGD Loss 1 Tau")
    plt.plot(range(len(loss_local_sgds[1])), loss_local_sgds[1], label="Local SGD Loss 5 Tau")
    plt.plot(range(len(loss_local_sgds[2])), loss_local_sgds[2], label="Local SGD Loss 20 Tau")

    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("dist_training_loss.jpg")

def main():
    data = np.load('hw3_synthetic_data.npz')
    X = data['X']
    y = data['y']

    theta0 = np.zeros(X.shape[1])
    loss_parallel, thetas_parallel = parallel_sgd(X, y, theta0, 10, 0.01, 4, 32, 42)
    # loss_parallel5, thetas_parallel5 = parallel_sgd(X, y, theta0, 5, 0.01, 4, 32, 42)
    # loss_parallel20, thetas_parallel20 = parallel_sgd(X, y, theta0, 20, 0.01, 4, 32, 42)

    loss_local_sgd1, thetas_local_sgd1 = local_sgd(X, y, theta0, 10, 0.01, 4, 32, 42, 1)
    loss_local_sgd5, thetas_local_sgd5 = local_sgd(X, y, theta0, 10, 0.01, 4, 32, 42, 5)
    loss_local_sgd20, thetas_local_sgd20 = local_sgd(X, y, theta0, 10, 0.01, 4, 32, 42, 20)

    plot_loss(loss_parallel, (loss_local_sgd1, loss_local_sgd5, loss_local_sgd20))

main()
