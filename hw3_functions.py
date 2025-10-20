import numpy as np


############### 5A ##############


def generate_W_matrix(n, alpha):
    """
    Generate the weight matrix W for a ring network with n nodes and parameter alpha.

    Args
    ----
    n : Size of the graph
    alpha : Weight parameter 

    
    """
    W = np.zeros((n,n))
    for i in range(n):
        W[i,i] = 1-2*alpha
        W[i, (i+1)%n] = alpha
        W[i, (i-1)%n] = alpha
    return W


def synchronous_consensus(x0, steps, W):
    """
    Run synchronous consensus on a ring for a given number of iterations.
    Returns an array X of shape (steps+1, n) containing x(0), x(1), ..., x(steps).

    Args
    ----
    x0 : (n,) numpy array for the initial values
    steps: number of iterations
    W : (n,n) weight matrix
    """
    consensus_steps = np.zeros((steps+1, x0.shape[0]))
    consensus_steps[0,] = x0
    for step in range(steps):
        consensus_steps[step+1, ] = W @ consensus_steps[step, ]
    return consensus_steps

    
def randomized_gossip(x0, steps, seed):
    """
    Run randomized gossip on a ring for a given number of iterations.
    Returns an array X of shape (steps+1, n) containing x(0), x(1), ..., x(steps).

    Args
    ----
    x0 : (n,) numpy array for the initial values
    steps: number of iterations
    seed : a random seed for code checking
    """
    np.random.seed(seed)

    gossip_steps = np.zeros((steps+1, x0.shape[0]))
    gossip_steps[0, ] = x0
    for step in range(steps):
        random_edge = np.random.random_integers(0, x0.shape[0]-1)
        gossip_steps[step+1, ] = gossip_steps[step, ].copy()
        avg = (gossip_steps[step, (random_edge + 1) % x0.shape[0]] + gossip_steps[step, (random_edge - 1) % x0.shape[0]]) / 2
        gossip_steps[step+1, (random_edge + 1) % x0.shape[0]] = avg
        gossip_steps[step+1, (random_edge) % x0.shape[0]] = avg
    return gossip_steps


############### 5B ##############

def sigmoid(z):
    # sigmoid
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z, dtype=float)
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1 + ez)
    return out

def bce_loss(theta, X, y):
    """
    Binary cross-entropy loss averaged over samples.
    y must be in {0,1}.
    """
    z = X @ theta
    loss = - y @ z + np.log1p(np.exp(z)).sum()
    return loss / len(X)
    

def bce_grad(theta, X, y):
    """
    Gradient of the average BCE wrt theta for logistic regression.
    """
    z = X @ theta
    p = sigmoid(z)
    # average gradient
    return (X.T @ (p - y)) / len(X)




def parallel_sgd(X, y, theta0, T, eta, K, batch_size, seed):
    """
    Parallel (synchronous) mini-batch SGD with K workers.

    The data split part is provided for you.

    Args
    ----
    X : (N, d) ndarray
    y : (N,) ndarray in {0,1}
    theta0 : (d,) initial parameter
    T : number of communication rounds (global steps)
    eta : learning rate
    K : number of workers
    batch_size : per-worker mini-batch size
    seed : RNG seed
    
    Returns
    -------
    A tuple of (loss_hist, theta_hist) where
      'loss_hist' : (T+1,) BCE over the full dataset per round
      'theta_hist' : (T+1, d) parameter trajectory
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    N, d = X.shape

    # Split the data into K shards 
    shard_idx = np.array_split(np.arange(N), K)
    shard_X = [X[idx] for idx in shard_idx]
    shard_y = [y[idx] for idx in shard_idx]

    loss_hist = np.empty((T+1))
    theta_hist = np.empty((T+1, d))
    theta = theta0.copy()

    loss_hist[0] = bce_loss(theta, X, y)
    theta_hist[0] = theta.copy()

    for t in range(1, T + 1):
        grads = []
        for k in range(K):
            inds = shard_idx[k]
            x_sample = rng.integers(len(inds), size=batch_size)

            x_b = shard_X[k][x_sample]
            y_b = shard_y[k][x_sample]

            grad_b = bce_grad(theta, x_b, y_b)
            grads.append(grad_b)

        grad_t = np.mean(grads, axis=0)
        theta = theta - eta*grad_t

        loss_hist[t] = bce_loss(theta, X, y)
        theta_hist[t] = theta.copy()

    return loss_hist, theta_hist



def local_sgd(X, y, theta0, T, eta, K, batch_size, seed, tau):
    """
    Local SGD with K workers and averaging period `tau`.

    Each communication round r = 1..T:
      - Initialize each worker's local model w_k := current global theta
      - Worker k performs `tau` local SGD steps on its own shard (no communication)
      - Average models
      - Record full-data BCE loss

    Args
    ----
    X : (N, d) ndarray
    y : (N,) ndarray in {0,1}
    theta0 : (d,) initial parameter
    T : number of **communication rounds**
    eta : learning rate
    K : number of workers
    batch_size : per-worker mini-batch size
    seed : RNG seed
    tau : int, number of **local** steps between global averaging

    Returns
    -------
    A tuple of (loss_hist, theta_hist) where
      loss_hist  : (T+1,) BCE over the full dataset per communication round
      theta_hist : (T+1, d) parameter trajectory (after each averaging)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    N, d = X.shape

    # Split the data into K shards 
    shard_idx = np.array_split(np.arange(N), K)
    shard_X = [X[idx] for idx in shard_idx]
    shard_y = [y[idx] for idx in shard_idx]

    loss_hist = np.empty((T+1))
    theta_hist = np.empty((T+1, d))
    theta = theta0.copy()

    loss_hist[0] = bce_loss(theta, X, y)
    theta_hist[0] = theta0.copy()

    for t in range(1, T + 1):
        thetas = [theta.copy() for _ in range(K)]
        for update in range(tau):
            for k in range(K):
                inds = shard_idx[k]
                sample_inds = rng.integers(len(inds), size=batch_size)

                x_b = shard_X[k][sample_inds]
                y_b = shard_y[k][sample_inds]

                grads_b = bce_grad(thetas[k], x_b, y_b)
                thetas[k] -= eta * grads_b
            
        theta = np.mean(thetas, axis=0)
        loss_hist[t] = bce_loss(theta, X, y)
        theta_hist[t] = theta.copy()
    
    return loss_hist, theta_hist

