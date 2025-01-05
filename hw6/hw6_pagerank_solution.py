import numpy as np

def pagerank(M, d=0.1, max_iterations=100, tol=1e-6):
    
    N = M.shape[0]
    x = np.ones(N) / N
    M_teleport = d / N + (1 - d) * M
    
    for i in range(max_iterations):
        x_new = np.dot(x, M_teleport)
        if np.linalg.norm(x_new - x, ord=1) < tol:
            break
        x = x_new
    
    return x

M1 = np.array([[0, 1, 1],
               [0, 0, 1],
               [0, 1, 0]])
M1 = M1 / M1.sum(axis=1, keepdims=True)

pagerank_result_1 = pagerank(M1, d=0.1)
print("PageRank1:", pagerank_result_1)


M2 = np.array([[0, 1, 1, 1, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])


M2 = np.where(M2.sum(axis=1, keepdims=True) != 0, 
              M2 / M2.sum(axis=1, keepdims=True), 
              1 / M2.shape[0])
pagerank_result_2 = pagerank(M2, d=0.1)
print("PageRank2:", pagerank_result_2)
