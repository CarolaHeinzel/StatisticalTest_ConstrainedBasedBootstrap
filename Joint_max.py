import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet, binom
import numpy as np
import itertools
# simulate the allele frequencies
def create_p(M, K):
    p = np.random.uniform(size=(M, K))
    return p

# Simulate the Individuals
def create_sample_pqbekannt(M, K, p, q):
    x = np.zeros((M))
    loc = np.dot(q, p.T)
    #print(loc)
    for m in range(M):
        x[m] = binom.rvs(n=2, p=loc[m])
    return x 

#M = 10
#K = 2
#p = create_p(M, K)
#x = create_sample_pqbekannt(M, K, p, [0.1, 0.9])

#print(x)


# Simulate the individuals with unkonw q, i.e. 
#  max >= epsilon under H0 and  max < epsilon under H1
# correct
def create_sample_pbekannt(M, K, p, epsilon, booli):
    x = np.zeros(M)
    if(booli == 0): # H0 is the truth
        q  = dirichlet.rvs(alpha=np.ones(K))
        #print(q[0])
        # max > epsilon
        while(max(q[0]) < epsilon):
            q = dirichlet.rvs(alpha=np.ones(K))

    else: # H1 is the truth
        q  = dirichlet.rvs(alpha=np.ones(K))
        # max < epsilon
        while(max(q[0]) >= epsilon):
            q = dirichlet.rvs(alpha=np.ones(K)) 
    for m in range(M):
        
        loc = np.dot(q, p[m,:])
        x[m] = np.random.binomial(2, loc)[0]
    return q, x


# likelihood in the Admixture Model
# correct
def l(q,x, p):
    K, M = p.shape
    res1 = 0
    #for j in range(J):
    #q = list(q) + [q_K]    
    #q = [q, q_K]
    #print(q)
    for m in range(M):
        loc = np.dot(q, p[m,:])
        x_temp = x[m] 
        res1 += x_temp * np.log(loc) + (2-x_temp)*np.log(1-loc)
    return -res1

# constraints for the minimization
def constraint1(q):
    return -np.sum(q)  + 1  

def constraint_q2(q, e):
    return max(q)  - e 


# Calculates the MLE under H0
def get_admixture_proportions_H0(x, p, K, e):
    q0 = np.random.rand(K)
    q0 /= np.sum(q0)  

    #print(q0)
    b = [(0, 1) for _ in range(K)]  # Bounds für jede Komponente von q
    cons = (
        {'type': 'ineq', 'fun': constraint1}#,# Sum to 1
       # {'type': 'ineq', 'fun': lambda q: constraint_q2(q, e)} # Maximum bigger than e, i..e under H0
        )
    result = minimize(l, q0, args=(x, p), constraints=cons, bounds = b) #)
    return result.x

# Example application
M = 100
K = 4
p = create_p(M, K)
x = create_sample_pqbekannt(M, K, p, [0.1, 0.65, 0.2, 0.05])
#test = get_admixture_proportions_H0(x, p, K, 0.1)
#print(test)
#print(l(test, x, p))
#%%

# Maximization for the whole parameter space
def get_admixture_proportions(x, p, tol=1e-6):
    K, M = p.shape
    res = dirichlet.rvs(alpha=np.ones(K))
    err = 1
    while err > tol:
        loc = fun2(res, p, x)
        err = np.sum(np.abs(res - loc))
        res = loc
    return res

def fun2(q, p, loc_x):
    K, M = p.shape
    E = np.zeros((K, M))
    loc = np.dot(q, p)
    loc[loc==0] = 1e-16
    loc[loc==1] = 1-1e-16
    for k in range(K):
        E[k, :] = (loc_x * p[k, :] / loc + (2 - loc_x) * (1 - p[k, :]) / (1 - loc))
    res = np.sum(E, axis=1) / M * q / 2
    return res / np.sum(res)

#test = get_admixture_proportions(x, p.T)
#print(test)
#print(l(test[0], x, p))

#%%
# Problem: The maximization does not work properly!!!

# Soluation: grid search and use whole parameter space, if applicable
def generate_reduced_simplex(K, step, mass):
    """Generates points on (K)-dimensional simplex that sum to `mass`."""
    result = []

    def recurse(remaining, depth, current):
        if depth == K - 1:
            val = remaining
            if 0 <= val <= 1:
                result.append(current + [val])
            return
        for i in range(int(remaining / step) + 1):
            val = i * step
            recurse(remaining - val, depth + 1, current + [val])

    recurse(mass, 0, [])
    return result

def grid_search_l(l, x, p, K, step=0.01, epsilon=0.9):
    """Efficient grid search over the simplex with max(q) == epsilon."""
    best_q = None
    best_val = np.inf

    # For each position where epsilon is inserted
    for i in range(K):
        # Create all (K-1)-dimensional vectors summing to (1 - epsilon)
        sub_vectors = generate_reduced_simplex(K - 1, step, 1 - epsilon)
        print(sub_vectors)
        for vec in sub_vectors:
            q = vec[:i] + [epsilon] + vec[i:]
            q = np.array(q)
            val = l(q, x, p)
            if val < best_val:
                best_val = val
                best_q = q.copy()

    return best_q, best_val

K = 3
M = 1000
p = create_p(M, K)
x = create_sample_pqbekannt(M, K, p, [0.1, 0.2, 0.7])
#best_q, best_val = grid_search_l(l, x, p, K)
#print("Best q:", best_q)
#print("Max value:", best_val)

def combined(x, p, K, e):
    
    estimator_all = get_admixture_proportions(x, p.T)
    # applicable 
    #print(estimator_all)
    if max(estimator_all[0]) >= e:
        return estimator_all[0]
    else:
        # Grid Search
        q, max_val = grid_search_l(l, x, p, K, step=0.01, epsilon=e)
        return q


#test_combined = combined(x, p, 3, 0.9)
#print(test_combined)
