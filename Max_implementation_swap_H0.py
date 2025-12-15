import numpy as np
from itertools import permutations
from scipy.stats import dirichlet, binom

#  Calculates the CBB Test for the Maximuum


# simulate the allele frequencies
def create_p(M, K):
    p = np.random.uniform(size=(M, K))
    #alpha = np.full(K, 0.1)  # Dirichlet concentration parameters
    #return np.random.dirichlet(alpha, size=M)
    return p

# Simulate the Individuals
def create_sample_pqbekannt(M, K, p, q):
    x = np.zeros((M))
    loc = np.dot(q, p.T)
    for m in range(M):
        x[m] = binom.rvs(n=2, p=loc[m])
    return x 

# Sample the ancestry wrt H0, i.e. max(q) = pa
def sample_array(K, pa):

    remaining_sum = 1.0 - pa
    values = []

    for i in range(K - 2):
        max_val = min(pa, remaining_sum)
        val = np.random.uniform(0, max_val)
        values.append(val)
        remaining_sum -= val

    last_val = remaining_sum
    if last_val > pa:
        scale = pa / last_val
        values = [v * scale for v in values]
        last_val = pa
    values.append(last_val)

    insert_idx = np.random.randint(K)
    result = np.insert(values, insert_idx, pa)

    return np.array(result)

# Simulate the individuals with unkonw q, i.e. 
#  max <= epsilon under H0 and  max > epsilon under H1
def create_sample_pbekannt(M, K, p, epsilon, pa):
    x = np.zeros(M)

    q = sample_array(K, pa) 

    for m in range(M):
        loc = np.dot(q, p[m,:])
        x[m] = np.random.binomial(2, loc)#[0]
    return q, x


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

# likelihood
def l(q,x, p):
    K, M = p.shape
    res1 = 0
    for m in range(M):
        loc = np.dot(q, p[m,:])
        x_temp = x[m] 
        res1 += x_temp * np.log(loc) + (2-x_temp)*np.log(1-loc)
    return res1


# Soluation: grid search and use whole parameter space, if applicable
def generate_reduced_simplex(K, step, mass):

    result = []

    def recurse(current, depth, remaining):
        if depth == K - 1:
            val = round(remaining, 10)
            if val > 0 and val <= mass and abs((val / step) - round(val / step)) < 1e-6:
                candidate = current + [val]
                if max(candidate) == mass:
                    result.append(candidate)
            return

        for i in range(1, int(min(mass, remaining) / step) + 1):
            val = i * step
            recurse(current + [val], depth + 1, remaining - val)

    recurse([], 0, 1.0)
    return result


def all_permutations_of_lists(list_of_lists):
    """For each list in list_of_lists, generate all permutations (including duplicates)."""
    result = []
    for lst in list_of_lists:
        perms = permutations(lst)
        result.extend(perms)
    return [list(p) for p in result]


def all_unique_permutations_from_lists(list_of_lists):
    """Generates all permutations from a list of lists and removes global duplicates."""
    unique_result = set()

    for lst in list_of_lists:
        for p in permutations(lst):
            unique_result.add(tuple(p))  # use tuple for set operations

    return [list(p) for p in unique_result]

  
def grid_search_l(x, p, K, step, epsilon):
    """Efficient grid search over the simplex with max(q) == epsilon."""
    best_q = None
    best_val = -np.inf
    sub_vectors = generate_reduced_simplex(K, step, epsilon)
    #print("s", sub_vectors)
    res_all = all_permutations_of_lists(sub_vectors)
    res_final = all_unique_permutations_from_lists(res_all)
    #print(res_final)
    for vec in res_final:
        q = np.array(vec)
        val = l(q, x, p)
        if val > best_val:
            best_val = val
            best_q = q.copy()

    return best_q, best_val


# step 2
def bootstrap_estimator(res, epsilon, K, x, p):
    d = max(res)

    if(d > epsilon):
        # Calculate the MLE under the constraint that d = epsilon
        print("Attention", epsilon)
        # 0.001
        hat_hat_q, max_val = grid_search_l(x, p, K, 0.05, epsilon)
        print(hat_hat_q, max_val)
        #hat_hat_q, max_val = grid_search_l(x, p, K, 0.005, epsilon)
        #print(hat_hat_q, max_val)

    else: # normal MLE
        hat_hat_q = res
    
    return hat_hat_q

# Generate bootstrap data with the allele frequencies 
def create_bootstrap(p, B, res, epsilon, K, M, x1):
    hat_q1 = []
    # This is \hat \hat q
    q = bootstrap_estimator(np.array(res[0]), epsilon, K, x1, p)
    #print("bootstrap", q)
    q = np.array([q])
    
    # This is step 3.1 to 3.3
    for b in range(B):
        x1 = create_sample_pqbekannt(M, K, p, q[0])
        temp_q2 = get_admixture_proportions(x1, p.T)
        d = max(temp_q2[0])
        #print("d", d)
        hat_q1.append(d)
    return hat_q1

def test_descicion(alpha, p, x, epsilon, B, K, M):
    # Step 1
    res = get_admixture_proportions(x, p.T)
    #print("res", res)
    # Test Statistic
    d = max(res[0])
    d_bootstrap = create_bootstrap(p, B, res, epsilon, K, M, x)
    # Step 4, i.e. calculation of the quantiles
    quantile_value = np.quantile(d_bootstrap, 1-alpha, method="nearest")    
    return d, quantile_value, d_bootstrap, res


def evaluation_now(n, alpha, epsilon, B, K, M, pa):

    res = []
    # swap also the truth 
    summe = 0
    p = create_p(M, K)
    for i in range(n):
        print(i)
        #print(p)
        q, x1 = create_sample_pbekannt(M, K, p, epsilon, pa)
        print("true_q", q)
        d, q, d_bootstrap, res_now = test_descicion(alpha, p, x1, epsilon, B, K, M)
       # print("q", res_now)
        # q is quantile
        if(d > q): # reject H0
            t = 1
        else: # Do not reejct H0
            t = 0
        summe += t
        print(summe/(i+1))
        res.append(t)
    return res #, p

#%%
import pandas as pd

results_0 = []

for M in [1000]:
    for T in [0.25]:
        for K in [5]:
            for epsilon in [0.5, 0.65, 0.750]:
                
                res_temp_1 = evaluation_now(100, 0.05, epsilon, 100, K, M, T)
                results_0.append({
                        "M": M,
                        "T": T,
                        "K": K,
                        "epsilon": epsilon,
                        "result": sum(res_temp_1)
                    })
                print(results_0)
    # Convert lists to DataFrames
    ädf_0_swap = pd.DataFrame(results_0)
    #df_0_swap.to_csv("results_0_swap_K5_0312_all_equal.csv", index=False)


