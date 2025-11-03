import numpy as np
import sys
import os
import pandas as pd
#  Calculates the CBB Test for the Maximuum for real data
# Here, we swapped the hypotheses
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 

sys.path.append(module_path)

import Joint_max as joint
from itertools import permutations
#%%
# Calculate the MLE and the Allele Freequencies
# 1) Read the Data
path = "1000G_AIMsetKidd.vcf"
res_all = []
def read_vcf(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            res_all.append(line)
    return res_all
res_x = read_vcf(path)

res_x_rm = res_x[255:310]

x_all = []
for i in res_x_rm:
    genotypes = [gt for gt in i.strip().split('\t') if gt]
    summed_alleles = [sum(map(int, gt.split('|'))) for gt in genotypes[9:]]
    x_all.append(summed_alleles) # 55 times 2504 many entries
x_all_inv = np.array(x_all).transpose()
#%%
# 2) Calculate p

path = "Frequencies_1000G_AIMsetKidd.csv"
data_p = pd.read_csv(path)
# EUR, AFR, SAS, EAS, AMR
continent_map = {
    "EUR": ["GBR", "FIN", "IBS", "TSI", "CEU"],
     "AMR": ["MXL", "CLM", "PUR", "PEL"],
     "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
     "EAS": ["KHV", "CDX", "CHB", "JPT", "CHS"],
     "AFR": ["ACB", "GWD", "MSL", "ESN", "YRI", "LWK", "ASW"]
}

df_grouped = pd.DataFrame()
df_grouped['Position'] = data_p['Unnamed: 0']  
# sort columns according to continent
for continent, populations in continent_map.items():
    df_grouped[continent] = data_p[populations].mean(axis=1)
p = np.array(df_grouped.iloc[:,1:]).transpose()

# 3) Calculate the MLE for one indiivudal
from scipy.stats import dirichlet, binom, norm

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

hat_q = get_admixture_proportions(x_all_inv[0], p)
#%%
# likelihood in the Admixture Model
def l(q,x, p):
    K, M = p.shape
    res1 = 0
    for m in range(M):
        loc = np.dot(q, p[m,:])
        x_temp = x[m] 
        res1 += x_temp * np.log(loc) + (2-x_temp)*np.log(1-loc)
    return -res1
#  grid search and use whole parameter space, if applicable
def generate_reduced_simplex(K, step, mass):
    """Generates K-dimensional points on the simplex where:
    - sum(x) == 1
    - all x_i > 0
    - max(x_i) == mass
    - all x_i are multiples of `step`
    """
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
    best_val = np.inf
    sub_vectors = generate_reduced_simplex(K, step, epsilon)
    res_all = all_permutations_of_lists(sub_vectors)
    res_final = all_unique_permutations_from_lists(res_all)
    for vec in res_final:
        q = np.array(vec)
        val = l(q, x, p)
        if val < best_val:
            best_val = val
            best_q = q.copy()

    return best_q, best_val

# step 2
def bootstrap_estimator(res, epsilon, K, x, p):
    d = max(res)
    if(d > epsilon):
        # Calculate the MLE under the constraint that d = epsilon
        hat_hat_q, max_val = grid_search_l(x, p, K, 0.01, epsilon)
    else: # normal MLE
        hat_hat_q = res
    return hat_hat_q
# Generate bootstrap data with the allele frequencies 
def create_bootstrap(p, B, res, epsilon, K, M, x1):
    hat_q1 = []
    # This is \hat \hat q
    q = bootstrap_estimator(np.array(res[0]), epsilon, K, x1, p)
    q = np.array([q])
    # This is step 3.1 to 3.3
    for b in range(B):
        x1 = joint.create_sample_pqbekannt(M, K, p, q[0])
        temp_q2 = joint.get_admixture_proportions(x1, p.T)
        d = max(temp_q2[0])
        hat_q1.append(d)
    return hat_q1

def test_descicion(alpha, p, x, epsilon, B, K, M):
    # Step 1
    res = joint.get_admixture_proportions(x, p.T)
    print("res", res)
    # Test Statistic
    d = max(res[0])
    d_bootstrap = create_bootstrap(p, B, res, epsilon, K, M, x)
    # Step 4, i.e. calculation of the quantiles
    quantile_value = np.quantile(d_bootstrap, 1-alpha, method="nearest")    
    return d, quantile_value, d_bootstrap, res

def evaluation_now(alpha, epsilon, B, K, M,p):

    res = []
    est_q = []
    # swap also the truth 
    summe = 0
    for i in range(2504):
        x  = x_all_inv[i]
        d, q, d_bootstrap, res_q = test_descicion(alpha, p.T, x, epsilon, B, K, M)
        est_q.append(res_q)
        # q is quantile
        if(d > q): # reject H0
            t = 1
        else: # Do not reject H0
            t = 0
        summe += t
        print(summe/(i+1))
        res.append(t)
    return res, est_q

x  = x_all_inv[0]
M = 55
K = 5
B = 100
epsilon = 0.75
alpha = 0.05
res_all_test = evaluation_now(alpha, epsilon, B, K, M,p)

#%%
# Prepare the data for plotting!
#  plot the results

path = "1000G_SampleListWithLocations.txt"
individual_list = pd.read_csv(path,sep='\t', header=None)

pattern = 'EUR'

indices = individual_list.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)

result = individual_list[indices]
index_list_EUR = result.index.tolist()

print("Indices mit §EUR§:", index_list_EUR)


pattern = 'AMR'

indices = individual_list.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)

result = individual_list[indices]
index_list_AMR = result.index.tolist()

print("Indices mit §EUR§:", index_list_AMR)


pattern = 'SAS'

indices = individual_list.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)

result = individual_list[indices]
index_list_SAS = result.index.tolist()

print("Indices mit §EUR§:", index_list_SAS)


pattern = 'EAS'

indices = individual_list.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)

result = individual_list[indices]
index_list_EAS = result.index.tolist()

print("Indices mit §EUR§:", index_list_EAS)


pattern = 'AFR'

indices = individual_list.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)

result = individual_list[indices]
index_list_AFR = result.index.tolist()

print("Indices mit §EUR§:", index_list_AFR)

res_all_test_backup = res_all_test
lst = res_all_test_backup[0]
total_EUR = sum(lst[i] for i in index_list_EUR)
print(total_EUR, len(index_list_EUR))
total_AFR = sum(lst[i] for i in index_list_AFR)
print(total_AFR, len(index_list_AFR))
total_AMR = sum(lst[i] for i in index_list_AMR)
print(total_AMR, len(index_list_AMR))
total_SAS = sum(lst[i] for i in index_list_SAS)
print(total_SAS, len(index_list_SAS))
total_EAS = sum(lst[i] for i in index_list_EAS)
print(total_EAS, len(index_list_EAS))
#%%

import matplotlib.pyplot as plt
import numpy as np

# Example results
intervals = [(284, 503),
             (602, 661),
             (45, 347),
             (200, 489),
             (416, 504)]

starts = [i[0] for i in intervals]
ends = [i[1] for i in intervals]

n = len(intervals)
x = np.arange(n)  # 

bar_width = 0.35  
# Plot
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, starts, bar_width, label='Number of Reject H0', color='steelblue')
plt.bar(x + bar_width/2, ends, bar_width, label='Number of Individuals in total', color='green')

plt.xticks(x, ["EUR", "AFR", "AMR", "SAS", "EAS"])
plt.ylabel('Number of Individuals')
plt.legend()
plt.tight_layout()
plt.show()
