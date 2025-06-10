import numpy as np
from scipy.stats import dirichlet
import sys
import os
#  Calculates the CBB Test for the Maximuum
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 

sys.path.append(module_path)

import Joint_Max as joint


def bootstrap_estimator(res, epsilon, K, x, p):
    d = max(res)
    
    if(d < epsilon):
        # Calculate the MLE under the constraint that d = epsilon
        res_test = joint.combined(x,p, K, epsilon)
        hat_hat_q = res_test
    else:
        hat_hat_q = res
    
    return hat_hat_q

# Generate bootstrap data with the allele frequencies 
def create_bootstrap(p, B, res, epsilon, K, M, x1):
    hat_q1 = []
    # This is \hat \hat q
    q = bootstrap_estimator(np.array(res[0]), epsilon, K, x1, p)
    #print("q2", q)
    q = np.array([q])
    
    # This is step 3.1 to 3.3
    for b in range(B):
        x1 = joint.create_sample_pqbekannt(M, K, p, q[0])
        temp_q2 = joint.get_admixture_proportions(x1, p.T)
        #print("t",  temp_q2)
        d = max(temp_q2[0])
        #print("d", d)
        hat_q1.append(d)
    return hat_q1

def test_descicion(alpha, p, x, epsilon, B, K, M):
    res = joint.get_admixture_proportions(x, p.T)
    # Test Statistic
    d = max(res[0])
    d_bootstrap = create_bootstrap(p, B, res, epsilon, K, M, x)
    # Step 4, i.e. calculation of the quantiles
    quantile_value = np.quantile(d_bootstrap, alpha)    
    return d, quantile_value, d_bootstrap




def evaluation_now(n, alpha, epsilon, B, K, M, booli):
    '''

    Parameters
    ----------
    n : Int
        Number of repetitions.
    alpha : Float
        Test Niveu.
    epsilon : TYPE
        DESCRIPTION.
    B : Int
        DESCRIPTION.
    q1 : Vector
        True Ancestry.
    M : Int
        Number of markers.

    Returns
    -------
    res : List
        Test descicions for repetition 1,..., n. 1 means that H0 can be rejected 
        and 0 means that H0 cannot be rejected.

    '''

    res = []
    summe = 0
    for i in range(n):
        p = joint.create_p(M, K)
        #print(p)
        q, x1 = joint.create_sample_pbekannt(M, K, p, epsilon, booli)
        #print("q", q)
        d, q, d_bootstrap = test_descicion(alpha, p, x1, epsilon, B, K, M)
        # q is quantile
        if(d < q): # reject H0
            t = 1
        else: # Do not reejct H0
            t = 0
        print("t", t)
        summe+= t
        print(i)
        print(summe/(i+1))
        res.append(t)
    return res
#%%
temp_hier_h1 = evaluation_now(100, 0.05, 0.9, 100, 2, 1000,1)
print(temp_hier_h1)
#%%
temp_hier_h0 = evaluation_now(100, 0.05, 0.75, 100, 2, 1000,0)
print(temp_hier_h0)
#print(temp_hier)
#%%
temp_hier_h1_3 = evaluation_now(100, 0.05, 0.75, 100, 3, 100,1)
print(temp_hier_h1_3)
temp_hier_h0_3 = evaluation_now(100, 0.05, 0.75, 100, 3, 100,0)
print(temp_hier_h0_3)
