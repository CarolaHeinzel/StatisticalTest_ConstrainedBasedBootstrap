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
    q = np.array([q])
    
    # This is step 3.1 to 3.3
    for b in range(B):
        x1 = joint.create_sample_pqbekannt(M, K, p, q[0])
        temp_q2 = joint.get_admixture_proportions(x1, p.T)
        d = max(temp_q2[0])
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
    booli: Int
        Either 0 (if H0 is true) or 1 (if alternative is true).

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
        q, x1 = joint.create_sample_pbekannt(M, K, p, epsilon, booli)
        d, q, d_bootstrap = test_descicion(alpha, p, x1, epsilon, B, K, M)
        # q is quantile
        if(d < q): # reject H0
            t = 1
        else: # Do not reject H0
            t = 0
        summe+= t
        res.append(t)
    return res
