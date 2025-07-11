import numpy as np
from numba import jit

@jit(nopython=True)
def vp_metric(train_1, train_2, cost):
    """
    Computes the Victor-Purpura distance, given a pair of spike train timings and a cost (q*Dt)
    - train_1, train_2: timings of spikes
    - cost: the q value
    """
    
    s_train_i = train_1
    s_train_j = train_2
    q = cost

    nspt_i = len(s_train_i)
    nspt_j = len(s_train_j)

    scr = np.zeros((nspt_i + 1, nspt_j +1))

    scr[:,0] = range(0, nspt_i + 1)
    scr[0,:] = range(0, nspt_j + 1)

    if nspt_i > 0 and nspt_j >0:
        
        for i in range(1, nspt_i + 1):

            for j in range(1, nspt_j + 1):

                scr[i,j] = min(scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1, j-1]+ q * abs(s_train_i[i-1] - s_train_j[j-1]))

    return scr[nspt_i, nspt_j]