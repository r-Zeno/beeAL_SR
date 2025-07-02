import numpy as np
from matplotlib import pyplot as plt

num = {
       "glo": 160
       }

def gauss_odor(n_glo: int, m: float, sd_b: float, a_rate: float, A: float=1.0, clip: float=0.0, m_a: float=0.25, sd_a: float=0.0, min_a: float=0.01, max_a: float=0.05, rand_a: bool=False ,hom_a: bool=True) -> np.array:
    """
    Generates odor binding rate and activation rate, following a gaussian for each glom,
    based on its distance (only for binding rate if hom_a=True) from the maximally responding glom = m (midpoint of gaussian).
    n_glo: number of glomeruli
    m: midpoint of gaussian profile for binding rates (also the idx of the maximum responding glom to that odor)
    sd_b: the standardard deviation of gaussian distr of binding rates (so how specific is the glom response to the odor)
    a_rate: activation rate for homogenous odor act rate. used only if rand_a=False (it is by default)
    A: amplitude of gaussian
    clip: cut-off for binding threshold, if a glom following the assignement based on distance would have a lower binding rate its set to 0
    m_a: mean activation rate
    sd_a: standard deviation of activation rate
    min_a: minimum act rate value
    max_a: maximum act rate value
    rand_a: whether the activation rate is randomly chosen or specified by "a_rate"
    hom_a: whether the activation rate is the same for all glomeruli (for the individual odor) 
    """
    odor = np.zeros((n_glo,2)) # row: glom, col: 0 -> binding rate|1 -> activation rate
    d = np.arange(0,n_glo) # representation of glomeruli
    d = d-m # each glom is represented by its distance to the glom 'm'
    d = np.minimum(np.abs(d), np.abs(d+n_glo)) # pathfinding/clock-like solution to find the distance of each glom to glom[m]
    d = np.minimum(np.abs(d), np.abs(d-n_glo))
    od = np.exp(-np.power(d,2)/(2*np.power(sd_b,2)))
    od *= np.power(10,A)
    od[od > 0] = od[od > 0] + clip
    odor[:,0] = od # assigning binding rates
    if rand_a:
        if hom_a:
            a = 0.0
            while a < min_a or a > max_a:
                a = np.random.normal(m_a,sd_a)
            odor[:,1] = a
        else:
            for i in range(n_glo):
                a = 0.0
                while a < min_a or a > max_a:
                    a = np.random.normal(m_a,sd_a)
                odor[i,1] = a
    else:
        odor[:,1] = a_rate

    return odor

# generating 2 (for now) odors
odors = []
odor_stable_params = {
    "A": 1.0,
    "a_rate": 0.1
}

od1_m = 17
od1_sd_b = 5
od1 = gauss_odor(n_glo=num["glo"], m = od1_m, sd_b= od1_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
odors.append(np.copy(od1))

od2_m = 79
od2_sd_b = 10
od2 = gauss_odor(n_glo=num["glo"], m = od2_m, sd_b= od2_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
odors.append(np.copy(od2))

print(np.size(odors))

plt.plot(od1[:,0])
plt.plot(od2[:,0])

