import numpy as np

def gauss_odor(n_glo: int, m: float, sd_b: float, a_rate: float, A: float=1.0, clip: float=0.0, m_a: float=0.25, sd_a: float=0.0, min_a: float=0.01, max_a: float=0.05, rand_a: bool=False ,hom_a: bool=True) -> np.array:
    """
    Generates odor binding rate and activation rate, following a gaussian for each glom,
    based on its distance (only for binding rate if hom_a=True) from the maximally responding glom = m (midpoint of gaussian).
    - n_glo: number of glomeruli
    - m: midpoint of gaussian profile for binding rates (also the idx of the maximum responding glom to that odor)
    - sd_b: the standardard deviation of gaussian distr of binding rates (so how specific is the glom response to the odor)
    - a_rate: activation rate for homogenous odor act rate. used only if rand_a=False (it is by default)
    - A: amplitude of gaussian
    - clip: cut-off for binding threshold, if a glom following the assignement based on distance would have a lower binding rate its set to 0
    - m_a: mean activation rate
    - sd_a: standard deviation of activation rate
    - min_a: minimum act rate value
    - max_a: maximum act rate value
    - rand_a: whether the activation rate is randomly chosen or specified by "a_rate"
    - hom_a: whether the activation rate is the same for all glomeruli (for the individual odor) 
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

def set_odor_simple(ors, slot, odor, con, hill):
    """
    setting parameters of ors for the chosen odor (effectively "presenting"the odor to the ors).
    Difference from the og function: it autonomously views and pushes the variables to the ors.
    - ors: the population of ors (receptors)
    - odor: 
    - slot: the odor slot to use for each or (each has 3, but can be easily augmented)
    - con: concentration of odor: 1e-7 to 1e-1
    - hill: the hill coefficient
    """
    od = np.squeeze(odor)
    kp1cn = np.power(od[:,0]*con,hill)
    km1 = 0.025
    kp2 = od[:,1]
    km2 = 0.025

    vars_names = [
        f"kp1cn_{slot}",
        f"km1_{slot}",
        f"kp2_{slot}",
        f"km2_{slot}"
    ]

    odor_vars = {
        f"kp1cn_{slot}": kp1cn,
        f"km1_{slot}": km1,
        f"kp2_{slot}": kp2,
        f"km2_{slot}": km2
    }

    for odvar_name, odvalues in odor_vars.items():
        ors.vars[odvar_name].view[:] = odvalues

    for name in vars_names:
        ors.vars[name].push_to_device()
        print(f"Pushed {name} to ors to device.")
