import numpy as np
from matplotlib import pyplot as plt
import pygenn
from pygenn import create_var_ref, init_postsynaptic, init_sparse_connectivity
from pygenn.genn_model import GeNNModel, create_weight_update_model, create_postsynaptic_model, create_sparse_connect_init_snippet, create_var_init_snippet, init_weight_update
import time
import os
try:
    import GPUtil
except: print("GPUtil not installed, you will have no info on gpu status, sim will start anyway.")

dt = 0.1
noise = 1.4/np.sqrt(dt) # og 1.4

### Model Components (are here only temporarly to better understand model)
## Odor Receptor
# parameters
param_or = {}
initial_param_or = {
    "r0": 1.0,
    "rb_0": 0.0, "ra_0": 0.0,
    "rb_1": 0.0, "ra_1": 0.0,
    "rb_2": 0.0, "ra_2": 0.0,
    "ra": 0.0,
    "kp1cn_0": 0.0, "km1_0": 0.0, "kp2_0": 0.0, "km2_0":0.0,
    "kp1cn_1": 0.0, "km1_1": 0.0, "kp2_1": 0.0, "km2_1":0.0,
    "kp1cn_2": 0.0, "km1_2": 0.0, "kp2_2": 0.0, "km2_2":0.0,
}

# Logic
or_eq = pygenn.create_neuron_model(
    "or_model",
    params =[],
    vars= [
        ("r0", "scalar"),
        ("rb_0", "scalar"), ("ra_0","scalar"),
        ("rb_1", "scalar"), ("ra_1", "scalar"),
        ("rb_2", "scalar"), ("ra_2", "scalar"),
        ("ra", "scalar"),
        ("kp1cn_0", "scalar"), ("km1_0", "scalar"), ("kp2_0", "scalar"), ("km2_0", "scalar"),
        ("kp1cn_1", "scalar"), ("km1_1", "scalar"), ("kp2_1", "scalar"), ("km2_1", "scalar"),
        ("kp1cn_2", "scalar"), ("km1_2", "scalar"), ("kp2_2", "scalar"), ("km2_2", "scalar"),
        ],
        # here not adding noise due to temperature yet, may do later
    sim_code = """
    // update all bound and activated receptors
    rb_0+= (kp1cn_0*r0 - km1_0*rb_0 + km2_0*ra_0 - kp2_0*rb_0)*dt;
    if (rb_0 > 1.0) rb_0= 1.0;
    if (rb_0 < 0.0) rb_0 = 0.0; // needed to add this to prevent instability
    ra_0+= (kp2_0*rb_0 - km2_0*ra_0)*dt;
    if (ra_0 > 1.0) ra_0= 1.0;
    if (ra_0 < 0.0) ra_0 = 0.0;
    rb_1+= (kp1cn_1*r0 - km1_1*rb_1 + km2_1*ra_1 - kp2_1*rb_1)*dt;
    if (rb_1 > 1.0) rb_1= 1.0;
    if (rb_1 < 0.0) rb_1 = 0.0;
    ra_1+= (kp2_1*rb_1 - km2_1*ra_1)*dt;
    if (ra_1 > 1.0) ra_1= 1.0;
    if (ra_1 < 0.0) ra_1 = 0.0;
    rb_2+= (kp1cn_2*r0 - km1_2*rb_2 + km2_2*ra_2 - kp2_2*rb_2)*dt;
    if (rb_2 > 1.0) rb_2= 1.0;
    if (rb_2 < 0.0) rb_2 = 0.0;
    ra_2+= (kp2_2*rb_2 - km2_2*ra_2)*dt;
    if (ra_2 > 1.0) ra_2= 1.0;
    if (ra_2 < 0.0) ra_2 = 0.0;
    // update ra and calculate the sum of bound receptors
    scalar rb= rb_0 + rb_1 + rb_2;
    if (rb > 1.0) rb= 1.0;
    ra= ra_0 + ra_1 + ra_2;
    if (ra > 1.0) ra= 1.0;
    // then update r0 as a function of rb and ra
    r0= 1.0 - rb - ra;
    if (r0 < 0.0) r0= 0.0;
    if (r0 > 1.0) r0 = 1.0;
    """,
    reset_code = "",
    threshold_condition_code = ""
)

## Neurons

# num neurons per glom
num = dict()
num["glo"]= 160
num["orn"]= 60
num["pn"]= 5
num["ln"]= 25

# Parameters
param_orn = {
    "C_mem": 1.0,
    "V_reset": -70.0,
    "V_thresh": -40.0,
    "V_leak": -60.0,
    "g_leak": 0.01,
    "r_scale": 10.0,
    "g_adapt": 0.0015,
    "V_adapt": -70.0,
    "tau_adapt": 1000.0,
    "noise_A": noise
}
initial_param_orn = {
    "V": -60.0, # in the og 2023 they do not start at rest, but at -60.0 (all of the neurons)!
    "a": 0.0
}

param_pn = {
    "C_mem": 1.0,
    "V_reset": -70.0,
    "V_thresh": -40.0,
    "V_leak": -60.0,
    "g_leak": 0.01,
    "r_scale": 1.0,
    "g_adapt": 0.0,
    "V_adapt": -70.0,
    "tau_adapt": 1000.0,
    "noise_A": noise
}
initial_param_pn = {
    "V": -60.0,
    "a": 0.0
}

param_ln = {
    "C_mem": 1.0,
    "V_reset": -70.0,
    "V_thresh": -40.0,
    "V_leak": -60.0,
    "g_leak": 0.01,
    "r_scale": 1.0,
    "g_adapt": 0.0005,
    "V_adapt": -70.0,
    "tau_adapt": 1000.0,
    "noise_A": noise
}
initial_param_ln = {
    "V": -60.0,
    "a": 0.0
}

# Logic (same for all neurons)
adapt_lifi = pygenn.create_neuron_model(
    "adaptive_LIF",
    params = [
        "C_mem", "V_reset", "V_thresh", "V_leak", "g_leak", "r_scale", "g_adapt", "V_adapt", "tau_adapt", "noise_A"
    ],
    vars = [
        ("V", "scalar"), ("a", "scalar")
    ],
    # in the Fantoni version there is also an eq for g_adapt and g_leak that are changed depending on Temperature. To add if temperature is to be added
    sim_code = """
    V += (-g_leak*(V-V_leak) - g_adapt*a*(V-V_adapt) + r_scale*Isyn + noise_A*gennrand_normal())*dt/C_mem;
    a += -a*dt/tau_adapt;
    """,
    threshold_condition_code = """
    V >= V_thresh
    """,
    reset_code = """
    V = V_reset;
    a += 0.5;
    """
)

### Constucting Model
model = GeNNModel("double", "beeAL") # in og not float, why? takes too long? could try w float
model.dt = dt

## adding neurons
ors = model.add_neuron_population("or", num["glo"], or_eq, param_or, initial_param_or)
orns = model.add_neuron_population("orn", num["glo"]*num["orn"], adapt_lifi, param_orn, initial_param_orn)
orns.spike_recording_enabled = True
pns = model.add_neuron_population("pn", num["glo"]*num["pn"], adapt_lifi, param_pn, initial_param_pn)
pns.spike_recording_enabled = True
lns = model.add_neuron_population("ln", num["glo"]*num["ln"], adapt_lifi, param_ln, initial_param_ln)
lns.spike_recording_enabled = True


### Synapses

## Parameters

n_orn_pn = 12
# initial ORN to PN
orns_pns_ini = {
    "g": 0.008    # (muS)     
    }
# post-synapse ORN to PN
orns_pns_post_params = {
    "tau": 10.0,     # (ms)
    "E": 0.0         # (mV)
    }

n_orn_ln = 12 
# initial ORN to LN
orns_lns_ini = {
    "g": 0.008     # (muS)
    }
# post-synapse ORN to LN
orns_lns_post_params = {
    "tau": 10.0,     # (ms)
    "E": 0.0         # (mV)
    }

# initial PN to LN
pns_lns_ini = {
    "g": 0.001     # (muS)
    }
# post-synapse PN to LN
pns_lns_post_params = {
    "tau": 10.0,     # (ms)
    "E": 0.0       # (mV)
    }

# initial LN to PN
lns_pns_g= 5.5e-5 # then modified by 'ino' weight in the 2023 paper as: "paras["lns_pns_g"] *= np.power(10,ino)"
# post-synapse LN to PN
lns_pns_post_params = {
    "tau": 20.0,     # (ms)
    "E": -80.0       # (mV)
}

# initial LN to LN
lns_lns_g= 2.0e-5 # then modified by 'ino' weight in the 2023 paper
# post-synapse LN to LN
lns_lns_post_params = {
    "tau": 20.0,     # (ms)
    "E": -80.0       # (mV)
}


## Synapses model
# OR to ORNS
ors2orns_connect = create_sparse_connect_init_snippet(
    "or_type_specific",
    row_build_code= """
    const unsigned int row_length = num_post/num_pre;
    const unsigned int offset = id_pre*row_length;
    for (unsigned int k = 0; k < row_length; k++) {
    addSynapse(offset + k);
    } // here endRow was removed, as there doesnt seem to exist a genn5 correspondent
    """,
    col_build_code=None,
    calc_max_row_len_func=lambda num_pre, num_post, pars: int(num_post / num_pre), # helper fucntion "create_cmlf_class" no longer needed in GeNN5
)

pass_or = create_weight_update_model(
    "pass_or",
    params = None,
    vars = None,
    pre_vars = None,
    post_vars = None,
    pre_neuron_var_refs=[("ra_pre_ref", "scalar")],
    post_neuron_var_refs=None,
    psm_var_refs=None,
    derived_params=None,
    synapse_dynamics_code= "addToPost(ra_pre_ref);" # here confusing since in the og code ra_pre is used, but never reference from anywhere
    # in the code, so here I create a reference to ra (that I assume is the variable referenced in the og model
    # (i.e., the fraction of bound and activated receptors)). Also in genn5 changed from synapse_dynamics_code to pre_spike_syn_code
)

pass_postsyn = create_postsynaptic_model(
    "pass_postsyn",
    params=None,
    vars=None,
    neuron_var_refs=None,
    derived_params=None,
    sim_code= """
    injectCurrent(inSyn); // change in how genn works: Isyn cannot be directly written to, instead to pass the value in OR to ORN must use the new injectCurrent
    inSyn = 0.0;
    """
)

# ORN to PN
orns_al_connect = create_sparse_connect_init_snippet(
    "orn_al_type_specific",
    params= ["n_orn", "n_trg", "n_pre"],
    # the logic for col building in the og is still unclear to me, must make sure that the "for (unsigned int c = 0; c < $(n_pre); c++) {}" I used instead of "if (c==0) { $(endCol)} ... c--" is correct
    # hope so. It is if the logic is: for each presynaptic neuron do this (so c is initialized = n_pre, then decreased at each iteration untill n_pre are "finished")
    # my logic does the opposite, from c=0 do this until c is less than n_pre
    col_build_code= """
    for (unsigned int c = 0; c < n_pre; c++) {
    const unsigned int glo = id_post / ((unsigned int) n_trg);
    const unsigned int offset = n_orn*glo;
    const unsigned int tid = gennrand_uniform()*n_orn;
    addSynapse(offset + tid + id_pre_begin);
    }
    """,
    row_build_code=None,
    calc_max_col_len_func=lambda num_pre, num_post, pars: int(pars["n_pre"])
)

# PN to LN within glo, each PN is connected to all LNs in its glomerulus
pns_lns_connect = create_sparse_connect_init_snippet(
    "pns_lns_within_glo",
    params = ["n_pn", "n_ln"],
    row_build_code= """
    const unsigned int offset= (unsigned int) id_pre/((unsigned int) n_pn)*n_ln;
    for (unsigned int k= 0; k < n_ln; k++) {
    addSynapse(offset+k);
    }
    """,
    col_build_code=None,
    calc_max_row_len_func=lambda num_pre, num_post, pars: int(pars["n_ln"])
)

# LN to PN initialization (dense)
ln2pn_conn_init = create_var_init_snippet(
    "lns_pns_conn_init",
    params= ["n_ln", "n_pn", "g"],
    var_init_code= """
    const unsigned int npn= (unsigned int) n_pn;
    const unsigned int nln= (unsigned int) n_ln;
    value=(id_pre/nln == id_post/npn) ? 0.0 : g;
    """
)

# LN to LN initialization (dense)
lns2lns_conn_init = create_var_init_snippet(
    "lns_lns_conn_init",
    params= ["n_ln", "g"],
    var_init_code= """
    const unsigned int nln = (unsigned int) n_ln;
    value = (id_pre/nln == id_post/nln) ? 0.0 : g;
    """
)


### adding connections to model

## ORs to ORNs

ra_from_or = create_var_ref(ors, "ra") # here a reference is created to "ra" from the odor receptors
# the weigth update model must be initialized in GeNN 5 before building the synapse pop
weigth_update_init_or = init_weight_update(
    snippet= pass_or,
    params={},
    vars={},
    pre_var_refs= {"ra_pre_ref": ra_from_or} #this is needed because the pass_or weight update model calls ra_pre
    # which I assume is referencing the fraction of activated receptors for the ORN type
)
# same for the postsynaptic and connectivity
or_orns_postsyn = init_postsynaptic(snippet= pass_postsyn, params={}, vars={})
or_orns_conn = init_sparse_connectivity(snippet= ors2orns_connect, params={})

ors_orns = model.add_synapse_population(
    "ORs_ORN",
    "SPARSE", # the matrix connectivity type I assume is the correspondent to "SPARSE_GLOBALG" in the genn4 model
    source=ors,
    target=orns,
    weight_update_init = weigth_update_init_or,
    postsynaptic_init= or_orns_postsyn,
    connectivity_init= or_orns_conn
)

## ORNs to PNs
v_pn_ref = create_var_ref(pns, "V")

weight_update_init_orn2pn = init_weight_update(
    "StaticPulse",
    params={},
    vars= orns_pns_ini
)
orn_pn_postsyn = init_postsynaptic(
    "ExpCond",
    params=orns_pns_post_params,
    vars={},
    var_refs= {"V": v_pn_ref}
)
orn_pn_conn = init_sparse_connectivity(
    orns_al_connect,
    params={
        "n_orn": num["orn"],
        "n_trg": num["pn"],
        "n_pre": n_orn_pn # to put everything into a dict at the end
    }
)

orns_pns = model.add_synapse_population(
    "ORNs_PNs",
    "SPARSE",
    source= orns,
    target= pns,
    weight_update_init= weight_update_init_orn2pn,
    postsynaptic_init= orn_pn_postsyn,
    connectivity_init= orn_pn_conn
)

## ORNs to LNs
v_ln_ref = create_var_ref(lns, "V")
weight_update_init_orn2ln = init_weight_update(
    "StaticPulse",
    params={},
    vars= orns_lns_ini
)
orn_ln_postsyn = init_postsynaptic(
    "ExpCond",
    params= orns_lns_post_params,
    vars= {},
    var_refs= {"V": v_ln_ref}
)
orn_ln_conn = init_sparse_connectivity(
    orns_al_connect,
    params={
        "n_orn": num["orn"],
        "n_trg": num["ln"],
        "n_pre": n_orn_ln # to put everything into a dict at the end
    }
)

orns_lns = model.add_synapse_population(
    "ORNs_LNs",
    "SPARSE",
    source= orns,
    target= lns,
    weight_update_init= weight_update_init_orn2ln,
    postsynaptic_init= orn_ln_postsyn,
    connectivity_init= orn_ln_conn
)

## PNs to LNs
v_ln_ref = create_var_ref(lns, "V")
weight_update_init_pn2ln = init_weight_update(
    "StaticPulse",
    params={},
    vars= pns_lns_ini
)
pn_ln_postsyn = init_postsynaptic(
    "ExpCond",
    params= pns_lns_post_params,
    vars= {},
    var_refs= {"V": v_ln_ref}
)
pn_ln_conn = init_sparse_connectivity(
    pns_lns_connect,
    params={
        "n_pn": num["pn"],
        "n_ln": num["ln"]
    }
)

pns_lns = model.add_synapse_population(
    "PNs_LNs",
    "SPARSE",
    source= pns,
    target= lns,
    weight_update_init= weight_update_init_pn2ln,
    postsynaptic_init= pn_ln_postsyn,
    connectivity_init= pn_ln_conn
)

## LNs to PNs
g_ln2pn = {"g": 5.5e-5} # value from 'data' in fantoni's version, see if and how they are changed.
# Also present in 2023 version, modified by user specified weight 'ino', see "lns_pns_g"
v_pn_ref = create_var_ref(pns, "V")

weight_update_init_ln2pn = init_weight_update(
    "StaticPulse",
    params={},
    vars= g_ln2pn
)
ln_pn_postsyn = init_postsynaptic(
    "ExpCond",
    params= lns_pns_post_params,
    vars= {},
    var_refs= {"V": v_pn_ref}
)

lns_pns = model.add_synapse_population(
    "LNs_PNs",
    "DENSE",
    source= lns,
    target= pns,
    weight_update_init= weight_update_init_ln2pn,
    postsynaptic_init= ln_pn_postsyn,
    connectivity_init= None
)

## LNs to LNs
g_ln2ln = {"g": 2.0e-5} # value from 'data' in fantoni's version, see if and how they are changed
# Also present in 2023 version, modified by user specified weight 'ino', see "lns_lns_g"

weight_update_init_ln2ln = init_weight_update(
    "StaticPulse",
    params={},
    vars= g_ln2ln
)
ln_ln_postsyn = init_postsynaptic(
    "ExpCond",
    params= lns_lns_post_params,
    vars= {},
    var_refs= {"V": v_ln_ref}
)

lns_lns = model.add_synapse_population(
    "LNs_LNs",
    "DENSE",
    source= lns,
    target= lns,
    weight_update_init= weight_update_init_ln2ln,
    postsynaptic_init= ln_ln_postsyn,
    connectivity_init= None
)

### Build Model
spk_rec_steps = 1000

print("all good, building model...")
print("GPU ram before build:")
try:
    GPUtil.showUtilization(all = False, attrList= None)
except: print("couldn't get gpu info: GPUtil not installed")

start = time.time()
model.build()
model.load(num_recording_timesteps = spk_rec_steps)
end = time.time()

timetaken = round(end-start, 2)
print(f"model was built and loaded successfully! it took {timetaken} s.")
print("showing gpu ram util after building:")
try:
    GPUtil.showUtilization(all = False, attrList = ['memoryUtil', 'memoryTotal', 'memoryUsed', 'memoryFree']) # attrList does not work
except: print("couldn't get gpu info: GPUtil not installed")


### Simulation

## Setting up directory
current_dir = os.path.dirname(os.path.abspath(__file__))
now = time.strftime("%Y%m%d_%H%M%S")
dirname = f"sim_{now}"
folder = os.path.join(current_dir, dirname)
os.makedirs(folder)

## Preparing odor for stim

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
    "A": 1,
    "a_rate": 0.1
}

od1_m = 19
od1_sd_b = 5
od1 = gauss_odor(n_glo=num["glo"], m = od1_m, sd_b= od1_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
odors.append(np.copy(od1))

od2_m = 89
od2_sd_b = 10
od2 = gauss_odor(n_glo=num["glo"], m = od2_m, sd_b= od2_sd_b, a_rate= odor_stable_params["a_rate"], A= odor_stable_params["A"])
odors.append(np.copy(od2))

# defining Hill coeff
hill_exp= np.random.uniform(0.95, 1.05, num["glo"])
np.save(os.path.join(folder,"_hill"),hill_exp)

# function for "presenting" odors to ors
def set_odor_simple(ors, slot, odor, con, hill):
    """
    setting parameters of ors for the chosen odor (effectively "presenting"the odor to the ors).
    Difference from the og function: it autonomously views and pushes the variables to the ors.
    ors: the population of ors (receptors)
    odor: 
    slot: the odor slot to use for each or (each has 3, but can be easily augmented)
    con: concentration of odor: 1e-7 to 1e-1
    hill: the hill coefficient
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


## Protocol timings for presenting odors
# selecting concentration and odor slot
base= np.power(10,0.25)
c = 12
on = 1e-7*np.power(base,c)
off = 0.0
odor_slot = 0

t_baseline_end = 1000
t_odor1_on = 1000
t_odor1_off = t_odor1_on + 3000
t_pause_end = t_odor1_off + 1000
t_odor2_on = t_pause_end
t_odor2_off = t_odor2_on + 3000

odor1_applied = False
odor1_removed = False
odor2_applied = False
odor2_removed = False

## rec spikes from neurons
pop_to_rec = ["orn", "pn", "ln"]
sim_time = 8000 #in ms

spike_t = dict()
spike_id = dict()
for pop in pop_to_rec:
    spike_t[pop] = []
    spike_id[pop] = []

print(spike_t)
print(spike_id)

# rec var state from neurons
what_to_rec = [("orn","V"),("pn","V"),("ln","V"),("or","rb_0")]
vars_rec = dict()
for pop, var in what_to_rec:
    vars_rec[f"{pop}_{var}"] = []
print(vars_rec)

var_view = {}
for pop, var in what_to_rec: # getting var directly (more efficient(?))
    var_view[f"{pop}_{var}"] = model.neuron_populations[pop].vars[var].view

state_rec_steps = 10 # pull state vars every 10 timesteps (curr every 1 ms)


## Preparing protocol
ors_population = model.neuron_populations["or"]

# making sure odor is off
print(f"Initial state: applying 0.0 concentration to type {odor_slot}")
set_odor_simple(ors_population, odor_slot, odors[0], off, hill_exp)


## start simulation
int_t = 0 # init internal counter
print("starting sim...")
start = time.time()

while model.t < sim_time:

    if not odor1_applied and model.t >= t_odor1_on:
        print(f"Time {model.t}, applying odor 1")
        set_odor_simple(ors_population, odor_slot, odors[0], on, hill_exp)
        odor1_applied = True

    if odor1_applied and not odor1_removed and model.t >= t_odor1_off:
        print(f"Time {model.t}, shutting off odor 1")
        set_odor_simple(ors_population, odor_slot, odors[0], off, hill_exp)
        odor1_removed = True

    if odor1_removed and not odor2_applied and model.t >= t_odor2_on:
        print(f"Time {model.t}, applying odor 2")
        set_odor_simple(ors_population, odor_slot, odors[1], on, hill_exp)
        odor2_applied = True

    if odor2_applied and not odor2_removed and model.t >= t_odor2_off:
        print(f"Time {model.t}, shutting off odor 2")
        set_odor_simple(ors_population, odor_slot, odors[1], off, hill_exp)
        odor2_removed = True

    model.step_time()
    int_t += 1

    # pulling var states
    if int_t%state_rec_steps == 0:

        for pop_name, var_name in what_to_rec:
            view_key = f"{pop_name}_{var_name}"

            var_obj = model.neuron_populations[pop_name].vars[var_name]
            var_obj.pull_from_device()
            current_var_view = var_view.get(view_key)

            current_val = np.copy(current_var_view)
            vars_rec[view_key].append(current_val)

    # pulling spikes
    if int_t%spk_rec_steps == 0: # every 1000 timesteps (int_t%1000 == 0 checks if int_t/1000 is equal to 0)
        model.pull_recording_buffers_from_device()
        
        for pop in pop_to_rec:
            pop_to_pull = model.neuron_populations[pop]

            if (pop_to_pull.spike_recording_data[0][0].size > 0):
                spike_t[pop].append(pop_to_pull.spike_recording_data[0][0])
                spike_id[pop].append(pop_to_pull.spike_recording_data[0][1])
                print(f"spikes fetched for time {model.t} from {pop}")
            else: print(f"no spikes in t {model.t} in {pop}")
              
end = time.time()

timetaken = round(end-start, 2)
print(f"sim ended. it took {timetaken} s.")

final_spike_t = {} # better to use new vars where to concatenate the spikes t/id?
final_spike_id = {}
for pop in pop_to_rec:
    if spike_t[pop]:
        spike_t[pop] = np.hstack(spike_t[pop])
        spike_id[pop] = np.hstack(spike_id[pop])

for pop in pop_to_rec:
    np.save(os.path.join(folder, pop + "_spike_t.npy"), spike_t[pop])
    np.save(os.path.join(folder, pop + "_spike_id.npy"), spike_id[pop])

final_recorded_vars = {}
for key, segments_list in vars_rec.items():
    if segments_list:
        vars_rec[key] = np.vstack(segments_list)
    else:
        vars_rec[key] = np.array([])

for pop_var2 in vars_rec:
    np.save(os.path.join(folder, f"{pop_var2}_states.npy"), vars_rec[pop_var2])

