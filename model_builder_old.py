import numpy as np
import pygenn
from pygenn import create_var_ref, init_postsynaptic, init_sparse_connectivity
from pygenn.genn_model import GeNNModel, create_weight_update_model, create_postsynaptic_model, create_sparse_connect_init_snippet, create_var_init_snippet, init_weight_update
import time
try:
    import GPUtil
except: print("GPUtil not installed, you will have no info on gpu status, sim will start anyway.")

def model_builder(paras, dt=0.1, eLns = False):
    ### Constucting Model
    model = GeNNModel("double", "beeAL")

    model.dt = dt
    
    ## adding neurons
    ors = model.add_neuron_population("or", paras["num"]["glo"], paras["or_eq"], paras["param_or"], paras["initial_param_or"])
    orns = model.add_neuron_population("orn", paras["num"]["glo"]*paras["num"]["orn"], paras["adapt_lifi"], paras["param_orn"], paras["initial_param_orn"])
    orns.spike_recording_enabled = True
    pns = model.add_neuron_population("pn", paras["num"]["glo"]*paras["num"]["pn"], paras["adapt_lifi"], paras["param_pn"], paras["initial_param_pn"])
    pns.spike_recording_enabled = True
    lns = model.add_neuron_population("ln", paras["num"]["glo"]*paras["num"]["ln"], paras["adapt_lifi"], paras["param_ln"], paras["initial_param_ln"])
    lns.spike_recording_enabled = True
    if eLns:
        elns = model.add_neuron_population("eln", paras["num"]["glo"]*paras["num"]["elns"], paras["adapt_lifi"], paras["param_eln"], paras["initial_param_eln"])
        elns.spike_recording_enabled = True
        
    ## making connections
    # OR to ORN
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
        synapse_dynamics_code= "addToPost(ra_pre_ref);" # here confusing since in old genn can reference the previous state of a var with "varname_pre", 
        # so here I create a reference to ra (that I assume is the variable referenced in the og model
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
    weight_update_init_or = init_weight_update(
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
        weight_update_init = weight_update_init_or,
        postsynaptic_init= or_orns_postsyn,
        connectivity_init= or_orns_conn
    )

    ## ORNs to PNs
    v_pn_ref = create_var_ref(pns, "V")

    weight_update_init_orn2pn = init_weight_update(
        "StaticPulse",
        params={},
        vars= paras["orns_pns_ini"]
    )
    orn_pn_postsyn = init_postsynaptic(
        "ExpCond",
        params= paras["orns_pns_post_params"],
        vars={},
        var_refs= {"V": v_pn_ref}
    )
    orn_pn_conn = init_sparse_connectivity(
        orns_al_connect,
        params={
            "n_orn": paras["num"]["orn"],
            "n_trg": paras["num"]["pn"],
            "n_pre": paras["n_orn_pn"]
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
        vars= paras["orns_lns_ini"]
    )
    orn_ln_postsyn = init_postsynaptic(
        "ExpCond",
        params= paras["orns_lns_post_params"],
        vars= {},
        var_refs= {"V": v_ln_ref}
    )
    orn_ln_conn = init_sparse_connectivity(
        orns_al_connect,
        params={
            "n_orn": paras["num"]["orn"],
            "n_trg": paras["num"]["ln"],
            "n_pre": paras["n_orn_ln"]
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
        vars= paras["pns_lns_ini"]
    )
    pn_ln_postsyn = init_postsynaptic(
        "ExpCond",
        params= paras["pns_lns_post_params"],
        vars= {},
        var_refs= {"V": v_ln_ref}
    )
    pn_ln_conn = init_sparse_connectivity(
        pns_lns_connect,
        params={
            "n_pn": paras["num"]["pn"],
            "n_ln": paras["num"]["ln"]
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
        params= paras["lns_pns_post_params"],
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
        params= ["lns_lns_post_params"],
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

    print("all good, building model...")
    print("GPU ram before build:")
    try:
        GPUtil.showUtilization(all = False, attrList= None)
    except: print("couldn't get gpu info: GPUtil not installed")

    start = time.time()
    model.build()
    model.load(num_recording_timesteps = paras["spk_rec_steps"])
    end = time.time()

    timetaken = round(end-start, 2)
    print(f"model was built and loaded successfully! it took {timetaken} s.")
    print("showing gpu ram util after building:")
    try:
        GPUtil.showUtilization(all = False, attrList = ['memoryUtil', 'memoryTotal', 'memoryUsed', 'memoryFree']) # attrList does not work
    except: print("couldn't get gpu info: GPUtil not installed")

    return model
