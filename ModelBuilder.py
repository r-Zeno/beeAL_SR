import numpy as np
import pygenn
from pygenn import create_var_ref, init_postsynaptic, init_sparse_connectivity
from pygenn.genn_model import GeNNModel, create_weight_update_model, create_postsynaptic_model, create_sparse_connect_init_snippet, create_var_init_snippet, init_weight_update
import time
try:
    import GPUtil
except: print("GPUtil not installed, you will have no info on gpu status, sim will start anyway.")

class ModelBuilder:
    """
    Creates the model, need to call explicitly 'model_builder.build' after creating the ModelBuilder instance,
    - paras: parameters dictionary (json file).
    - dt: default to 0.1 ms.
    """
    def __init__(self, paras:dict, dt=0.1):

        self.paras = paras
        self.dt = dt

        self.model_settings = dict()

        self.model = GeNNModel("double", "beeAL")
        self.model.dt = dt

        self.ors = None
        self.orns = None
        self.pns = None
        self.lns = None
        self.elns = None

        # set the needed pops for the connections that could be selected
        self.build_plan = {
            "or2orn": (self._or2orn, "ors", "orns"),
            "orn2pn": (self._orn2pn, "orns", "pns"),
            "orn2ln": (self._orn2ln, "orns", "lns"),
            "pn2ln": (self._pn2ln, "pns", "lns"),
            "ln2pn": (self._ln2pn, "lns", "pns"),
            "ln2ln": (self._ln2ln, "lns") 
        }

        self.neuron_pops = ["ors","orns","pns","lns","elns"]

    def _neuron_groups_init(self, ors=True, orns=True, spike_orn=True, pns=True, spike_pn=True, lns=True, spike_ln=True, elns=False, spike_eln=False):

        if ors:
            or_eq = self._or_model_builder()
            self.ors = self.model.add_neuron_population("or", int(self.paras["num"]["glo"]), or_eq,
                                                        self.paras["param_or"], self.paras["initial_param_or"])

        adapt_lifi = None
        if (orns or pns or lns or elns) and adapt_lifi==None: # ugly? to load adapt_lifi only once
            adapt_lifi = self._lifi_builder()

        if orns:
            self.orns = self.model.add_neuron_population("orn", int(self.paras["num"]["glo"])*int(self.paras["num"]["orn"]),
                                                        adapt_lifi, self.paras["param_orn"],
                                                        self.paras["initial_param_orn"])
            if spike_orn:
                self.orns.spike_recording_enabled = True
        if pns:
            self.pns = self.model.add_neuron_population("pn", int(self.paras["num"]["glo"])*int(self.paras["num"]["pn"]),
                                                        adapt_lifi, self.paras["param_pn"],
                                                        self.paras["initial_param_pn"])
            if spike_pn:
                self.pns.spike_recording_enabled = True
        if lns:
            self.lns = self.model.add_neuron_population("ln", int(self.paras["num"]["glo"])*int(self.paras["num"]["ln"]),
                                                        adapt_lifi, self.paras["param_ln"],
                                                        self.paras["initial_param_ln"])
            if spike_ln:
                self.lns.spike_recording_enabled = True
        if elns:
            self.elns = self.model.add_neuron_population("eln", int(self.paras["num"]["glo"])*int(self.paras["num"]["elns"]),
                                                        adapt_lifi, self.paras["param_eln"],
                                                        self.paras["initial_param_eln"])
            if spike_eln:
                self.elns.spike_recording_enabled = True

    def _or2orn(self):

        self.ors2orns_connect = create_sparse_connect_init_snippet(
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

        self.ra_from_or = create_var_ref(self.ors, "ra")

        self.pass_or = create_weight_update_model(
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
        
        self.weight_update_init_or = init_weight_update(
            snippet= self.pass_or,
            params={},
            vars={},
            pre_var_refs= {"ra_pre_ref": self.ra_from_or} #this is needed because the pass_or weight update model calls ra_pre
            # which relies on implicit referencing not available in genn 5.2.0
        )
        
        self.pass_postsyn = create_postsynaptic_model(
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
        
        self.or_orns_postsyn = init_postsynaptic(snippet= self.pass_postsyn, params={}, vars={})
        
        self.or_orns_conn = init_sparse_connectivity(snippet= self.ors2orns_connect, params={})
        
        self.ors_orns = self.model.add_synapse_population(
                "ORs_ORN",
                "SPARSE", # the matrix connectivity type I assume is the correspondent to "SPARSE_GLOBALG" in the genn4 model
                source=self.ors,
                target=self.orns,
                weight_update_init = self.weight_update_init_or,
                postsynaptic_init= self.or_orns_postsyn,
                connectivity_init= self.or_orns_conn
            )

    def _orn2pn(self):

        self.orns_pns_connect = create_sparse_connect_init_snippet(
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

        self.v_pn_ref = create_var_ref(self.pns, "V")

        self.weight_update_init_orn2pn = init_weight_update(
            "StaticPulse",
            params={},
            vars= self.paras["orns_pns_ini"]
        )

        self.orn_pn_postsyn = init_postsynaptic(
            "ExpCond",
            params=self.paras["orns_pns_post_params"],
            vars={},
            var_refs= {"V": self.v_pn_ref}
        )

        self.orn_pn_conn = init_sparse_connectivity(
            self.orns_pns_connect,
            params={
                "n_orn": int(self.paras["num"]["orn"]),
                "n_trg": int(self.paras["num"]["pn"]),
                "n_pre": int(self.paras["n_orn_pn"])
            }
        )

        self.orns_pns = self.model.add_synapse_population(
                "ORNs_PNs",
                "SPARSE",
                source= self.orns,
                target= self.pns,
                weight_update_init= self.weight_update_init_orn2pn,
                postsynaptic_init= self.orn_pn_postsyn,
                connectivity_init= self.orn_pn_conn
            )

    def _orn2ln(self):

        self.orns_lns_connect = create_sparse_connect_init_snippet(
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
        
        self.v_ln_ref = create_var_ref(self.lns, "V")

        self.weight_update_init_orn2ln = init_weight_update(
            "StaticPulse",
            params={},
            vars= self.paras["orns_lns_ini"]
        )

        self.orn_ln_postsyn = init_postsynaptic(
            "ExpCond",
            params= self.paras["orns_lns_post_params"],
            vars= {},
            var_refs= {"V": self.v_ln_ref}
        )
        
        self.orn_ln_conn = init_sparse_connectivity(
            self.orns_lns_connect,
            params={
                "n_orn": int(self.paras["num"]["orn"]),
                "n_trg": int(self.paras["num"]["ln"]),
                "n_pre": int(self.paras["n_orn_ln"])
            }
        )
        
        self.orns_lns = self.model.add_synapse_population(
            "ORNs_LNs",
            "SPARSE",
            source= self.orns,
            target= self.lns,
            weight_update_init= self.weight_update_init_orn2ln,
            postsynaptic_init= self.orn_ln_postsyn,
            connectivity_init= self.orn_ln_conn
        )

    def _pn2ln(self):

        self.pns_lns_connect = create_sparse_connect_init_snippet(
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

        self.v_ln_ref = create_var_ref(self.lns, "V")

        self.weight_update_init_pn2ln = init_weight_update(
            "StaticPulse",
            params={},
            vars= self.paras["pns_lns_ini"]
        )
        
        self.pn_ln_postsyn = init_postsynaptic(
            "ExpCond",
            params= self.paras["pns_lns_post_params"],
            vars= {},
            var_refs= {"V": self.v_ln_ref}
        )

        self.pn_ln_conn = init_sparse_connectivity(
            self.pns_lns_connect,
            params={
                "n_pn": int(self.paras["num"]["pn"]),
                "n_ln": int(self.paras["num"]["ln"])
            }
        )

        self.pns_lns = self.model.add_synapse_population(
                "PNs_LNs",
                "SPARSE",
                source= self.pns,
                target= self.lns,
                weight_update_init= self.weight_update_init_pn2ln,
                postsynaptic_init= self.pn_ln_postsyn,
                connectivity_init= self.pn_ln_conn
            )
        
    def _ln2pn(self):

        g_ln2pn = {"g": 5.5e-5} # value from 'data' in fantoni's version, see if and how they are changed.
        # Also present in 2023 version, modified by user specified weight 'ino', see "lns_pns_g"

        self.v_pn_ref = create_var_ref(self.pns, "V")

        self.weight_update_init_ln2pn = init_weight_update(
            "StaticPulse",
            params={},
            vars= g_ln2pn
        )

        self.ln_pn_postsyn = init_postsynaptic(
            "ExpCond",
            params= self.paras["lns_pns_post_params"],
            vars= {},
            var_refs= {"V": self.v_pn_ref}
        )

        self.lns_pns = self.model.add_synapse_population(
            "LNs_PNs",
            "DENSE",
            source= self.lns,
            target= self.pns,
            weight_update_init= self.weight_update_init_ln2pn,
            postsynaptic_init= self.ln_pn_postsyn,
            connectivity_init= None # since everyone is connected to everyone (in old GeNN here would put
            # the g for inhibitory synapses, however now its already done in 'init_weight_update')
        )

    def _ln2ln(self):

        g_ln2ln = {"g": 2.0e-5} # value from 'data' in fantoni's version, see if and how they are changed
            # Also present in 2023 version, modified by user specified weight 'ino', see "lns_lns_g"

        self.v_ln_ref = create_var_ref(self.lns, "V")

        self.weight_update_init_ln2ln = init_weight_update(
            "StaticPulse",
            params={},
            vars= g_ln2ln
        )

        self.ln_ln_postsyn = init_postsynaptic(
            "ExpCond",
            params= self.paras["lns_lns_post_params"],
            vars= {},
            var_refs= {"V": self.v_ln_ref}
        )

        self.lns_lns = self.model.add_synapse_population(
                "LNs_LNs",
                "DENSE",
                source= self.lns,
                target= self.lns,
                weight_update_init= self.weight_update_init_ln2ln,
                postsynaptic_init= self.ln_ln_postsyn,
                connectivity_init= None # since everyone is connected to everyone
            )

    def _loader(self):
        print("all good, building model...")
        print("GPU ram before build:")
        try:
            GPUtil.showUtilization(all = False, attrList= None)
        except: print("couldn't get gpu info: GPUtil not installed")

        start = time.time()
        self.model.build()
        self.model.load(num_recording_timesteps = int(self.paras["spk_rec_steps"]))
        end = time.time()

        timetaken = round(end-start, 2)
        print(f"model was built and loaded successfully! it took {timetaken} s.")
        print("showing gpu ram util after building:")
        try:
            GPUtil.showUtilization(all = False, attrList = ['memoryUtil', 'memoryTotal', 'memoryUsed', 'memoryFree']) # attrList does not work
        except: print("couldn't get gpu info: GPUtil not installed")

    def _or_model_builder(self):

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

        return or_eq

    def _lifi_builder(self):

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

        return adapt_lifi

    def build(self):
        """
        This is the main method to build the network, calls every GeNN function to build and load the model.
        
        """
        components = self.paras.get("components")
        neurons = components.get("neurons")
        connections = components.get("synapses")
        spikes_rec = components.get("spikes to record")

        connections_tobuild = []
        for type, tobuild in connections.items():
            
                if tobuild:

                    if type in self.build_plan:
                        connections_tobuild.append(type)
                    else: print(f"Warning: {type} not in AL model, check json")

        print(connections_tobuild)

        # now use a set to add neurons, this way no duplicates appear if a population is present
        # explicitly in the .json configuration and implicitly in the required neurons for a given connection
        neurons_tobuild = set()

        for conn_type in connections_tobuild:  # take the neuron pops necessary for the specified connections

            if conn_type != "ln2ln":
                neurons_tobuild.add(self.build_plan[conn_type][1])
                neurons_tobuild.add(self.build_plan[conn_type][2])
            else: neurons_tobuild.add(self.build_plan[conn_type][1]) # ugly way to avoid indexing error since ln2ln only has 1 necessary pop
            
        for ntype, tobuild in neurons.items():
            
            if ntype in self.neuron_pops:
                if tobuild:
                    neurons_tobuild.add(ntype) # its a set so duplicate are ignored!
            else: print(f"Error: {ntype} not present in AL")

        print(neurons_tobuild)

        print(f"building model with {neurons_tobuild} populations and {connections_tobuild} connections...")
        self._neuron_groups_init(
            ors=("ors" in neurons_tobuild), orns=("orns" in neurons_tobuild),
            pns=("pns" in neurons_tobuild), lns=("lns" in neurons_tobuild),
            elns=("elns" in neurons_tobuild),
            spike_orn=spikes_rec.get("orn", True), # the second argument set the defaults if not specified
            spike_pn=spikes_rec.get("pn", True),
            spike_ln=spikes_rec.get("ln", True),
            spike_eln=spikes_rec.get("eln", False),
        )

        if connections_tobuild:
            print(f"building connections {connections_tobuild}...")
            
            for ctype in connections_tobuild:
                conn_builder = self.build_plan[ctype][0] # the called method is set, the method is referenced from the build_plan dict created in dict
                conn_builder()
        else: print("Warning: building a model without connections!")

        self._loader()
        return self.model
