import numpy as np
import os
import time
import json
from collections import defaultdict
from pygenn.genn_model import GeNNModel
from ExperimentStatic import ExperimentStatic
from ExperimentDynamicSingle import *
from helpers import gauss_odor, single_glom_odor


class Experimenter:

    def __init__(self, model: GeNNModel, exp_paras: dict, folder: str, which_exp: str, num_orn, num_pn, debugmode: bool):

        self.data_paths = []
        self.folder = folder
        self.model = model
        self.debug = debugmode
        self.num_orn = int(num_orn)
        self.num_pn = int(num_pn)

        self.exp_type = which_exp
        self.paras = exp_paras
        # assuming all possible exp define runs and trial numbers
        self.runs = self.paras["noise"]["noiselvl_steps"]
        self.trials = self.paras["iterations_per_noise_lvl"]

        self.path = os.path.join(folder, self.exp_type)

    ################## inputs prep

    def _noise_levels(self):
        n = self.paras["noise"]
        return np.linspace(float(n["noiselvl_min"]),
                           float(n["noiselvl_max"]),
                           int(n["noiselvl_steps"]))

    def _odor_bank(self):
        # hills generated once per sim
        p = self.paras["stimulus"]
        stable = p["odors_stable_params"]
        n_glo = int(p["num_glo"])

        ors = self.model.neuron_populations.get("or")
        if ors is not None and int(ors.num_neurons) != n_glo:
            raise ValueError(
                f"stimulus.num_glo={n_glo} but the 'or' population has "
                f"{int(ors.num_neurons)} neurons. num_glo must equal "
                f"model_parameters.num.glom."
            )

        het = bool(p.get("het", True))
        odors = []
        for key in ("odor1", "odor2"):
            if het:
                od = gauss_odor(
                    n_glo=n_glo,
                    m=float(p[f"{key}_midpoint"]),
                    sd_b=float(p[f"{key}_sd"]),
                    a_rate=float(stable["a_rate"]),
                    A=float(stable["A"]),
                )
            else:
                od = single_glom_odor(
                    n_glo=n_glo,
                    m=float(p[f"{key}_midpoint"]),
                    a_rate=float(stable["a_rate"]),
                    A=float(stable["A"]),
                )
            odors.append(np.copy(od))

        if self.debug:
            active = [int((o[:, 0] > 0).sum()) for o in odors]
            shared = int(((odors[0][:, 0] > 0) & (odors[1][:, 0] > 0)).sum())
            print(f"odors: het={het}, glomeruli with nonzero binding "
                  f"{active}, shared by both odors {shared}")
        if not het and float(p["odor1_midpoint"]) == float(p["odor2_midpoint"]):
            print("WARNING: het=false and both odors share a midpoint, presenting the same odor twice!")

        lo, hi = p["hill_range"]
        rng_seed = p.get("random_seed", None)
        rng = np.random.default_rng(rng_seed) if rng_seed is not None \
            else np.random.default_rng()
        if rng_seed is None:
            print("WARNING: stimulus.random_seed is unset, so hill_exp will differ between trials!")
        hill_exp = rng.uniform(float(lo), float(hi), n_glo)
        return odors, hill_exp

    def _noise_lvl_injecter(self, noise_lvl):

        if self.paras["noise"].get("double_sqrt_dt_correction", False):
            injected = float(noise_lvl) / np.sqrt(self.model.dt)
        else:
            injected = float(noise_lvl)

        for pop in self.paras["noisy_pop"]:
            popobj = self.model.neuron_populations.get(pop)
            if popobj is not None:
                popobj.set_dynamic_param_value("noise_A", injected)
            else:
                print(f"WARNING: noisy_pop '{pop}' is not in the built model")

        if self.debug:
            print(f"noise set to {noise_lvl:.4f} (injected: {injected:.4f})")
        return injected

    def _snapshot_state(self):
        # here copy the network state after creations, so that it can be used to reset the model for every trial
        snap = {"neuron": {}, "synapse": {}}

        skipped = []
        for name, pop in self.model.neuron_populations.items():
            for var_name, var in pop.vars.items():
                try:
                    var.pull_from_device()
                    snap["neuron"][(name, var_name)] = np.copy(var.view)
                except Exception as e:
                    # show if any var cannot be saved
                    skipped.append(f"{name}.{var_name} ({type(e).__name__})")

        for name, sg in self.model.synapse_populations.items():
            out_post = getattr(sg, "out_post", None)
            if out_post is not None:
                try:
                    out_post.pull_from_device()
                    snap["synapse"][name] = np.copy(out_post.view)
                except Exception:
                    continue

        # show which vars cannot be saved. note, none
        if skipped:
            print("WARNING: these variables could not be snapshotted and will "
                  "therefore NOT be reset between presentations:")
            for s in skipped:
                print(f"    {s}")
        if self.debug:
            print(f"state snapshot: {len(snap['neuron'])} neuron vars, "
                  f"{len(snap['synapse'])} synapse buffers")
        if not snap["synapse"]:
            print("WARNING: no per-synapse out_post arrays can be saved "
                  "Synaptic conductance will not be reset between "
                  "presentations; the settle window should be long enough to tolerate it")
        return snap

    def _restore_state(self, snap):
        # copy the initial state, so clean start without rebuilding
        for (pop_name, var_name), values in snap["neuron"].items():
            var = self.model.neuron_populations[pop_name].vars[var_name]
            var.view[:] = values
            var.push_to_device()

        for name, values in snap["synapse"].items():
            out_post = self.model.synapse_populations[name].out_post
            out_post.view[:] = values
            out_post.push_to_device()


    def _check_trials_differ(self, data_log):
        # check that the noisy trials are not bit-bit identical, it would mean that noise is generated everytime from the start
        # (not the case, but better to keep the check)
        seen = defaultdict(set)
        for e in data_log:
            if e["noise_lvl"] == 0.0:
                continue
            seen[(e["level"], e["odor"], e["pop"])].add(e["fingerprint"])

        bad = [k for k, v in seen.items() if len(v) < self.trials]
        if bad and self.trials > 1:
            print("=" * 70)
            print("WARNING: repeated spike trains across trials at nonzero noise:")
            for k in bad[:10]:
                print(f"  level={k[0]} odor={k[1]} pop={k[2]}")
            print("Trials are not independent")
            print("=" * 70)

    def _check_level0_determinism(self, data_log):
        # OUTDATED: used to check if the restore was working (if multiple 0 noise runs with the same network, wo rebuild were bit-bit id)
        # they are, and as of now noise 0 trials are kept to a single run!
        seen = defaultdict(set)
        for e in data_log:
            if e["noise_lvl"] != 0.0:
                continue
            seen[(e["level"], e["odor"], e["pop"])].add(e["fingerprint"])

        bad = [k for k, v in seen.items() if len(v) > 1]
        if bad:
            print("=" * 70)
            print("WARNING: noise-0 repeats are NOT identical for:")
            for k in bad[:10]:
                print(f"  level={k[0]} odor={k[1]} pop={k[2]}")
            print("The state reset is incomplete, so presentations inherit")
            print("history. Lengthen protocol.settle_ms, or check that")
            print("synaptic out_post buffers are host-visible.")
            print("=" * 70)
        elif self.debug and seen:
            print("noise-0 repeats are bit-identical: state reset is complete")

    def _snapshot_connectivity(self, sweep_dir):
        # save connectivity for reproducibility
        syn_pops = getattr(self.model, "synapse_populations", None)
        if not syn_pops:
            print("could not access synapse_populations, skipping snapshot")
            return

        conn_dir = os.path.join(sweep_dir, "connectivity")
        os.makedirs(conn_dir, exist_ok=True)
        for name, sg in syn_pops.items():
            try:
                sg.pull_connectivity_from_device()
                pre = np.asarray(sg.get_sparse_pre_inds())
                post = np.asarray(sg.get_sparse_post_inds())
            except Exception:
                continue
            np.save(os.path.join(conn_dir, f"{name}_pre_inds.npy"), pre)
            np.save(os.path.join(conn_dir, f"{name}_post_inds.npy"), post)
            if self.debug:
                print(f"  {name}: {pre.size} synapses saved")

    ############# RUN
    def run(self):

        data_log = []

        match self.exp_type:

            case "static_single":

                sweep_dir = os.path.join(self.folder, f"pop_n{self.num_orn}")
                os.makedirs(sweep_dir, exist_ok=False)

                noise_lvls = self._noise_levels()
                timeline = ExperimentStatic.timeline(self.paras, self.model.dt)
                settle_steps = int(round(
                    float(self.paras["protocol"].get("level_settle_ms", 0)) / self.model.dt))
                spk = int(self.paras["spk_rec_steps"])
                if settle_steps % spk != 0:
                    raise ValueError(
                        f"level_settle_ms is {settle_steps} timesteps, not a "
                        f"multiple of spk_rec_steps={spk}")

                n_zero = int(np.sum(noise_lvls == 0.0))
                n_presentations = 2 * ((self.runs - n_zero) * self.trials + n_zero)
                total_ms = (n_presentations * timeline["trial_steps"]
                            + self.runs * settle_steps) * self.model.dt
                if self.debug:
                    print(f"noise sweep: {np.round(noise_lvls, 4).tolist()}")
                    print(f"{self.trials} repeats x 2 odors x {self.runs} levels "
                          f"= {n_presentations} presentations, "
                          f"{total_ms / 1000:.1f} s simulated")

                self.model.load(num_recording_timesteps=spk)

                try:
                    odors, hill_exp = self._odor_bank()
                    np.save(os.path.join(sweep_dir, "noise_levels.npy"), noise_lvls)
                    np.save(os.path.join(sweep_dir, "odors.npy"), np.stack(odors))
                    np.save(os.path.join(sweep_dir, "hill.npy"), hill_exp)
                    self._snapshot_connectivity(sweep_dir)

                    snap = self._snapshot_state()

                    start = time.time()
                    step_cursor = 0

                    for i, lvl in enumerate(noise_lvls):

                        lvl_dir = os.path.join(sweep_dir, f"lvl_{i}")
                        os.makedirs(lvl_dir, exist_ok=False)
                        injected = self._noise_lvl_injecter(lvl)

                        # let the noise change settle before the first
                        # presentation of this level
                        for step in range(settle_steps):
                            self.model.step_time()
                            if (step + 1) % spk == 0:
                                self.model.pull_recording_buffers_from_device()
                        step_cursor += settle_steps

                        # noise 0 purely det, so run only once
                        trials_here = 1 if float(lvl) == 0.0 else self.trials
                        if self.debug and trials_here != self.trials:
                            print(f"  noise 0 is deterministic: running "
                                  f"1 trial instead of {self.trials}")

                        for k, odor in enumerate(odors, start=1):

                            odor_dir = os.path.join(lvl_dir, f"odor_{k}")
                            os.makedirs(odor_dir, exist_ok=False)

                            for j in range(trials_here):

                                if self.debug:
                                    print("****************************")
                                    print(f"lvl {i} ({lvl:.4f}), odor {k}, trial {j}")
                                    print("****************************")

                                # re intialize net
                                self._restore_state(snap)

                                exp = ExperimentStatic(
                                    self.model,
                                    self.paras,
                                    os.path.join(odor_dir, f"trial_{j}"),
                                    lvl, i, k, j,
                                    odor, hill_exp,
                                    step_cursor,
                                    self.debug,
                                )
                                trial_log, step_cursor = exp.run()

                                for entry in trial_log:
                                    entry["num_orn"] = self.num_orn
                                    entry["num_pn"] = self.num_pn
                                    entry["noise_lvl_injected"] = injected
                                data_log.extend(trial_log)

                                del exp, trial_log

                finally:
                    self.model.unload()

                if self.debug:
                    took = round(time.time() - start, 2)
                    print(f"static sweep done in {took} s ({took / 60:.2f} min)")

                self._check_level0_determinism(data_log)
                self._check_trials_differ(data_log)

                with open(os.path.join(sweep_dir, "data_log.json"), "w") as fp:
                    json.dump(data_log, fp, indent=2)

                stim_path = None

            case "DynamicSingle":

                stim_path = []
                pop2record = self.paras["pop_to_record"]
                stim_gen = None

                for i in range(self.runs):

                    dirname = os.path.join(self.folder, f"pop_n{self.num_orn}_lvl_{i}")
                    os.makedirs(dirname, exist_ok=False)

                    for j in range(self.trials):

                        if self.debug:
                            print("****************************")
                            print(f"Starting Sim for pop size: {self.num_orn}, lvl: {i}, it: {j}")
                            print("****************************")

                        exp = ExperimentDynamicSingle(self.paras, self.model, stim_gen, self.num_orn, self.num_pn, self.debug)
                        stim, spk_id, spk_t = exp.run(i)

                        for pop in pop2record:

                            spk_id_path = os.path.join(dirname, f"spk_id_pop_n{self.num_orn}_lvl{i}_it{j}_{pop}.npy")
                            spk_t_path = os.path.join(dirname, f"spk_t_pop_n{self.num_orn}_lvl{i}_it{j}_{pop}.npy")

                            if len(spk_id[pop]) > 0:
                                flat_spk_id = np.concatenate(spk_id[pop])
                                flat_spk_t = np.concatenate(spk_t[pop])
                            else:
                                flat_spk_id = np.array([], dtype=float)
                                flat_spk_t = np.array([], dtype=float)

                            np.save(spk_id_path, flat_spk_id)
                            np.save(spk_t_path, flat_spk_t)

                            data_log.append({
                                "num_orn": self.num_orn,
                                "level": i,
                                "trial": j,
                                "pop": pop,
                                "spk_id_path": spk_id_path,
                                "spk_t_path": spk_t_path
                            })

                        if stim_gen is None:
                            # need to generate and save only once, ugly this way
                            stim_path = os.path.join(dirname, "stim.npy")
                            np.save(stim_path, stim)

                            stim_gen = stim

                        del stim, spk_id, spk_t, flat_spk_id, flat_spk_t

            case _: raise ValueError("invalid experiment selected, check json")

        return stim_path, data_log