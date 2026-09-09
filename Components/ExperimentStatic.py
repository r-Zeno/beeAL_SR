import numpy as np
import time
import os
import json
import hashlib
from helpers import set_odor_simple


class ExperimentStatic:
    """
    Runs the presentation of one odor at a given level. called by experimenter the necessary number of times, with the correct inputs
    """

    def __init__(self, model, experiment_parameters: dict, trial_folder: str,
                 noise_lvl: float, n_run: int, n_odor: int, n_trial: int,
                 odor, hill_exp, start_step: int, debugmode: bool):

        self.model = model
        self.paras = experiment_parameters
        self.folder = trial_folder
        self.noise_lvl = float(noise_lvl)
        self.n_run = int(n_run)
        self.n_odor = int(n_odor)
        self.n_trial = int(n_trial)
        self.odor = odor
        self.hill_exp = hill_exp
        self.start_step = int(start_step)
        self.debug = debugmode

        self.rec_states = self.paras["rec_states"]
        self.spk_rec_steps = int(self.paras["spk_rec_steps"])
        self.pop_to_rec = self.paras["pop_to_rec"]
        self.what_to_rec = self.paras["what_to_rec"]
        self.state_rec_steps = int(self.paras.get("state_rec_steps", 10))
        self.odor_slot = int(self.paras["stimulus"].get("odor_slot", 0))

        self.run_settings = dict()
        self.t_start_ms = self.start_step * self.model.dt

        os.makedirs(self.folder, exist_ok=True)

    @staticmethod
    def timeline(paras, dt):
        """
        this creates a timesteps grid on which spikes are assigned. using model.t and the reported timing instead would
        work allthesame, and be easier, so never do this again.
        Initially this was thought to be necessary since the experiments where to be run on a never restarting model instance, so model.t would accumulate float errors.
        But model.t is restarted at every reloading of the network, so completely useless
        """
        p = paras["protocol"]

        def steps(ms):
            n = float(ms) / dt
            if abs(n - round(n)) > 1e-9:
                raise ValueError(f"{ms} ms is not an integer number of dt={dt} ms steps")
            return int(round(n))

        n_settle = steps(p.get("settle_ms", 0))
        n_base = steps(p["baseline_ms"])
        n_stim = steps(p["stim_ms"])
        n_relax = steps(p["relaxation_ms"])

        t = {
            "settle_end": n_settle,
            "odor_on": n_settle + n_base,
            "odor_off": n_settle + n_base + n_stim,
        }
        t["trial_steps"] = t["odor_off"] + n_relax

        # check if spike time pulling is a multiple of the total time
        spk = int(paras["spk_rec_steps"])
        if t["trial_steps"] % spk != 0:
            raise ValueError(
                f"presentation is {t['trial_steps']} timesteps, not a multiple "
                f"of spk_rec_steps={spk}"
            )
        return t

    def _concentration(self):
        s = self.paras["stimulus"]
        base = np.power(10.0, float(s["base_exp"]))
        return float(s["scale"]) * np.power(base, float(s["c"])), 0.0

    def _rec_var_init(self):
        spike_t = {pop: [] for pop in self.pop_to_rec}
        spike_id = {pop: [] for pop in self.pop_to_rec}
        self.vars_rec = {f"{pop}_{var}": [] for pop, var in self.what_to_rec}
        return spike_t, spike_id

    def _var_views(self):
        return {f"{pop}_{var}": self.model.neuron_populations[pop].vars[var].view
                for pop, var in self.what_to_rec}

    ############### Pulling spikes

    def _pull(self, int_t, spike_t, spike_id, var_view):
        if self.rec_states and self.what_to_rec and int_t % self.state_rec_steps == 0:
            for pop_name, var_name in self.what_to_rec:
                key = f"{pop_name}_{var_name}"
                self.model.neuron_populations[pop_name].vars[var_name].pull_from_device()
                self.vars_rec[key].append(np.copy(var_view.get(key)))

        if int_t % self.spk_rec_steps == 0:
            self.model.pull_recording_buffers_from_device()
            for pop in self.pop_to_rec:
                data = self.model.neuron_populations[pop].spike_recording_data[0]
                if data[0].size > 0:
                    spike_t[pop].append(data[0])
                    spike_id[pop].append(data[1])

    def run(self):
        """
        Steps the model through one presentation and saves the spikes.
        Returns (log_entries, end_step).
        """
        t = self.timeline(self.paras, self.model.dt)
        on, off = self._concentration()

        model_step = getattr(self.model, "timestep", None)
        if model_step is not None and int(model_step) != self.start_step:
            print(f"WARNING: model.timestep={int(model_step)} but segment expects "
                  f"start_step={self.start_step}; spike times may be misaligned")

        spike_t, spike_id = self._rec_var_init()
        var_view = self._var_views()
        ors_population = self.model.neuron_populations["or"]

        # The state restore has already put the receptors back to their initial
        # (odor-off) values, so no explicit off-write is needed here.
        self.run_settings.update({
            "noise_lvl": self.noise_lvl,
            "level": self.n_run,
            "odor": self.n_odor,
            "trial": self.n_trial,
            "start_step": self.start_step,
            "t_start_ms": float(self.t_start_ms),
            "timeline_steps": {k: int(v) for k, v in t.items()},
            "dt": float(self.model.dt),
        })

        start = time.time()
        for step in range(t["trial_steps"]):

            if step == t["odor_on"]:
                if self.debug:
                    print(f"  step {step}: odor {self.n_odor} on")
                set_odor_simple(ors_population, self.odor_slot, self.odor, on, self.hill_exp)

            elif step == t["odor_off"]:
                if self.debug:
                    print(f"  step {step}: odor {self.n_odor} off")
                set_odor_simple(ors_population, self.odor_slot, self.odor, off, self.hill_exp)

            self.model.step_time()
            self._pull(step + 1, spike_t, spike_id, var_view)

        if self.debug:
            print(f"  presentation ran in {round(time.time() - start, 2)} s")

        log = self._data_saver(spike_t, spike_id)
        return log, self.start_step + t["trial_steps"]

        ################ Saver

    def _data_saver(self, spike_t, spike_id):
        os.makedirs(self.folder, exist_ok=True)
        log = []
        dt = float(self.model.dt)
        max_resid = 0.0

        for pop in self.pop_to_rec:
            if len(spike_t[pop]) > 0:
                # this is a somewhat contrived and unnecessary solution to assign spikes at precise timings in a grid 
                rel = (np.hstack(spike_t[pop]).astype(np.float64)
                       - self.t_start_ms) / dt
                steps = np.rint(rel).astype(np.int64)
                if steps.size:
                    max_resid = max(max_resid, float(np.abs(rel - steps).max()))
                flat_t = steps.astype(np.float64) * dt
                flat_id = np.hstack(spike_id[pop])
            else:
                steps = np.array([], dtype=np.int64)
                flat_t = np.array([], dtype=np.float64)
                flat_id = np.array([], dtype=int)

            t_path = os.path.join(self.folder, f"{pop}_spike_t.npy")
            id_path = os.path.join(self.folder, f"{pop}_spike_id.npy")
            np.save(t_path, flat_t)
            np.save(id_path, flat_id)

            log.append({
                "level": self.n_run,
                "noise_lvl": self.noise_lvl,
                "odor": self.n_odor,
                "trial": self.n_trial,
                "pop": pop,
                "n_spikes": int(flat_t.size),
                # to check if runs are bit-bit identical
                "fingerprint": hashlib.md5(
                    steps.tobytes() + np.asarray(flat_id, dtype=np.int64).tobytes()
                ).hexdigest()[:16],
                "spk_t_path": t_path,
                "spk_id_path": id_path,
            })

        if self.rec_states:
            for key, segs in self.vars_rec.items():
                if segs:
                    np.save(os.path.join(self.folder, f"{key}_states.npy"), np.vstack(segs))

        # check of the grid alignment precision, this is an ai suggested check that is completely useless, but does no harm
        self.run_settings["max_grid_residual_steps"] = max_resid
        if max_resid > 0.25:
            print(f"WARNING: spike times where {max_resid:.3f} timesteps off "
                  f"the dt grid before snapping. Check that ModelBuilder passes "
                  f"time_precision='double'.")

        with open(os.path.join(self.folder, "run_settings.json"), "w") as fp:
            json.dump(self.run_settings, fp, indent=2)

        return log
