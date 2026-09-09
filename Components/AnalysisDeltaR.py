import os
import numpy as np
from AnalysisStaticSingle import resolve_events, resolve_windows

############### Firing rates analysis
def glom_rates(spk_t, spk_id, n_glom, n_per_glom, t0, t1):
    """
    Mean firing rate per neuron, per glomerulus
    """
    duration_s = (t1 - t0) / 1000.0
    if spk_t.size == 0:
        return np.zeros(n_glom, dtype=np.float64)

    sel = (spk_t >= t0) & (spk_t < t1)
    if not np.any(sel):
        return np.zeros(n_glom, dtype=np.float64)

    ids = spk_id[sel].astype(np.int64)
    if ids.max() >= n_glom * n_per_glom:
        raise ValueError(
            f"spike id {ids.max()} exceeds population size "
            f"{n_glom * n_per_glom} ({n_glom} glomeruli x {n_per_glom})"
        )
    counts = np.bincount(ids // n_per_glom, minlength=n_glom)
    return counts.astype(np.float64) / (n_per_glom * duration_s)


def bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    """
    CI for Delta<r> via bootstrapping
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        nan = np.full(x.shape[1:], np.nan)
        return nan, nan
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    return (np.percentile(boots, 100 * alpha / 2, axis=0),
            np.percentile(boots, 100 * (1 - alpha / 2), axis=0))

def compute_delta_r(data_num, paras_an, paras_model, protocol, noise_lvls,
                    debugmode=False):
    """
    Returns in the form:
      {pop}_{period}_rate        (n_levels, n_odors, n_trials, n_glom)
      {pop}_{period}_delta       (n_levels, n_trials, n_glom)
      {pop}_{period}_delta_mean  (n_levels, n_glom)
      {pop}_{period}_delta_ci_low / _ci_highh
    """
    n_glom = int(paras_model["num"]["glom"])
    per_glom = {"orn": int(paras_model["num"]["orn"]),
                "pn": int(paras_model["num"]["pn"])}
    pops = list(paras_an["pop_to_analyze"])

    stim_name = paras_an.get("delta_stim_window", "steady_state")
    windows = resolve_windows(protocol, paras_an["windows"])
    if stim_name not in windows:
        raise ValueError(
            f"delta_stim_window '{stim_name}' is not defined in "
            f"analysis_parameters.windows (have {sorted(windows)})")
    periods = {"baseline": windows["baseline"], "stimulation": windows[stim_name]}

    levels = sorted(data_num)
    odors = sorted(data_num[levels[0]])
    if len(odors) != 2:
        raise ValueError(f"Delta<r> needs exactly 2 odors, found {odors}")

    n_tr_lvl = {}
    for lvl in levels:
        counts = {od: len(data_num[lvl][od]) for od in odors}
        # check for different counts among odors in the same level (useless!)
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"level {lvl} has unequal trial counts across odors: {counts}")
        n_tr_lvl[lvl] = counts[odors[0]]
    max_tr = max(n_tr_lvl.values())

    rng = np.random.default_rng(0)
    out = {
        "noise_lvls": np.asarray(noise_lvls, dtype=float),
        "levels": np.array(levels),
        "odors": np.array(odors),
        "n_trials": np.array([n_tr_lvl[l] for l in levels]),
        "glom": np.arange(n_glom),
        "stim_window_name": np.array(stim_name),
        "windows_ms": np.array([[periods["baseline"][0], periods["baseline"][1]],
                                [periods["stimulation"][0], periods["stimulation"][1]]]),
    }

    thin = [l for l in levels if 1 < n_tr_lvl[l] < 3]
    if thin:
        print(f"WARNING: levels {thin} have less than 3 trials. Confidence intervals are not computed")

    for pop in pops:
        npg = per_glom[pop]

        for period, (t0, t1) in periods.items():
            # pad noise 0 with nans, to match dim of other noise lvls
            rate = np.full((len(levels), len(odors), max_tr, n_glom), np.nan)

            for li, lvl in enumerate(levels):
                for oi, od in enumerate(odors):
                    for ti, tr in enumerate(sorted(data_num[lvl][od])):
                        paths = data_num[lvl][od][tr].get(pop)
                        if paths is None:
                            raise KeyError(
                                f"no {pop} data for level {lvl} odor {od} trial {tr}")
                        spk_t = np.load(paths["spk_t_path"])
                        spk_id = np.load(paths["spk_id_path"])
                        rate[li, oi, ti] = glom_rates(
                            spk_t, spk_id, n_glom, npg, t0, t1)
                        del spk_t, spk_id

            delta = rate[:, 0] - rate[:, 1]

            lo = np.full((len(levels), n_glom), np.nan)
            hi = np.full((len(levels), n_glom), np.nan)
            for li, lvl in enumerate(levels):
                n = n_tr_lvl[lvl]

                if n >= 3:
                    lo[li], hi[li] = bootstrap_ci(delta[li, :n], rng=rng)

            out[f"{pop}_{period}_rate"] = rate
            out[f"{pop}_{period}_delta"] = delta
            out[f"{pop}_{period}_delta_mean"] = np.nanmean(delta, axis=1)
            out[f"{pop}_{period}_delta_ci_lo"] = lo
            out[f"{pop}_{period}_delta_ci_hi"] = hi

            if debugmode:
                m = np.nanmean(delta, axis=1)
                print(f"{pop} {period:11s} [{t0:.0f},{t1:.0f}] ms  "
                      f"|Delta<r>| max {np.abs(m).max():7.3f} Hz  "
                      f"mean {m.mean():+7.4f} Hz  "
                      f"(per level max: "
                      f"{', '.join(f'{v:.2f}' for v in np.abs(m).max(axis=1))})")

    return out


def save_delta_results(out, folder, prefix="static"):
    path = os.path.join(folder, f"{prefix}_delta_r.npz")
    np.savez_compressed(path, **out)
    return [path]


#################### Plotting

def plot_delta_r(out, folder=None, stem=True):
    import matplotlib.pyplot as plt

    levels = [int(l) for l in np.atleast_1d(out["levels"])]
    noise = np.atleast_1d(out["noise_lvls"])
    glom = np.atleast_1d(out["glom"]).astype(float)
    n_tr = np.atleast_1d(out["n_trials"])
    pops = sorted({k.split("_stimulation_delta_mean")[0] for k in out
                   if k.endswith("_stimulation_delta_mean")})

    n_lvl = len(levels)
    # dodge is used to not make stem overlap for different noise lvls fro the same glom
    dodge = 0.0 if (n_lvl == 1 or not stem) else min(0.30, 0.8 / n_lvl)
    cap = max(0.18, dodge * 0.45)

    for pop in pops:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)

        for ax, period in zip(axes, ("stimulation", "baseline")):
            mean = np.atleast_2d(out[f"{pop}_{period}_delta_mean"])
            lo = np.atleast_2d(out[f"{pop}_{period}_delta_ci_lo"])
            hi = np.atleast_2d(out[f"{pop}_{period}_delta_ci_hi"])

            for li, lvl in enumerate(levels):
                c = np.array(plt.cm.viridis(li / max(1, n_lvl - 1)))
                light = 1.0 - 0.55 * (1.0 - c)
                light[3] = 1.0
                x = glom + (li - (n_lvl - 1) / 2.0) * dodge
                label = f"noise {noise[lvl]:.2f} (n={int(n_tr[li])})"
                ok = np.isfinite(lo[li]) & np.isfinite(hi[li])

                if stem:
                    ax.vlines(x, 0.0, mean[li], color=c, lw=1.0, zorder=2)
                    ax.plot(x, mean[li], "o", ms=3.0, color=c, mec=c, zorder=3,
                            label=label)
                    if ok.any():
                        ax.hlines(hi[li][ok], x[ok] - cap, x[ok] + cap,
                                  color=light, lw=1.0, zorder=1)
                        ax.hlines(lo[li][ok], x[ok] - cap, x[ok] + cap,
                                  color=light, lw=1.0, zorder=1)
                else:
                    ax.plot(x, mean[li], color=c, lw=1.4, zorder=2, label=label)
                    if ok.any():

                        ax.fill_between(x, lo[li], hi[li], where=ok,
                                        color=c, alpha=0.25, linewidth=0,
                                        zorder=1)

            ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=0)
            ax.set_title(f"{pop.upper()} - {period}")
            ax.set_xlabel("glomerulus")

        axes[0].set_ylabel(r"$\Delta\langle r\rangle$  odor1 - odor2  (Hz)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        if folder is not None:
            suffix = "stem" if stem else "line"
            fig.savefig(os.path.join(folder, f"delta_r_{pop}_{suffix}.png"), dpi=300)

    plt.show()
