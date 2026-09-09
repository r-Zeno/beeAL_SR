import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


def resolve_events(protocol):
    # windowing
    settle = float(protocol.get("settle_ms", 0.0))
    base = float(protocol["baseline_ms"])
    stim = float(protocol["stim_ms"])
    relax = float(protocol["relaxation_ms"])

    ev = {"trial_start": 0.0, "settle_end": settle}
    ev["odor_on"] = settle + base
    ev["odor_off"] = ev["odor_on"] + stim
    ev["trial_end"] = ev["odor_off"] + relax
    return ev


def resolve_windows(protocol, windows_cfg):
    ev = resolve_events(protocol)

    def edge(spec):
        anchor, offset = spec[0], float(spec[1])
        if anchor not in ev:
            raise ValueError(f"unknown window anchor '{anchor}'; "
                             f"available: {sorted(ev)}")
        return ev[anchor] + offset

    out = {}
    for name, w in windows_cfg.items():
        if name.startswith("_"):
            continue
        a, b = edge(w["start"]), edge(w["end"])
        if not (0.0 <= a < b <= ev["trial_end"]):
            raise ValueError(
                f"window '{name}' resolves to [{a}, {b}] ms, outside the "
                f"presentation [0, {ev['trial_end']}] ms"
            )
        if a < ev["settle_end"]:
            raise ValueError(
                f"window '{name}' starts at {a} ms, inside the {ev['settle_end']} ms"
            )
        out[name] = (a, b)
    return out


# ------------------------------------------------------------------------ sdf

def sdf_matrix(spk_t, spk_id, n_neurons, t_end, dt_ms, sigma_ms, dtype=np.float32):
    """
    Spike density function vectorized for speed, needed for orns
    """
    n_bins = int(round(t_end / dt_ms))
    counts = np.zeros((n_bins, n_neurons), dtype=np.float64)

    if spk_t.size > 0:
        sel = (spk_t >= 0.0) & (spk_t < t_end)
        if np.any(sel):
            bins = (spk_t[sel] / dt_ms).astype(np.int64)
            np.clip(bins, 0, n_bins - 1, out=bins)
            flat = bins * n_neurons + spk_id[sel].astype(np.int64)
            counts = np.bincount(
                flat, minlength=n_bins * n_neurons
            ).astype(np.float64).reshape(n_bins, n_neurons)

    sigma_bins = sigma_ms / dt_ms
    # truncate=3 matches make_sdf half-width = 3*sigma
    smooth = gaussian_filter1d(counts, sigma_bins, axis=0, mode="constant", truncate=3.0)
    return (smooth * (1000.0 / dt_ms)).astype(dtype)


def sdf_slice(sdf, dt_ms, t0, t1):
    # index by ms
    return sdf[int(round(t0 / dt_ms)):int(round(t1 / dt_ms))]


def glom_reduce(x, n_per_glom, axis=-1):
    # mean over glom
    x = np.asarray(x)
    n = x.shape[axis]
    if n % n_per_glom != 0:
        raise ValueError(f"{n} neurons is not a multiple of {n_per_glom} per glomerulus")
    x = np.moveaxis(x, axis, -1)
    x = x.reshape(*x.shape[:-1], n // n_per_glom, n_per_glom).mean(axis=-1)
    return np.moveaxis(x, -1, axis)

def fit_clean_pca(traj_a, traj_b, n_components=3, center=True):
    # find basis of pca space (that is, do svd)
    X = np.vstack([traj_a, traj_b]).astype(np.float64)
    mean = X.mean(axis=0) if center else None
    Xc = X - mean if center else X
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    total = float((Xc ** 2).sum())
    captured = float((S[:n_components] ** 2).sum())
    return comps, mean, (captured / total if total > 0 else np.nan)


def project(traj, comps, mean=None):
    # projection on pca space
    X = np.asarray(traj, dtype=np.float64)
    if mean is not None:
        X = X - mean
    return X @ comps.T


def subspace_capture(traj, comps, mean=None):
    # compute captured variance
    X = np.asarray(traj, dtype=np.float64)
    if mean is not None:
        X = X - mean
    total = float((X ** 2).sum())
    if total <= 0:
        return np.nan
    return float(((X @ comps.T) ** 2).sum() / total)

def analysis_trial_static_single(level, noise_lvl, odor, trial, trial_data, paras_an,
                                 paras_model, protocol, out_dir, debugmode=False):
    """
    computes one trial's sdf for each population, saves it, returs it as a path
    """
    n_glom = int(paras_model["num"]["glom"])
    per_glom = {"orn": int(paras_model["num"]["orn"]),
                "pn": int(paras_model["num"]["pn"])}
    sizes = {pop: n_glom * npg for pop, npg in per_glom.items()}

    dt_ms = float(paras_an["sdf"]["dt_ms"])
    sigma_ms = float(paras_an["sdf"]["sigma_ms"])
    t_end = resolve_events(protocol)["trial_end"]

    res = {
        "level": int(level),
        "noise_lvl": float(noise_lvl),
        "odor": int(odor),
        "trial": int(trial),
        "sdf_paths": {},
    }

    for pop in paras_an["pop_to_analyze"]:
        if pop not in trial_data:
            continue

        spk_t = np.load(trial_data[pop]["spk_t_path"])
        spk_id = np.load(trial_data[pop]["spk_id_path"]).astype(np.int64)

        if pop == "pn":
            sdf = sdf_matrix(spk_t, spk_id, sizes[pop], t_end, dt_ms, sigma_ms)
        else:
            # orns are 12x more than pn, so take mean over gloms
            glom_id = spk_id // per_glom[pop]
            sdf = sdf_matrix(spk_t, glom_id, n_glom, t_end, dt_ms, sigma_ms)
            sdf = sdf / per_glom[pop]

        path = os.path.join(out_dir, f"sdf_{pop}_lvl{level}_od{odor}_tr{trial}.npy")
        np.save(path, sdf)
        res["sdf_paths"][pop] = path

        if debugmode:
            print(f"  lvl {level} odor {odor} trial {trial} {pop}: "
                  f"{spk_t.size} spikes, sdf {sdf.shape}")

        del spk_t, spk_id, sdf

    return res

def aggregate_static_single(results, sweep_dir, paras_an, paras_model, protocol,
                            noise_lvls, keep_trial_sdf=False, debugmode=False):
    # trial averaged 

    dt_ms = float(paras_an["sdf"]["dt_ms"])
    n_comp = int(paras_an.get("pca_components", 3))
    center = bool(paras_an.get("pca_center", True))
    pca_window = paras_an.get("pca_window", "full")
    fit_on = paras_an.get("pca_fit", "clean")
    use_sqrt = bool(paras_an.get("sqrt_transform", False))
    ev = resolve_events(protocol)
    base_win = resolve_windows(protocol, paras_an["windows"])["baseline"]

    # checks if the window limits specified in the json comply with the 3*sigma window overlap from the sdf function
    sigma_ms = float(paras_an["sdf"]["sigma_ms"])
    guard = 3.0 * sigma_ms
    if base_win[0] < ev["settle_end"] + guard:
        print(f"WARNING: baseline window starts at {base_win[0]:.0f} ms but the "
              f"settle period ends at {ev['settle_end']:.0f} ms. With "
              f"sigma={sigma_ms:.0f} ms the reset transient enters till "
              f"{guard:.0f} ms past the cut. Start the baseline at "
              f">= {ev['settle_end'] + guard:.0f} ms.")
    if base_win[1] > ev["odor_on"] - guard:
        print(f"WARNING: baseline window ends at {base_win[1]:.0f} ms, within "
              f"{guard:.0f} ms of odor onset at {ev['odor_on']:.0f} ms. "
              f"smoothed baseline will be biased "
              f"End the baseline at <= {ev['odor_on'] - guard:.0f} ms.")

    if pca_window not in ("full", "stim"):
        raise ValueError(f"pca_window must be 'full' or 'stim', got {pca_window!r}")
    if fit_on not in ("clean", "pooled"):
        raise ValueError(f"pca_fit must be 'clean' or 'pooled', got {fit_on!r}")

    by_key = {}
    for r in results:
        by_key.setdefault((r["level"], r["odor"]), []).append(r)
    levels = sorted({k[0] for k in by_key})
    odors = sorted({k[1] for k in by_key})

    pops = [p for p in paras_an["pop_to_analyze"]
            if any(p in r["sdf_paths"] for r in results)]

    out = {
        "noise_lvls": np.asarray(noise_lvls, dtype=float),
        "levels": np.array(levels),
        "odors": np.array(odors),
        "dt_ms": np.array(dt_ms),
        "n_trials": np.array([len(by_key[(l, odors[0])]) for l in levels]),
        "pca_center": np.array(center),
        "pca_window": np.array(pca_window),
        "pca_fit": np.array(fit_on),
        "sqrt_transform": np.array(use_sqrt),
    }

    for pop in pops:

        # trajectories are averaged over trials
        split = {}
        for lvl in levels:
            for odor in odors:
                acc = None
                for r in by_key[(lvl, odor)]:
                    sdf = np.load(r["sdf_paths"][pop]).astype(np.float64)
                    if use_sqrt:
                        # in a previous version i used sqrt correction to control for higher firing rates, but its harmful since we already baseline subtract
                        sdf = np.sqrt(sdf)
                    sdf = sdf - sdf_slice(sdf, dt_ms, *base_win).mean(axis=0)
                    acc = sdf if acc is None else acc + sdf
                mean_traj = acc / len(by_key[(lvl, odor)])

                # full includes also the baseline period, otherwise only the stimulation period is used for trajs
                seg = ((base_win[0], ev["odor_off"]) if pca_window == "full"
                       else (ev["odor_on"], ev["odor_off"]))
                split[(lvl, odor)] = sdf_slice(mean_traj, dt_ms, *seg)
                del acc, mean_traj

        clean = levels[0]
        if fit_on == "clean":
            fit_a, fit_b = split[(clean, odors[0])], split[(clean, odors[1])]
        else:
            # if not clean, fit on all noise levels, this is not good, as the noise dominates the response space and response at lvl = 0 is artificially flattened (since it has no power in the noise dim)
            stacked = [split[(l, o)] for l in levels for o in odors]
            half = len(stacked) // 2
            fit_a = np.vstack(stacked[:half])
            fit_b = np.vstack(stacked[half:])

        comps, pca_mean, evr = fit_clean_pca(fit_a, fit_b, n_comp, center=center)
        out[f"{pop}_pca_components"] = comps
        out[f"{pop}_pca_mean"] = pca_mean if pca_mean is not None else np.array([])
        out[f"{pop}_pca_explained_clean"] = np.array(evr)

        for lvl in levels:
            for odor in odors:
                out[f"{pop}_traj_lvl{lvl}_od{odor}"] = project(
                    split[(lvl, odor)], comps, pca_mean)
            out[f"{pop}_subspace_capture_lvl{lvl}"] = np.array(np.mean([
                subspace_capture(split[(lvl, odor)], comps, pca_mean)
                for odor in odors
            ]))

            # check how different are the individual odor trajs, as the variance comp before only tells explained var for both
            if len(odors) >= 2:
                diff = split[(lvl, odors[0])] - split[(lvl, odors[1])]
                tot = float((diff ** 2).sum())
                out[f"{pop}_contrast_in_basis_lvl{lvl}"] = np.array(
                    float(((diff @ comps.T) ** 2).sum() / tot) if tot > 0 else np.nan)

                both = np.vstack([split[(lvl, o)] for o in odors])
                bt = float(((both - both.mean(0)) ** 2).sum())
                out[f"{pop}_contrast_var_frac_lvl{lvl}"] = np.array(
                    float(tot / bt) if bt > 0 else np.nan)

        # check for saturation
        out[f"{pop}_mean_evoked_rate"] = np.array([
            float(np.mean([split[(lvl, o)].mean() for o in odors])) for lvl in levels
        ])

        if debugmode:
            hi = levels[-1]
            print(f"{pop}: clean PCs explain {evr:.3f} of clean variance; "
                  f"subspace capture {float(out[f'{pop}_subspace_capture_lvl{clean}']):.3f} "
                  f"(clean) -> {float(out[f'{pop}_subspace_capture_lvl{hi}']):.3f} (noisiest)")
            if f"{pop}_contrast_in_basis_lvl{clean}" in out:
                print(f"{pop}: odor contrast is "
                      f"{float(out[f'{pop}_contrast_var_frac_lvl{clean}'])*100:.2f}% of "
                      f"total variance; fraction of it retained in the basis "
                      f"{float(out[f'{pop}_contrast_in_basis_lvl{clean}']):.3f} (clean) -> "
                      f"{float(out[f'{pop}_contrast_in_basis_lvl{hi}']):.3f} (noisiest)")

    if not keep_trial_sdf:
        for r in results:
            for path in r.get("sdf_paths", {}).values():
                try:
                    os.remove(path)
                except OSError:
                    pass

    return out


def save_results(out, folder, prefix="static"):
    npz = os.path.join(folder, f"{prefix}_pca.npz")
    np.savez_compressed(npz, **out)
    return [npz]

def data_log_compile_static(data_log):
    # as the helpers onem but with odor info
    out = {}
    for e in data_log:
        num, lvl, od, tr, pop = (e["num_orn"], e["level"], e["odor"],
                                 e["trial"], e["pop"])
        out.setdefault(num, {}).setdefault(lvl, {}).setdefault(od, {}) \
           .setdefault(tr, {})[pop] = {
               "spk_id_path": e["spk_id_path"],
               "spk_t_path": e["spk_t_path"],
           }
    return out


############## plotting

def plot_debug_trajectories(out, folder=None):
    # debug plot, trajectories for max and min noise levels
    import matplotlib.pyplot as plt

    levels = [int(l) for l in np.atleast_1d(out["levels"])]
    noise = np.atleast_1d(out["noise_lvls"])
    odors = [int(o) for o in np.atleast_1d(out["odors"])]
    lo, hi = levels[0], levels[-1]
    pops = sorted({k[:-len("_pca_components")] for k in out
                   if k.endswith("_pca_components")})

    for pop in pops:
        n_dim = np.asarray(out[f"{pop}_traj_lvl{lo}_od{odors[0]}"]).shape[1]
        three_d = n_dim >= 3
        if n_dim < 2:
            continue

        fig = plt.figure(figsize=(11, 5))
        axes = []
        for i, lvl in enumerate((lo, hi)):
            ax = (fig.add_subplot(1, 2, i + 1, projection="3d") if three_d
                  else fig.add_subplot(1, 2, i + 1))
            for odor, colour in zip(odors, ("tab:blue", "tab:red")):
                t = np.asarray(out[f"{pop}_traj_lvl{lvl}_od{odor}"])
                args = (t[:, 0], t[:, 1], t[:, 2]) if three_d else (t[:, 0], t[:, 1])
                ax.plot(*args, color=colour, lw=1.5, label=f"odor {odor}")
                ax.scatter(*[[a[0]] for a in args], color=colour, s=25)
            cap = float(np.atleast_1d(out[f"{pop}_subspace_capture_lvl{lvl}"])[0])
            ax.set_title(f"{pop} - noise {noise[lvl]:.2f} (capture {cap:.2f})")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            if three_d:
                ax.set_zlabel("PC 3")
            if i == 0:
                ax.legend(fontsize=8)
            axes.append(ax)

        # shared limits across both panels
        lims = [(min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes)),
                (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))]
        if three_d:
            lims.append((min(a.get_zlim()[0] for a in axes),
                         max(a.get_zlim()[1] for a in axes)))
        for a in axes:
            a.set_xlim(lims[0])
            a.set_ylim(lims[1])
            if three_d:
                a.set_zlim(lims[2])

        fig.tight_layout()
        if folder is not None:
            fig.savefig(os.path.join(folder, f"debug_traj_{pop}.png"), dpi=300)

    plt.show()
