"""Assemble Q / FAQ / PG genuine-anticipation maps into a 3-panel figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rules = [("q", "Q-learning (k=100)"),
         ("faq", "FAQ (k=1500)"),
         ("pg", "Policy gradient (k=300)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ims = []
for ax, (key, title) in zip(axes, rules):
    d = np.load(f"map_{key}.npz")
    GA, rs, bs = d["GA"], d["rs"], d["bs"]
    ext = [rs[0], rs[-1], bs[0], bs[-1]]
    im = ax.imshow(GA, origin="lower", aspect="auto", extent=ext,
                   cmap="magma", vmin=0.5, vmax=1.0)
    ims.append(im)
    ax.set_title(title); ax.set_xlabel("r (anti-coordination reward)")
    # hatch non-bistable region lightly by overlaying where GA is NaN
    nanmask = np.isnan(GA)
    ax.imshow(np.where(nanmask, 1.0, np.nan), origin="lower", aspect="auto",
              extent=ext, cmap="Greys", vmin=0, vmax=1, alpha=0.12)
axes[0].set_ylabel(r"$\beta$ (cyclic asymmetry)")
cb = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)
cb.set_label("genuine anticipation AUC (undecided subset)")
fig.suptitle("Where early prediction of equilibrium selection is genuine, by learning rule\n"
             "(grey = not bistable; bright = genuine anticipation; dark = illusory)",
             y=1.04)
fig.savefig("fig_maps_3rules.png", dpi=140, bbox_inches="tight")
print("saved fig_maps_3rules.png")

# quick numeric summary
for key, title in rules:
    d = np.load(f"map_{key}.npz"); GA, bs = d["GA"], d["bs"]
    lowb = GA[bs <= 0.06]; highb = GA[bs >= 0.30]
    lm = np.nanmean(lowb) if np.any(~np.isnan(lowb)) else np.nan
    hm = np.nanmean(highb) if np.any(~np.isnan(highb)) else np.nan
    nlow = np.sum(~np.isnan(lowb)); nhigh = np.sum(~np.isnan(highb))
    print(f"{title:>26}: low-beta mean AUC={lm:.3f} (n={nlow}) | "
          f"high-beta mean AUC={hm:.3f} (n={nhigh})")
