"""Read actual numerical values from flame_param .npz files for all 12 motions."""
import modal
import os

app = modal.App("lam-motion-values")
volume = modal.Volume.from_name("lam-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy==1.23.0")

@app.function(volumes={"/vol": volume}, timeout=300, image=image)
def dump_values():
    import numpy as np

    base = "/vol/assets/sample_motion/export"
    motions = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    # First: determine keys from first file
    first_dir = os.path.join(base, motions[0], "flame_param")
    first_file = sorted(os.listdir(first_dir))[0]
    sample = np.load(os.path.join(first_dir, first_file))
    keys = list(sample.keys())
    print(f"NPZ keys: {keys}")
    for k in keys:
        print(f"  {k}: shape={sample[k].shape}, dtype={sample[k].dtype}")
    print()

    # For each motion: load ALL frames, compute stats per parameter
    print("=" * 120)
    print(f"{'Motion':<30} | {'expression (10D)':<55} | {'rotation (3D)':<35} | {'translation (3D)':<35}")
    print(f"{'':30} | {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'dims':>5} | {'min':>8} {'max':>8} {'mean':>8} | {'min':>8} {'max':>8} {'mean':>8}")
    print("-" * 120)

    for m in motions:
        fp_dir = os.path.join(base, m, "flame_param")
        files = sorted(os.listdir(fp_dir))

        exprs, rots, trans = [], [], []
        for f in files:
            d = np.load(os.path.join(fp_dir, f))
            # Try common key names
            for ek in ["expression", "expr"]:
                if ek in d:
                    exprs.append(d[ek].flatten())
                    break
            for rk in ["rotation", "pose"]:
                if rk in d:
                    rots.append(d[rk].flatten())
                    break
            for tk in ["translation"]:
                if tk in d:
                    trans.append(d[tk].flatten())
                    break

        if exprs:
            ea = np.array(exprs)
            expr_str = f"{ea.min():>8.3f} {ea.max():>8.3f} {ea.mean():>8.3f} {ea.std():>8.3f} {ea.shape[1]:>5}"
        else:
            expr_str = "N/A"

        if rots:
            ra = np.array(rots)
            rot_str = f"{ra.min():>8.3f} {ra.max():>8.3f} {ra.mean():>8.3f}"
        else:
            rot_str = "N/A"

        if trans:
            ta = np.array(trans)
            trans_str = f"{ta.min():>8.4f} {ta.max():>8.4f} {ta.mean():>8.4f}"
        else:
            trans_str = "N/A"

        print(f"{m:<30} | {expr_str} | {rot_str} | {trans_str}")

    # Detailed per-dimension breakdown for expression (first motion as example)
    print(f"\n{'=' * 100}")
    print(f"Expression 10D per-dimension breakdown (all motions)")
    print(f"{'=' * 100}")
    print(f"{'Motion':<25} | {'dim0':>6} {'dim1':>6} {'dim2':>6} {'dim3':>6} {'dim4':>6} {'dim5':>6} {'dim6':>6} {'dim7':>6} {'dim8':>6} {'dim9':>6}")
    print("-" * 100)

    for m in motions:
        fp_dir = os.path.join(base, m, "flame_param")
        files = sorted(os.listdir(fp_dir))
        exprs = []
        for f in files:
            d = np.load(os.path.join(fp_dir, f))
            for ek in ["expression", "expr"]:
                if ek in d:
                    exprs.append(d[ek].flatten())
                    break
        if exprs:
            ea = np.array(exprs)
            dims = min(ea.shape[1], 10)
            means = ea.mean(axis=0)[:dims]
            vals = " ".join(f"{v:>6.3f}" for v in means)
            print(f"{m:<25} | {vals}")

    # Rotation per-dimension
    print(f"\n{'=' * 60}")
    print(f"Rotation 3D per-dimension mean (all motions)")
    print(f"{'=' * 60}")
    print(f"{'Motion':<25} | {'yaw':>8} {'pitch':>8} {'roll':>8}")
    print("-" * 60)

    for m in motions:
        fp_dir = os.path.join(base, m, "flame_param")
        files = sorted(os.listdir(fp_dir))
        rots = []
        for f in files:
            d = np.load(os.path.join(fp_dir, f))
            for rk in ["rotation", "pose"]:
                if rk in d:
                    rots.append(d[rk].flatten())
                    break
        if rots:
            ra = np.array(rots)
            means = ra.mean(axis=0)[:3]
            print(f"{m:<25} | {means[0]:>8.4f} {means[1]:>8.4f} {means[2]:>8.4f}")

@app.local_entrypoint()
def main():
    dump_values.remote()
