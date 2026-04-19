"""Quick analysis of flame_params.json for all 12 motions in the Volume."""
import modal
import os
import json

app = modal.App("lam-motion-analysis")
volume = modal.Volume.from_name("lam-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy")

@app.function(volumes={"/vol": volume}, timeout=120, image=image)
def analyze_motions():
    import numpy as np

    base = "/vol/assets/sample_motion/export"
    motions = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    print(f"{'Motion':<30} {'Frames':>7} {'Expr dim':>9} {'Rot dim':>8} {'Trans dim':>10} {'Expr range':>20} {'Rot range':>20} {'Trans range':>20}")
    print("-" * 130)

    for m in motions:
        fp = os.path.join(base, m, "flame_params.json")
        if not os.path.isfile(fp):
            # Try flame_param directory
            fp_dir = os.path.join(base, m, "flame_param")
            if os.path.isdir(fp_dir):
                # Count json files in flame_param/
                frames = len([f for f in os.listdir(fp_dir) if f.endswith('.json')])
                # Read first frame to get dimensions
                first = sorted([f for f in os.listdir(fp_dir) if f.endswith('.json')])[0]
                with open(os.path.join(fp_dir, first)) as f:
                    data = json.load(f)
                expr_dim = len(data.get("expression", data.get("expr", [])))
                rot_dim = len(data.get("rotation", data.get("pose", [])))
                trans_dim = len(data.get("translation", []))
                print(f"{m:<30} {frames:>7} {expr_dim:>9} {rot_dim:>8} {trans_dim:>10} {'(per-frame)':>20} {'(per-frame)':>20} {'(per-frame)':>20}")
            else:
                print(f"{m:<30} {'N/A':>7}")
            continue

        with open(fp) as f:
            data = json.load(f)

        # flame_params.json can have different structures
        if isinstance(data, list):
            frames = len(data)
            if frames > 0:
                sample = data[0]
                expr = sample.get("expression", sample.get("expr", []))
                rot = sample.get("rotation", sample.get("pose", []))
                trans = sample.get("translation", [])
        elif isinstance(data, dict):
            # Could be {expression: [[...], [...]], rotation: [[...]], ...}
            expr_key = "expression" if "expression" in data else "expr"
            rot_key = "rotation" if "rotation" in data else "pose"
            trans_key = "translation"

            expr_arr = np.array(data.get(expr_key, []))
            rot_arr = np.array(data.get(rot_key, []))
            trans_arr = np.array(data.get(trans_key, []))

            if expr_arr.ndim == 2:
                frames = expr_arr.shape[0]
                expr_dim = expr_arr.shape[1]
                rot_dim = rot_arr.shape[1] if rot_arr.ndim == 2 else 0
                trans_dim = trans_arr.shape[1] if trans_arr.ndim == 2 else 0
                expr_range = f"[{expr_arr.min():.3f}, {expr_arr.max():.3f}]"
                rot_range = f"[{rot_arr.min():.3f}, {rot_arr.max():.3f}]" if rot_arr.size else "N/A"
                trans_range = f"[{trans_arr.min():.4f}, {trans_arr.max():.4f}]" if trans_arr.size else "N/A"
                print(f"{m:<30} {frames:>7} {expr_dim:>9} {rot_dim:>8} {trans_dim:>10} {expr_range:>20} {rot_range:>20} {trans_range:>20}")
                continue
            else:
                frames = "?"
                expr = data.get(expr_key, [])
                rot = data.get(rot_key, [])
                trans = data.get(trans_key, [])

        print(f"{m:<30} {frames:>7}")

    # Also check canonical params
    print("\n--- Canonical FLAME params ---")
    for m in motions:
        cp = os.path.join(base, m, "canonical_flame_param.npz")
        if os.path.isfile(cp):
            npz = np.load(cp)
            keys = list(npz.keys())
            shapes = {k: npz[k].shape for k in keys}
            print(f"{m:<30} {shapes}")

@app.local_entrypoint()
def main():
    analyze_motions.remote()
