"""Analyze flame_params.json for all motions in the Volume. No numpy needed."""
import modal
import os
import json

app = modal.App("lam-motion-analysis-v2")
volume = modal.Volume.from_name("lam-data", create_if_missing=False)

@app.function(volumes={"/vol": volume}, timeout=120)
def analyze():
    base = "/vol/assets/sample_motion/export"
    motions = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    results = []
    for m in motions:
        fp_dir = os.path.join(base, m, "flame_param")
        fp_json = os.path.join(base, m, "flame_params.json")

        if os.path.isdir(fp_dir):
            files = sorted([f for f in os.listdir(fp_dir) if f.endswith('.json')])
            frames = len(files)
            if files:
                with open(os.path.join(fp_dir, files[0])) as f:
                    d = json.load(f)
                expr = d.get("expression", d.get("expr", []))
                rot = d.get("rotation", d.get("pose", []))
                trans = d.get("translation", [])
                results.append((m, frames, len(expr), len(rot), len(trans)))
            else:
                results.append((m, 0, 0, 0, 0))
        elif os.path.isfile(fp_json):
            with open(fp_json) as f:
                d = json.load(f)
            if isinstance(d, dict):
                expr = d.get("expression", d.get("expr", []))
                if isinstance(expr[0], list):
                    frames = len(expr)
                    expr_dim = len(expr[0])
                else:
                    frames = 1
                    expr_dim = len(expr)
                rot = d.get("rotation", d.get("pose", []))
                rot_dim = len(rot[0]) if isinstance(rot[0], list) else len(rot)
                trans = d.get("translation", [])
                trans_dim = len(trans[0]) if trans and isinstance(trans[0], list) else len(trans)
                results.append((m, frames, expr_dim, rot_dim, trans_dim))
        else:
            results.append((m, "N/A", "?", "?", "?"))

    # Print table
    print(f"\n{'#':<4} {'Motion':<32} {'Frames':>7} {'Expr':>6} {'Rot':>6} {'Trans':>6}")
    print("-" * 65)
    for i, (m, frames, e, r, t) in enumerate(results, 1):
        print(f"{i:<4} {m:<32} {frames:>7} {e:>6} {r:>6} {t:>6}")

    # Also list available files per motion
    print(f"\n--- Files per motion (first one) ---")
    first = motions[0] if motions else None
    if first:
        path = os.path.join(base, first)
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                print(f"  {f:<45} {sz:>12,} bytes")
            else:
                count = len(os.listdir(fp))
                print(f"  {f + '/':<45} {count} files")

@app.local_entrypoint()
def main():
    analyze.remote()
