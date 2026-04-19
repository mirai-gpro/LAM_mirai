"""Analyze motion parameters for all 12 motions in the Volume."""
import modal
import os
import json

app = modal.App("lam-motion-analysis-v3")
volume = modal.Volume.from_name("lam-data", create_if_missing=False)

@app.function(volumes={"/vol": volume}, timeout=120)
def analyze():
    base = "/vol/assets/sample_motion/export"
    motions = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    results = []
    for m in motions:
        fp_dir = os.path.join(base, m, "flame_param")
        if not os.path.isdir(fp_dir):
            results.append((m, 0, "?", "?", "?", "?"))
            continue

        files = sorted(os.listdir(fp_dir))
        frames = len(files)

        # Read first frame file to get structure
        if frames > 0:
            first_file = os.path.join(fp_dir, files[0])
            ext = os.path.splitext(files[0])[1]

            if ext == '.json':
                with open(first_file) as f:
                    d = json.load(f)
                keys = list(d.keys())
                dims = {k: len(d[k]) if isinstance(d[k], list) else str(type(d[k]).__name__) for k in keys}
            elif ext == '.npz':
                # npz needs numpy, skip detail
                dims = {"format": "npz"}
                keys = ["npz"]
            else:
                # Try reading as json anyway
                try:
                    with open(first_file) as f:
                        d = json.load(f)
                    keys = list(d.keys())
                    dims = {k: len(d[k]) if isinstance(d[k], list) else str(type(d[k]).__name__) for k in keys}
                except:
                    dims = {"format": ext}
                    keys = [ext]

            # Get wav/mp4 sizes
            wav = os.path.join(base, m, m + ".wav")
            mp4 = os.path.join(base, m, m + ".mp4")
            wav_mb = f"{os.path.getsize(wav)/1024/1024:.1f}" if os.path.isfile(wav) else "-"
            mp4_mb = f"{os.path.getsize(mp4)/1024/1024:.1f}" if os.path.isfile(mp4) else "-"

            results.append((m, frames, ext, dims, wav_mb, mp4_mb))

    # Print table
    print(f"\n{'#':<4} {'Motion':<32} {'Frames':>7} {'Ext':>5} {'WAV MB':>7} {'MP4 MB':>7}")
    print("=" * 65)
    for i, (m, frames, ext, dims, wav_mb, mp4_mb) in enumerate(results, 1):
        print(f"{i:<4} {m:<32} {frames:>7} {ext:>5} {wav_mb:>7} {mp4_mb:>7}")

    # Print parameter structure (from first motion's first frame)
    print(f"\n--- Parameter structure (per frame) ---")
    if results:
        m, frames, ext, dims, _, _ = results[0]
        print(f"Source: {m}/flame_param/{ext} files")
        for k, v in dims.items():
            print(f"  {k:<25} dim={v}")

        # Also show a few keys from first frame
        fp_dir = os.path.join(base, results[0][0], "flame_param")
        files = sorted(os.listdir(fp_dir))
        first = os.path.join(fp_dir, files[0])
        try:
            with open(first) as f:
                d = json.load(f)
            print(f"\n--- First frame keys & shapes ---")
            for k, v in d.items():
                if isinstance(v, list):
                    if v and isinstance(v[0], list):
                        print(f"  {k:<25} [{len(v)} x {len(v[0])}]")
                    else:
                        print(f"  {k:<25} [{len(v)}]")
                else:
                    print(f"  {k:<25} {type(v).__name__}: {v}")
        except:
            pass

@app.local_entrypoint()
def main():
    analyze.remote()
