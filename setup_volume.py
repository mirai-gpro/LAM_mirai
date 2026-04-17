"""
LAM Modal Volume Setup
======================
Downloads all model data from HuggingFace into a persistent Modal Volume.
Run this ONCE before deploying modal_app.py.

Usage:
    modal run setup_volume.py

Volume structure after setup:
    /vol/
    ├── pretrained_models/           (~907MB) FLAME tracking models
    │   ├── 68_keypoints_model.pkl
    │   ├── FaceBoxesV2.pth
    │   ├── human_model_files/       (FLAME mesh, from LAM-assets)
    │   ├── matting/
    │   └── vgghead/
    ├── exps/releases/lam/lam-20k/step_045500/  (~2.36GB) LAM model weights
    │   ├── config.json
    │   ├── model.safetensors
    │   └── README.md
    └── assets/                      (~264MB) sample data
        ├── sample_input/
        └── sample_motion/
"""

import modal
import os

VOLUME_NAME = "lam-data"

app = modal.App("lam-setup")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

setup_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "wget")
    .pip_install("huggingface_hub==0.23.2")
)


@app.function(
    image=setup_image,
    volumes={"/vol": volume},
    timeout=7200,
)
def download_all():
    """Download all LAM model data into the volume."""
    import subprocess
    import shutil
    from huggingface_hub import hf_hub_download

    VOL = "/vol"
    TMP = "/tmp/lam_dl"
    os.makedirs(TMP, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 1. FLAME Tracking pretrained models
    #    Source: yuandong513/flametracking_model (HuggingFace)
    #    Contains: 68_keypoints_model.pkl, FaceBoxesV2.pth, matting/, vgghead/
    # ──────────────────────────────────────────────────────────
    pretrained_dir = os.path.join(VOL, "pretrained_models")
    marker = os.path.join(pretrained_dir, "vgghead", "vgg_heads_l.trcd")

    if not os.path.exists(marker):
        print("=" * 60)
        print("[1/4] Downloading FLAME tracking models...")
        print("=" * 60)
        tar_path = hf_hub_download(
            repo_id="yuandong513/flametracking_model",
            repo_type="model",
            filename="pretrain_model.tar",
            local_dir=TMP,
        )
        # Extract: creates pretrain_model/ (no 'd')
        subprocess.run(["tar", "-xf", tar_path, "-C", TMP], check=True)
        # Rename to pretrained_models/ (with 'd') to match ModelScope app.py
        src = os.path.join(TMP, "pretrain_model")
        if os.path.exists(pretrained_dir):
            shutil.rmtree(pretrained_dir)
        shutil.copytree(src, pretrained_dir)
        os.remove(tar_path)
        shutil.rmtree(src)
        print(f"  -> {pretrained_dir}")
    else:
        print("[1/4] FLAME tracking models already exist, skipping.")

    # ──────────────────────────────────────────────────────────
    # 2. FLAME human model files (mesh templates)
    #    Source: 3DAIGC/LAM-assets (HuggingFace)
    #    Contains: flame_assets/, flame_vhap/
    # ──────────────────────────────────────────────────────────
    human_model_dir = os.path.join(pretrained_dir, "human_model_files")
    marker2 = os.path.join(human_model_dir, "flame_assets", "flame", "flame2023.pkl")

    if not os.path.exists(marker2):
        print("=" * 60)
        print("[2/4] Downloading FLAME human model files...")
        print("=" * 60)
        tar_path = hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="LAM_human_model.tar",
            local_dir=TMP,
        )
        # Extract: creates pretrained_models/human_model_files/
        subprocess.run(["tar", "-xf", tar_path, "-C", VOL], check=True)
        os.remove(tar_path)
        print(f"  -> {human_model_dir}")
    else:
        print("[2/4] FLAME human model files already exist, skipping.")

    # ──────────────────────────────────────────────────────────
    # 3. LAM-20K model weights
    #    Source: 3DAIGC/LAM-20K (HuggingFace)
    #    Contains: config.json, model.safetensors (~2.36GB), README.md
    # ──────────────────────────────────────────────────────────
    model_dir = os.path.join(VOL, "exps", "releases", "lam", "lam-20k", "step_045500")
    safetensors_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(safetensors_path) or os.path.getsize(safetensors_path) < 1_000_000:
        print("=" * 60)
        print("[3/4] Downloading LAM-20K model weights (~2.36GB)...")
        print("=" * 60)
        os.makedirs(model_dir, exist_ok=True)
        for fname in ["config.json", "model.safetensors", "README.md"]:
            print(f"  Downloading {fname}...")
            hf_hub_download(
                repo_id="3DAIGC/LAM-20K",
                repo_type="model",
                filename=fname,
                local_dir=model_dir,
            )
        size_gb = os.path.getsize(safetensors_path) / (1024**3)
        print(f"  -> {model_dir} ({size_gb:.2f} GB)")
    else:
        size_gb = os.path.getsize(safetensors_path) / (1024**3)
        print(f"[3/4] Model weights already exist ({size_gb:.2f} GB), skipping.")

    # ──────────────────────────────────────────────────────────
    # 4. Sample assets (motion sequences, input images)
    #    Source: 3DAIGC/LAM-assets (HuggingFace)
    #    Contains: assets/sample_input/, assets/sample_motion/
    # ──────────────────────────────────────────────────────────
    assets_dir = os.path.join(VOL, "assets")
    motion_dir = os.path.join(assets_dir, "sample_motion")

    if not os.path.exists(motion_dir):
        print("=" * 60)
        print("[4/4] Downloading sample assets...")
        print("=" * 60)
        tar_path = hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type="model",
            filename="LAM_assets.tar",
            local_dir=TMP,
        )
        # Extract: creates assets/sample_input/, assets/sample_motion/
        subprocess.run(["tar", "-xf", tar_path, "-C", VOL], check=True)
        os.remove(tar_path)
        print(f"  -> {assets_dir}")
    else:
        print("[4/4] Sample assets already exist, skipping.")

    # ──────────────────────────────────────────────────────────
    # Verify
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = [
        ("pretrained_models/68_keypoints_model.pkl", 200),
        ("pretrained_models/FaceBoxesV2.pth", 3),
        ("pretrained_models/vgghead/vgg_heads_l.trcd", 300),
        ("pretrained_models/matting/stylematte_synth.pt", 100),
        ("pretrained_models/human_model_files/flame_assets/flame/flame2023.pkl", 40),
        ("pretrained_models/human_model_files/flame_vhap/flame2023.pkl", 40),
        ("exps/releases/lam/lam-20k/step_045500/model.safetensors", 2000),
        ("exps/releases/lam/lam-20k/step_045500/config.json", 0),
        ("assets/sample_motion", None),  # directory check
    ]

    all_ok = True
    for path, min_mb in checks:
        full_path = os.path.join(VOL, path)
        if min_mb is None:
            # Directory check
            ok = os.path.isdir(full_path)
            status = "OK (dir)" if ok else "MISSING"
        else:
            exists = os.path.isfile(full_path)
            if exists:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                ok = size_mb >= min_mb
                status = f"OK ({size_mb:.1f} MB)" if ok else f"TOO SMALL ({size_mb:.1f} MB < {min_mb} MB)"
            else:
                ok = False
                status = "MISSING"

        marker_char = "OK" if ok else "NG"
        print(f"  [{marker_char}] {path}: {status}")
        if not ok:
            all_ok = False

    volume.commit()

    if all_ok:
        print("\n=== Volume setup complete! All files verified. ===")
    else:
        print("\n=== WARNING: Some files are missing or too small! ===")
        print("    Re-run this script or check network connectivity.")

    return all_ok


@app.local_entrypoint()
def main():
    result = download_all.remote()
    if result:
        print(f"\nVolume '{VOLUME_NAME}' is ready.")
        print("Next: modal serve modal_app.py")
    else:
        print("\nSetup incomplete. Please re-run.")
