"""
LAM on Modal - Exact ModelScope Reproduction
=============================================

Reproduces the working ModelScope Studio environment on Modal.

Key decisions (informed by previous failure analysis in docs/app_modal.py):
  - model.safetensors SHA256 verified on startup (f527e6e7...) → abort if mismatched
  - All paths exactly match ModelScope app.py (no model_zoo indirection)
  - nvdiffrast installed from /vol/external/nvdiffrast/ source (not wheel)
  - xformers uninstalled (DINOv2 must use PyTorch native attention)
  - torch.compile disabled (sed at build time + TORCHDYNAMO_DISABLE=1)
  - chumpy + numpy 1.23 compatibility patched
  - cpu_nms.pyx numpy deprecation patched (np.int → np.intp)
  - Blender 4.0.2 from Volume (matching ModelScope, not 4.2 from blender.org)
  - Source code cloned from GitHub (branch: claude/setup-lam-modal-Jlv8f)

Prerequisites:
  1. Modal Volume `lam-data` populated with NAS files (see docs/MODAL_SETUP.md)
  2. `modal token new` done

Deploy:
  modal serve modal_app.py    # dev, temporary URL
  modal deploy modal_app.py   # production, persistent URL
"""

import os
import sys
import shutil
import modal

# ==================== Modal Infrastructure ====================

APP_NAME = "lam-mirai"
VOLUME_NAME = "lam-data"
OUTPUT_VOLUME_NAME = "lam-output"
GITHUB_BRANCH = "claude/setup-lam-modal-Jlv8f"
GITHUB_REPO = "https://github.com/mirai-gpro/LAM_mirai.git"

# ModelScope-verified model.safetensors hash (from LFS pointer in Studio repo)
EXPECTED_SHA256 = "f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925"
EXPECTED_SIZE = 2356556212

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

# ==================== OAC Vertex Correction Defaults ====================
# 全部位 disabled = 素の LAM 出力 (FLAME 欧米人バイアスのまま)
# UI/CLI から corrections dict を渡すことで各部位を有効化できる。
DEFAULT_CORRECTIONS = {
    "cheek": {"enabled": False, "y_scale": 1.0,  "margin": 0.002},
    "nose":  {"enabled": False, "x_scale": 1.0,  "z_scale": 1.0, "margin": 0.001},
    "jaw":   {"enabled": False, "y_scale": 1.0,  "margin": 0.002},
    "eye":   {"enabled": False, "x_scale": 1.0,  "margin": 0.002},
}

# ==================== Image Definition ====================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "wget", "ffmpeg", "tree", "xz-utils", "zip",
        "build-essential", "gcc", "g++", "ninja-build",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libusb-1.0-0",
        "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libxkbcommon0",
    )
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        # Broad arch list: T4=7.5, A10G=8.6, L4=8.9
        "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9",
        "CC": "gcc", "CXX": "g++",
        "CXXFLAGS": "-std=c++17",
        "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
        "TORCHDYNAMO_DISABLE": "1",
        "NUMBA_THREADING_LAYER": "forseq",
    })
    # Pin numpy early so downstream packages respect it
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy==1.23.0'",
    )
    # PyTorch 2.4.0 + CUDA 12.1 (matching ModelScope requirements.txt)
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    # chumpy + numpy 1.23 compat fix (from previous solved list)
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "CHUMPY_INIT=$(python -c \"import importlib.util; print(importlib.util.find_spec('chumpy').origin)\") && "
        "sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/"
        "from numpy import nan, inf; import numpy; bool = numpy.bool_; int = numpy.int_; "
        "float = numpy.float64; complex = numpy.complex128; object = numpy.object_; "
        "unicode = numpy.str_; str = numpy.str_/' "
        "\"$CHUMPY_INIT\" && "
        "find $(dirname \"$CHUMPY_INIT\") -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true",
    )
    # Base requirements - EXACT versions from ModelScope requirements.txt
    .pip_install(
        "einops", "roma", "accelerate", "smplx", "iopath", "wheel",
        # face-detection-tflite removed: pulls tensorflow 1.2GB+, only
        # used for iris detection (detect_iris_landmarks=False in our config)
        "moviepy==1.0.3",
        "decord==0.6.0",
        "diffusers",
        "dna==0.0.1",
        "gfpgan==1.3.8",
        "gsplat==1.4.0",
        "huggingface_hub==0.23.2",
        "imageio==2.19.3",
        "jaxtyping==0.2.38",
        "kiui==0.2.14",
        "kornia==0.7.2",
        "loguru==0.7.3",
        "lpips==0.1.4",
        "matplotlib==3.5.3",
        "megfile==4.1.0.post2",
        "omegaconf==2.3.0",
        "open3d==0.19.0",
        "opencv-python",
        "opencv-python-headless",
        "Pillow>=10.0.0,<11.0",
        "plyfile",
        "pygltflib==1.16.2",
        "pyrender==0.1.45",
        "PyYAML==6.0.1",
        "rembg==2.0.63",
        "Requests==2.32.3",
        "scipy<1.12",  # simps removed in 1.12; needed by external/landmark_detection
        "setuptools==74.0.0",
        "taming-transformers-rom1504==0.0.6",
        "timm==1.0.15",
        "pymcubes==0.1.6",
        "tqdm==4.66.4",
        "transformers==4.41.2",
        "trimesh==4.4.9",
        "typeguard",
        "xatlas==0.0.9",
        "imageio-ffmpeg",
        "tyro==0.9.17",
        "pandas==2.2.3",
        "pydantic==2.8.0",
        "Cython",
        "patool",
        "safetensors",
        "tensorboard",
        # Gradio + compat pins (starlette<0.41 keeps old TemplateResponse API
        # that gradio 4.44 depends on; 0.41+ breaks it)
        "gradio==4.44.1", "gradio-client==1.3.0",
        "fastapi==0.115.6", "starlette==0.40.0",
    )
    # Clone LAM_mirai source code from GitHub
    # GIT_LFS_SKIP_SMUDGE=1: skip LFS download (model.safetensors LFS pointer
    # would 404; the real file comes from the Volume via symlink anyway)
    # force_build=True ensures we always get the latest commit on the branch
    # (so newly-added sample images are picked up).
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone -b {GITHUB_BRANCH} --depth 1 {GITHUB_REPO} /app",
        force_build=True,
    )
    # Disable @torch.compile decorators (sed on cloned repo)
    .run_commands(
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/app/lam/models/modeling_lam.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/app/lam/models/encoders/dinov2_fusion_wrapper.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/app/lam/losses/tvloss.py",
        "sed -i 's/^    @torch.compile$/    # @torch.compile  # DISABLED/' "
        "/app/lam/losses/pixelwise.py",
    )
    # Pre-cache DINOv2 weights
    .run_commands(
        "python -c \""
        "import torch; "
        "url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'; "
        "torch.hub.load_state_dict_from_url(url, map_location='cpu'); "
        "print('[DINOv2] cached OK')\"",
    )
    # Final numpy re-pin in case any pip install bumped it
    .run_commands(
        # Uninstall tensorflow (pulled by gfpgan→basicsr→tb-nightly chain).
        # tensorflow 2.21 requires numpy>=1.26 but we pin 1.23.
        # Keep tensorboard (torch.utils.tensorboard needs it) but remove tb-nightly
        # (which tries to load tensorflow).
        "pip uninstall -y tensorflow tensorflow-io-gcs-filesystem tb-nightly 2>/dev/null; true",
        # Install plain tensorboard (works without tensorflow)
        "pip install 'tensorboard<2.16'",
        "pip install 'numpy==1.23.0' --force-reinstall",
    )
)


# ==================== Build-time: bake wheels + external into image ====================
# Uses Image.run_function with volumes= to access the pre-uploaded Volume during build.
# This moves slow/hanging work (nvdiffrast CUDA compile, NMS Cython build) from
# per-container cold start into image build (cached across all containers).

def _bake_wheels_and_external():
    """Install wheels + build NMS + install nvdiffrast (source) during image build."""
    import subprocess
    import shutil
    import os

    # Copy external/ from Volume into image permanent location
    if os.path.exists("/opt/lam_external"):
        shutil.rmtree("/opt/lam_external")
    shutil.copytree("/vol/external", "/opt/lam_external")
    print("[BUILD] copied external -> /opt/lam_external")

    # Install wheels (same order as ModelScope app.py)
    wheels_in_order = [
        "diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
        "simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl",
    ]
    for whl in wheels_in_order:
        path = f"/vol/wheels/{whl}"
        subprocess.run(
            ["pip", "install", "--force-reinstall", "--no-deps", path], check=True
        )
        print(f"[BUILD][WHEEL] {whl}")

    # nvdiffrast from SOURCE (matches ModelScope app.py: pip install ./external/nvdiffrast/)
    subprocess.run(
        ["pip", "install", "/opt/lam_external/nvdiffrast/"], check=True
    )
    print("[BUILD][NVDIFFRAST] installed from source")

    for whl in [
        "pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl",
        "fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl",
    ]:
        path = f"/vol/wheels/{whl}"
        subprocess.run(
            ["pip", "install", "--force-reinstall", path], check=True
        )
        print(f"[BUILD][WHEEL] {whl}")

    # Uninstall xformers (ModelScope app.py does this)
    subprocess.run(["pip", "uninstall", "-y", "xformers"], check=False)
    # Re-pin numpy (wheels bumped it)
    subprocess.run(["pip", "install", "numpy==1.23.0"], check=True)

    # Build NMS Cython extension now (needs external/ available)
    nms_dir = "/opt/lam_external/landmark_detection/FaceBoxesV2/utils/nms"
    subprocess.run(
        ["sed", "-i", "s/dtype=np\\.int)/dtype=np.intp)/", f"{nms_dir}/cpu_nms.pyx"],
        check=False,
    )
    subprocess.run(
        [
            "python", "-c",
            "from setuptools import setup, Extension; "
            "from Cython.Build import cythonize; import numpy; "
            "setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), "
            "include_dirs=[numpy.get_include()])",
            "build_ext", "--inplace",
        ],
        cwd=nms_dir, check=True,
    )
    print("[BUILD][NMS] Cython extension built")

    # Copy built external/ into /app so runtime imports work
    # (app was cloned from git; external is gitignored)
    if os.path.exists("/app/external"):
        shutil.rmtree("/app/external")
    shutil.copytree("/opt/lam_external", "/app/external")
    print("[BUILD] external copied into /app/external (with NMS .so)")


image = image.run_function(
    _bake_wheels_and_external,
    volumes={"/vol": volume},
)


# ==================== Helper Functions (run inside container) ====================

def _verify_model_safetensors():
    """Verify model.safetensors SHA256 and size match ModelScope ground truth."""
    import hashlib

    path = "/vol/exps/releases/lam/lam-20k/step_045500/model.safetensors"
    if not os.path.isfile(path):
        raise RuntimeError(
            f"model.safetensors not found at {path}.\n"
            "Upload it from NAS using docs/MODAL_SETUP.md Step 3-1."
        )

    size = os.path.getsize(path)
    print(f"[VERIFY] {path} ({size:,} bytes)")
    if size != EXPECTED_SIZE:
        raise RuntimeError(
            f"Size mismatch: got {size:,}, expected {EXPECTED_SIZE:,}.\n"
            "The file is corrupted or wrong. Re-upload from NAS."
        )

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA256 mismatch!\n"
            f"  Got:      {got}\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            "The model file is not the ModelScope ground truth. "
            "DO NOT PROCEED (this is what caused 'bird monster' before)."
        )
    print(f"[VERIFY OK] SHA256 = {got}")


def _setup_symlinks_runtime():
    """Symlink Volume paths onto /app (only data, external/wheels are baked)."""
    mappings = [
        ("/vol/exps", "/app/exps"),
        ("/vol/pretrained_models", "/app/pretrained_models"),
        ("/vol/assets/sample_motion", "/app/assets/sample_motion"),
        ("/vol/blender-4.0.2-linux-x64.tar.xz", "/app/blender-4.0.2-linux-x64.tar.xz"),
    ]
    for src, dst in mappings:
        if not os.path.exists(src):
            raise RuntimeError(
                f"Required Volume path missing: {src}\n"
                "Run 'modal volume put lam-data ...' (see docs/MODAL_SETUP.md)"
            )
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.isfile(dst):
            os.remove(dst)
        os.makedirs(os.path.dirname(dst) or "/", exist_ok=True)
        os.symlink(src, dst)
        print(f"[SYMLINK] {dst} -> {src}")


def _install_wheels_and_nvdiffrast():
    """Install custom wheels and nvdiffrast from /app/external/ source."""
    import subprocess

    # Install in EXACT order from ModelScope app.py (lines 26-39)
    wheels_in_order = [
        "diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
        "simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl",
    ]
    for whl in wheels_in_order:
        path = f"/app/wheels/{whl}"
        if not os.path.isfile(path):
            raise RuntimeError(f"Wheel missing: {path}")
        subprocess.run(
            ["pip", "install", "--force-reinstall", "--no-deps", path],
            check=True,
        )
        print(f"[WHEEL] {whl}")

    # nvdiffrast from SOURCE (NOT wheel - this is the ModelScope spec!)
    subprocess.run(["pip", "install", "/app/external/nvdiffrast/"], check=True)
    print("[NVDIFFRAST] installed from /app/external/nvdiffrast/ (source)")

    # Then pytorch3d + fbx wheels
    for whl in [
        "pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl",
        "fbx-2020.3.4-cp310-cp310-manylinux1_x86_64.whl",
    ]:
        path = f"/app/wheels/{whl}"
        subprocess.run(
            ["pip", "install", "--force-reinstall", path],
            check=True,
        )
        print(f"[WHEEL] {whl}")

    # Uninstall xformers (matches ModelScope app.py line 22)
    subprocess.run(["pip", "uninstall", "-y", "xformers"], check=False)
    print("[XFORMERS] uninstalled")

    # Re-pin numpy (wheels may have bumped it)
    subprocess.run(["pip", "install", "numpy==1.23.0"], check=True)


def _build_nms_cython():
    """Build cpu_nms Cython extension from /app/external/."""
    import subprocess

    nms_dir = "/app/external/landmark_detection/FaceBoxesV2/utils/nms"
    pyx_path = f"{nms_dir}/cpu_nms.pyx"

    # Apply numpy deprecation fix (np.int → np.intp)
    subprocess.run(
        ["sed", "-i", "s/dtype=np\\.int)/dtype=np.intp)/", pyx_path],
        check=False,
    )
    # Build inplace
    subprocess.run(
        [
            "python", "-c",
            "from setuptools import setup, Extension; "
            "from Cython.Build import cythonize; "
            "import numpy; "
            "setup(ext_modules=cythonize([Extension('cpu_nms', ['cpu_nms.pyx'])]), "
            "include_dirs=[numpy.get_include()])",
            "build_ext", "--inplace",
        ],
        cwd=nms_dir,
        check=True,
    )
    print("[NMS] Cython extension built")


def _extract_blender():
    """Extract Blender 4.0.2 tarball from Volume."""
    import subprocess

    if os.path.exists("/app/blender-4.0.2-linux-x64/blender"):
        return
    tar_path = "/app/blender-4.0.2-linux-x64.tar.xz"
    subprocess.run(["tar", "-xf", tar_path, "-C", "/app/"], check=True)
    os.chmod("/app/blender-4.0.2-linux-x64/blender", 0o755)
    print("[BLENDER] extracted to /app/blender-4.0.2-linux-x64/")


# ==================== Core Pipeline Functions (from ModelScope app.py) ====================

def _parse_configs():
    """Mirrors app.py parse_configs() (lines 248-306)."""
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument(
        "--blender_path", type=str,
        default="./blender-4.0.2-linux-x64/blender",
    )
    args, unknown = parser.parse_known_args([])  # no CLI args in Modal

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg.blender_path = args.blender_path
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")
    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except Exception:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    assert cfg.model_name is not None, "model_name is required"
    return cfg, cfg_train


def _build_model(cfg):
    """Mirrors app.py _build_model() (lines 641-648)."""
    from lam.models import model_dict
    from lam.utils.hf_hub import wrap_model_hub

    hf_model_cls = wrap_model_hub(model_dict["lam"])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    return model


def _save_imgs_2_video(imgs, v_pth, fps=30):
    """Mirrors app.py save_imgs_2_video() (lines 207-222)."""
    import numpy as np
    from moviepy.editor import ImageSequenceClip

    images = [img.astype(np.uint8) for img in imgs]
    clip = ImageSequenceClip(images, fps=fps)
    clip = clip.subclip(0, len(images) / fps)
    clip.write_videofile(v_pth, codec="libx264")
    print(f"[VIDEO] saved: {v_pth}")


def _add_audio_to_video(video_path, out_path, audio_path, fps=30):
    """Mirrors app.py add_audio_to_video() (lines 225-245)."""
    from moviepy.editor import VideoFileClip, AudioFileClip

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    if audio_clip.duration > 10:
        audio_clip = audio_clip.subclip(0, 10)
    merged = video_clip.set_audio(audio_clip)
    merged.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=fps)
    print(f"[AUDIO] merged: {out_path}")


# ==================== Generator (GPU worker) ====================

@app.cls(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume, "/vol_out": output_vol},
    timeout=900,
    scaledown_window=600,
    enable_memory_snapshot=False,
)
class Generator:
    """GPU worker that loads LAM and runs inference.

    Cold start: symlinks → wheels install → NMS build → blender extract →
                model hash verify → LAM load → FlameTracking load.
    Total: ~3-4 minutes first run; warm calls are <5s setup + inference.
    """

    @modal.enter()
    def setup(self):
        print("=" * 70)
        print("LAM-Modal Container Setup (runtime)")
        print("=" * 70)

        # 1. Symlink Volume → /app for model weights and sample motion
        #    (wheels/external already baked into image at build time)
        _setup_symlinks_runtime()

        # 2. Verify model.safetensors SHA256 (abort if wrong)
        _verify_model_safetensors()

        # 3. Extract Blender if not already (from Volume)
        _extract_blender()

        # 4. Set env vars (matching ModelScope app.py launch_gradio_app())
        os.chdir("/app")
        sys.path.insert(0, "/app")
        os.environ.update({
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": "./exps/releases/lam/lam-20k/step_045500/",
            "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
            "APP_TYPE": "infer.lam",
            "NUMBA_THREADING_LAYER": "forseq",
        })

        # 5. Load LAM model
        print("[LAM] Building model from checkpoint...")
        self.cfg, _ = _parse_configs()
        self.lam = _build_model(self.cfg)
        self.lam.to("cuda")
        print("[LAM] loaded to CUDA")

        # 6. Load FlameTracking (paths match ModelScope app.py lines 666-671)
        from flame_tracking_single_image import FlameTrackingSingleImage
        self.flametracking = FlameTrackingSingleImage(
            output_dir="tracking_output",
            alignment_model_path="./pretrained_models/68_keypoints_model.pkl",
            vgghead_model_path="./pretrained_models/vgghead/vgg_heads_l.trcd",
            human_matting_path="./pretrained_models/matting/stylematte_synth.pt",
            facebox_model_path="./pretrained_models/FaceBoxesV2.pth",
            detect_iris_landmarks=False,
        )
        print("[FlameTracking] loaded")
        print("=" * 70)
        print("READY")
        print("=" * 70)

    @modal.method()
    def generate(self, image_bytes: bytes, motion_name: str,
                 enable_oac_file: bool, corrections: dict = None):
        """Run LAM inference. Mirrors app.py core_fn() (lines 311-471).

        Args:
            corrections: OAC 頂点補正設定。None なら DEFAULT_CORRECTIONS (全 disabled)。
                各部位の dict が `enabled=False` なら補正スキップ。
        """
        # Scaffolding: コピーしたうえで欠損キーを DEFAULT で埋める。
        # 適用ロジックは後続コミットで OAC ブロックに追加する。
        _corr = {k: dict(v) for k, v in DEFAULT_CORRECTIONS.items()}
        if corrections:
            for k, v in corrections.items():
                if k in _corr and isinstance(v, dict):
                    _corr[k].update(v)
        corrections = _corr
        from datetime import datetime
        from pathlib import Path

        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

        os.chdir("/app")

        working_dir = "/tmp/work"
        # Fresh working dir per call
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        image_raw = os.path.join(working_dir, "raw.png")
        with open(image_raw, "wb") as f:
            f.write(image_bytes)

        base_vid = motion_name
        flame_params_dir = os.path.join(
            "./assets/sample_motion/export", base_vid, "flame_param"
        )
        base_iid = "chatting_avatar_" + datetime.now().strftime("%Y%m%d%H%M%S")

        # 補正パラメータをファイル名に埋め込む (再現性担保)。
        # 例: _c0.88_nx1.08_nz0.85_j0.85_e0.90
        # 部位が disabled / scale=1.0 のものはサフィックスに含めない。
        def _fmt_corr_suffix(_c: dict) -> str:
            parts = []
            cc = _c.get("cheek", {})
            if cc.get("enabled") and cc.get("y_scale", 1.0) != 1.0:
                parts.append(f"c{cc['y_scale']:.2f}")
            nc = _c.get("nose", {})
            if nc.get("enabled"):
                if nc.get("x_scale", 1.0) != 1.0:
                    parts.append(f"nx{nc['x_scale']:.2f}")
                if nc.get("z_scale", 1.0) != 1.0:
                    parts.append(f"nz{nc['z_scale']:.2f}")
            jc = _c.get("jaw", {})
            if jc.get("enabled") and jc.get("y_scale", 1.0) != 1.0:
                parts.append(f"j{jc['y_scale']:.2f}")
            ec = _c.get("eye", {})
            if ec.get("enabled") and ec.get("x_scale", 1.0) != 1.0:
                parts.append(f"e{ec['x_scale']:.2f}")
            return ("_" + "_".join(parts)) if parts else ""

        _corr_suffix = _fmt_corr_suffix(corrections)
        base_iid = base_iid + _corr_suffix

        dump_video_path = os.path.join(working_dir, "output.mp4")
        dump_image_path = os.path.join(working_dir, "output.png")
        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)
        dump_tmp_dir = dump_image_dir
        motion_seqs_dir = flame_params_dir

        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)
        vis_motion = self.cfg.get("vis_motion", False)

        # --- FLAME tracking: preprocess → optimize → export ---
        rc = self.flametracking.preprocess(image_raw)
        assert rc == 0, "flametracking preprocess failed!"
        rc = self.flametracking.optimize()
        assert rc == 0, "flametracking optimize failed!"
        rc, output_dir = self.flametracking.export()
        assert rc == 0, "flametracking export failed!"

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        aspect_standard = 1.0 / 1.0
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = 30

        image_tensor, _, _, shape_param = preprocess_image(
            image_path, mask_path=mask_path, intr=None, pad_ratio=0, bg_color=1.0,
            max_tgt_size=None, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size, multiply=14, need_mask=True,
            get_shape_param=True,
        )

        vis_ref_img = (
            image_tensor[0].permute(1, 2, 0).cpu().detach().numpy() * 255
        ).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(os.path.join(dump_tmp_dir, "output.png"))

        src = image_path.split("/")[-3]
        driven = motion_seqs_dir.split("/")[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
            bg_color=1.0, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],  # keep ModelScope quirk verbatim
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven, max_squen_length=300,
        )

        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32

        print("[INFER] start…")
        with torch.no_grad():
            res = self.lam.infer_single_view(
                image_tensor.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={
                    k: v.to(device) for k, v in motion_seq["flame_params"].items()
                },
            )
        print("[INFER] done")

        # --- OAC ZIP (optional) ---
        output_zip_name = None
        if enable_oac_file:
            try:
                sys.path.insert(0, "/app")
                from generateARKITGLBWithBlender import generate_glb

                oac_dir = os.path.join("/tmp", base_iid)
                os.makedirs(oac_dir, exist_ok=True)

                saved_head = self.lam.renderer.flame_model.save_shaped_mesh(
                    shape_param.unsqueeze(0).cuda(), fd=oac_dir,
                )
                _ply_path = os.path.join(oac_dir, "offset.ply")
                res["cano_gs_lst"][0].save_ply(
                    _ply_path, rgb2sh=False, offset2xyz=True,
                )

                # === Vertex corrections (UI-configurable) ===
                # corrections["cheek"]["enabled"] が True のときのみ頬を補正。
                # mesh (skin.glb) のみ補正 (PLY xyz は offset2xyz でオフセット保存
                # のため bbox 選択が成立せず、視覚効果に寄与しないので触らない)。
                _corr_cheek = corrections["cheek"]
                if _corr_cheek["enabled"]:
                    try:
                        import pickle
                        import trimesh as _trimesh

                        _log = lambda m: open("/vol_out/oac_debug.txt", "a").write(m + "\n")
                        _log(f"[OAC] cheek correction start: y_scale={_corr_cheek['y_scale']}")

                        _masks_path = (
                            "/vol/pretrained_models/human_model_files/"
                            "flame_assets/flame/FLAME_masks.pkl"
                        )
                        with open(_masks_path, "rb") as _mf:
                            _part_masks = pickle.load(_mf, encoding="latin1")

                        _n_orig = 5023
                        _face_idx = np.asarray(_part_masks["face"])
                        _face_idx = _face_idx[_face_idx < _n_orig]
                        _exclude = set()
                        for _r_name in ["nose", "lips", "eye_region", "forehead"]:
                            _r = np.asarray(_part_masks[_r_name])
                            _exclude.update(_r[_r < _n_orig].tolist())
                        _cheek_idx_orig = np.array(
                            [i for i in _face_idx if i not in _exclude],
                            dtype=np.int64,
                        )

                        _mesh = _trimesh.load_mesh(saved_head, process=False)
                        _verts = _mesh.vertices.copy()

                        # Spatial bbox expansion to include subdivided verts
                        _cheek_ref = _verts[_cheek_idx_orig]
                        _ymin, _ymax = _cheek_ref[:, 1].min(), _cheek_ref[:, 1].max()
                        _xmin, _xmax = _cheek_ref[:, 0].min(), _cheek_ref[:, 0].max()
                        _zmin, _zmax = _cheek_ref[:, 2].min(), _cheek_ref[:, 2].max()
                        _m = _corr_cheek["margin"]
                        _sel = np.where(
                            (_verts[:, 0] >= _xmin - _m) & (_verts[:, 0] <= _xmax + _m) &
                            (_verts[:, 1] >= _ymin - _m) & (_verts[:, 1] <= _ymax + _m) &
                            (_verts[:, 2] >= _zmin - _m) & (_verts[:, 2] <= _zmax + _m)
                        )[0]

                        _cy = _verts[_sel, 1].mean()
                        _verts[_sel, 1] = _cy + (_verts[_sel, 1] - _cy) * _corr_cheek["y_scale"]
                        _mesh.vertices = _verts
                        _mesh.export(saved_head)
                        _log(f"[OAC] cheek: orig={len(_cheek_idx_orig)} "
                             f"spatial={len(_sel)}/{len(_verts)} OK")
                    except Exception as _ce:
                        import traceback
                        _err = f"[OAC] cheek ERROR: {_ce}\n{traceback.format_exc()}"
                        with open("/vol_out/oac_error.txt", "w") as _ef:
                            _ef.write(_err)
                        output_vol.commit()
                        raise

                # Nose correction (X width, Z height)
                _corr_nose = corrections["nose"]
                if _corr_nose["enabled"]:
                    try:
                        import pickle
                        import trimesh as _trimesh

                        _log = lambda m: open("/vol_out/oac_debug.txt", "a").write(m + "\n")
                        _log(f"[OAC] nose correction start: "
                             f"x_scale={_corr_nose['x_scale']}, z_scale={_corr_nose['z_scale']}")

                        _masks_path = (
                            "/vol/pretrained_models/human_model_files/"
                            "flame_assets/flame/FLAME_masks.pkl"
                        )
                        with open(_masks_path, "rb") as _mf:
                            _part_masks = pickle.load(_mf, encoding="latin1")
                        _n_orig = 5023
                        _nose_idx = np.asarray(_part_masks["nose"])
                        _nose_idx = _nose_idx[_nose_idx < _n_orig]

                        _mesh = _trimesh.load_mesh(saved_head, process=False)
                        _verts = _mesh.vertices.copy()

                        _nose_ref = _verts[_nose_idx]
                        _nxmin, _nxmax = _nose_ref[:, 0].min(), _nose_ref[:, 0].max()
                        _nymin, _nymax = _nose_ref[:, 1].min(), _nose_ref[:, 1].max()
                        _nzmin, _nzmax = _nose_ref[:, 2].min(), _nose_ref[:, 2].max()
                        _nm = _corr_nose["margin"]
                        _nsel = np.where(
                            (_verts[:, 0] >= _nxmin - _nm) & (_verts[:, 0] <= _nxmax + _nm) &
                            (_verts[:, 1] >= _nymin - _nm) & (_verts[:, 1] <= _nymax + _nm) &
                            (_verts[:, 2] >= _nzmin - _nm) & (_verts[:, 2] <= _nzmax + _nm)
                        )[0]

                        # X 方向 (鼻幅)
                        if _corr_nose["x_scale"] != 1.0:
                            _ncx = _verts[_nsel, 0].mean()
                            _verts[_nsel, 0] = _ncx + (_verts[_nsel, 0] - _ncx) * _corr_nose["x_scale"]
                        # Z 方向 (鼻高さ/突出)
                        if _corr_nose["z_scale"] != 1.0:
                            _ncz = _verts[_nsel, 2].mean()
                            _verts[_nsel, 2] = _ncz + (_verts[_nsel, 2] - _ncz) * _corr_nose["z_scale"]

                        _mesh.vertices = _verts
                        _mesh.export(saved_head)
                        _log(f"[OAC] nose: orig={len(_nose_idx)} "
                             f"spatial={len(_nsel)}/{len(_verts)} OK")
                    except Exception as _ne:
                        import traceback
                        _err = f"[OAC] nose ERROR: {_ne}\n{traceback.format_exc()}"
                        with open("/vol_out/oac_error.txt", "w") as _ef:
                            _ef.write(_err)
                        output_vol.commit()
                        raise

                # Jaw correction: boundary マスクの下半分 (顎ライン) を Y 縮小
                _corr_jaw = corrections["jaw"]
                if _corr_jaw["enabled"]:
                    try:
                        import pickle
                        import trimesh as _trimesh

                        _log = lambda m: open("/vol_out/oac_debug.txt", "a").write(m + "\n")
                        _log(f"[OAC] jaw correction start: y_scale={_corr_jaw['y_scale']}")

                        _masks_path = (
                            "/vol/pretrained_models/human_model_files/"
                            "flame_assets/flame/FLAME_masks.pkl"
                        )
                        with open(_masks_path, "rb") as _mf:
                            _part_masks = pickle.load(_mf, encoding="latin1")
                        _n_orig = 5023
                        _bidx = np.asarray(_part_masks["boundary"])
                        _bidx = _bidx[_bidx < _n_orig]

                        _mesh = _trimesh.load_mesh(saved_head, process=False)
                        _verts = _mesh.vertices.copy()

                        _bref = _verts[_bidx]
                        _bxmin, _bxmax = _bref[:, 0].min(), _bref[:, 0].max()
                        _bymin, _bymax = _bref[:, 1].min(), _bref[:, 1].max()
                        _bzmin, _bzmax = _bref[:, 2].min(), _bref[:, 2].max()
                        _bm = _corr_jaw["margin"]
                        _ball = np.where(
                            (_verts[:, 0] >= _bxmin - _bm) & (_verts[:, 0] <= _bxmax + _bm) &
                            (_verts[:, 1] >= _bymin - _bm) & (_verts[:, 1] <= _bymax + _bm) &
                            (_verts[:, 2] >= _bzmin - _bm) & (_verts[:, 2] <= _bzmax + _bm)
                        )[0]
                        # 顎 = boundary の下半分 (Y が中央値より小さい側)
                        _by_mid = (_bymin + _bymax) / 2
                        _jaw_sel = _ball[_verts[_ball, 1] < _by_mid]
                        if len(_jaw_sel) > 0:
                            _jy = _verts[_jaw_sel, 1].mean()
                            _verts[_jaw_sel, 1] = _jy + (
                                _verts[_jaw_sel, 1] - _jy
                            ) * _corr_jaw["y_scale"]
                            _mesh.vertices = _verts
                            _mesh.export(saved_head)
                        _log(f"[OAC] jaw: boundary={len(_bidx)} "
                             f"lower_half={len(_jaw_sel)}/{len(_verts)} OK")
                    except Exception as _je:
                        import traceback
                        _err = f"[OAC] jaw ERROR: {_je}\n{traceback.format_exc()}"
                        with open("/vol_out/oac_error.txt", "w") as _ef:
                            _ef.write(_err)
                        output_vol.commit()
                        raise

                # Eye correction: eye_region マスクを X 縮小 (目尻を狭める)
                _corr_eye = corrections["eye"]
                if _corr_eye["enabled"]:
                    try:
                        import pickle
                        import trimesh as _trimesh

                        _log = lambda m: open("/vol_out/oac_debug.txt", "a").write(m + "\n")
                        _log(f"[OAC] eye correction start: x_scale={_corr_eye['x_scale']}")

                        _masks_path = (
                            "/vol/pretrained_models/human_model_files/"
                            "flame_assets/flame/FLAME_masks.pkl"
                        )
                        with open(_masks_path, "rb") as _mf:
                            _part_masks = pickle.load(_mf, encoding="latin1")
                        _n_orig = 5023
                        _eidx = np.asarray(_part_masks["eye_region"])
                        _eidx = _eidx[_eidx < _n_orig]

                        _mesh = _trimesh.load_mesh(saved_head, process=False)
                        _verts = _mesh.vertices.copy()

                        _eref = _verts[_eidx]
                        _exmin, _exmax = _eref[:, 0].min(), _eref[:, 0].max()
                        _eymin, _eymax = _eref[:, 1].min(), _eref[:, 1].max()
                        _ezmin, _ezmax = _eref[:, 2].min(), _eref[:, 2].max()
                        _em = _corr_eye["margin"]
                        _esel = np.where(
                            (_verts[:, 0] >= _exmin - _em) & (_verts[:, 0] <= _exmax + _em) &
                            (_verts[:, 1] >= _eymin - _em) & (_verts[:, 1] <= _eymax + _em) &
                            (_verts[:, 2] >= _ezmin - _em) & (_verts[:, 2] <= _ezmax + _em)
                        )[0]
                        # 左右対称に縮小するため中心を 0 固定 (顔の対称軸)
                        _verts[_esel, 0] = _verts[_esel, 0] * _corr_eye["x_scale"]
                        _mesh.vertices = _verts
                        _mesh.export(saved_head)
                        _log(f"[OAC] eye: orig={len(_eidx)} "
                             f"spatial={len(_esel)}/{len(_verts)} OK")
                    except Exception as _ee:
                        import traceback
                        _err = f"[OAC] eye ERROR: {_ee}\n{traceback.format_exc()}"
                        with open("/vol_out/oac_error.txt", "w") as _ef:
                            _ef.write(_err)
                        output_vol.commit()
                        raise
                # === Vertex corrections end ===

                generate_glb(
                    input_mesh=Path(saved_head),
                    template_fbx=Path("./assets/sample_oac/template_file.fbx"),
                    output_glb=Path(os.path.join(oac_dir, "skin.glb")),
                    blender_exec=Path("./blender-4.0.2-linux-x64/blender"),
                )
                shutil.copy(
                    "./assets/sample_oac/animation.glb",
                    os.path.join(oac_dir, "animation.glb"),
                )
                if os.path.exists(saved_head):
                    os.remove(saved_head)

                output_zip_name = base_iid + ".zip"
                zip_out = os.path.join("/vol_out", output_zip_name)
                if os.path.exists(zip_out):
                    os.remove(zip_out)
                os.system(f"cd /tmp && zip -r {zip_out} {base_iid}")
                shutil.rmtree(oac_dir)
                print(f"[OAC] ZIP -> {zip_out}")
            except Exception as e:
                print(f"[OAC] ERROR: {e}")
                output_zip_name = None

        # --- Video + audio ---
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        _save_imgs_2_video(rgb, dump_video_path, render_fps)
        audio_path = os.path.join(
            "./assets/sample_motion/export", base_vid, base_vid + ".wav"
        )
        dump_video_path_wa = os.path.join("/vol_out", base_iid + "_audio.mp4")
        _add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path, render_fps)

        output_vol.commit()
        return os.path.basename(dump_video_path_wa), output_zip_name


# ==================== Web UI (FastAPI + vanilla HTML) ====================
#
# Gradio 4.x は Modal ASGI 上で queue/SSE が不安定だったため FastAPI 直書きに
# 切替。WebSocket/SSE 不使用、同期 POST のみ、@modal.asgi_app() との相性が
# 良く、Modal 公式の推奨デプロイパターン。依存追加なし (fastapi は既存)。

_MOTION_CHOICES = [
    "Speeding_Scandal", "Look_In_My_Eyes", "D_ANgelo_Dinero",
    "Michael_Wayne_Rosen", "I_Am_Iron_Man", "Anti_Drugs",
    "Pen_Pineapple_Apple_Pen", "Taylor_Swift", "GEM",
    "The_Shawshank_Redemption",
]

_INDEX_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>LAM Test Harness</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 960px; margin: 1.5em auto; padding: 0 1em;
         background: #fafafa; color: #222; }
  h1 { margin: 0 0 0.2em; color: #0066cc; }
  .subtitle { color: #666; margin-bottom: 1.5em; }
  form { display: grid; gap: 0.8em; background: white; padding: 1.5em;
         border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,.05); }
  .row { display: grid; grid-template-columns: 180px 1fr 70px;
         gap: 1em; align-items: center; }
  .row label { font-weight: 500; }
  input[type="file"], select { padding: 0.4em; font-size: 1em; }
  input[type="range"] { width: 100%; accent-color: #0066cc; }
  output { font-family: "SF Mono", Consolas, monospace; font-size: 0.95em;
           color: #444; text-align: right; }
  .section { border: 1px solid #ddd; padding: 1em 1.2em;
             border-radius: 8px; margin-top: 0.5em; }
  .section h3 { margin: 0 0 0.6em; font-size: 1em; color: #444; }
  .section .note { color: #888; font-size: 0.85em; margin-bottom: 0.8em; }
  button { padding: 0.8em 2em; background: #0066cc; color: white;
           border: none; border-radius: 6px; cursor: pointer;
           font-size: 1.05em; font-weight: 500; }
  button:hover:not(:disabled) { background: #0055aa; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  #spinner { display: none; margin-top: 0.8em; color: #666; font-size: 0.95em; }
  #spinner.active { display: block; }
  #error { display: none; margin-top: 1em; padding: 0.8em 1em;
           background: #fde; color: #a00; border-left: 4px solid #c00;
           border-radius: 4px; white-space: pre-wrap;
           font-family: "SF Mono", Consolas, monospace; font-size: 0.9em; }
  #result { display: none; margin-top: 1.5em; padding: 1.5em; background: white;
            border: 2px solid #28a745; border-radius: 10px; }
  #result h2 { margin: 0 0 0.8em; color: #28a745; }
  #result video { width: 100%; max-width: 600px; display: block;
                  margin: 1em auto; background: #000; border-radius: 6px; }
  #result a { color: #0066cc; text-decoration: none;
              font-family: "SF Mono", Consolas, monospace; }
  #result a:hover { text-decoration: underline; }
  .params { font-family: "SF Mono", Consolas, monospace; font-size: 0.88em;
            color: #666; padding: 0.5em 0.8em; background: #f4f4f4;
            border-radius: 4px; margin-top: 0.8em; }
</style>
</head>
<body>
<h1>LAM Test Harness</h1>
<div class="subtitle">FLAME 頂点補正 × モーション の組合せテスト (FastAPI)</div>

<form id="form" enctype="multipart/form-data">
  <div class="row">
    <label for="image">入力画像:</label>
    <input type="file" name="image" id="image" accept="image/*" required>
    <span></span>
  </div>
  <div class="row">
    <label for="motion">モーション:</label>
    <select name="motion" id="motion">__MOTION_OPTIONS__</select>
    <span></span>
  </div>
  <div class="row">
    <label><input type="checkbox" name="enable_oac"> OAC ZIP を出力</label>
    <span></span>
    <span></span>
  </div>

  <div class="section">
    <h3>🛠 OAC 頂点補正パラメータ (未有効化時は無補正)</h3>
    <div class="note">1.00 = 補正なし。&lt;1.00 で縮小、&gt;1.00 で拡大。
      大きな値は bbox 境界に段差 (イボ) が出やすい。</div>

    <div class="row">
      <label><input type="checkbox" name="cheek_on"> 頬 (Cheek) Y:</label>
      <input type="range" name="cheek_y" min="0.70" max="1.30" step="0.01" value="1.00"
             oninput="this.nextElementSibling.value=Number(this.value).toFixed(2)">
      <output>1.00</output>
    </div>
    <div class="row">
      <label><input type="checkbox" name="nose_on"> 鼻 (Nose) X (幅):</label>
      <input type="range" name="nose_x" min="0.80" max="1.30" step="0.01" value="1.00"
             oninput="this.nextElementSibling.value=Number(this.value).toFixed(2)">
      <output>1.00</output>
    </div>
    <div class="row">
      <label style="padding-left:1.5em">鼻 Z (高さ):</label>
      <input type="range" name="nose_z" min="0.30" max="1.20" step="0.01" value="1.00"
             oninput="this.nextElementSibling.value=Number(this.value).toFixed(2)">
      <output>1.00</output>
    </div>
    <div class="row">
      <label><input type="checkbox" name="jaw_on"> 顎 (Jaw) Y:</label>
      <input type="range" name="jaw_y" min="0.70" max="1.20" step="0.01" value="1.00"
             oninput="this.nextElementSibling.value=Number(this.value).toFixed(2)">
      <output>1.00</output>
    </div>
    <div class="row">
      <label><input type="checkbox" name="eye_on"> 目 (Eye) X (目尻):</label>
      <input type="range" name="eye_x" min="0.70" max="1.10" step="0.01" value="1.00"
             oninput="this.nextElementSibling.value=Number(this.value).toFixed(2)">
      <output>1.00</output>
    </div>
  </div>

  <button type="submit" id="submit-btn">Generate (Ctrl+Enter)</button>
  <div id="spinner">⏳ 生成中... (Generator cold start 含め 2〜4 分)</div>
</form>

<div id="error"></div>

<div id="result">
  <h2>✓ 生成完了</h2>
  <video id="video" controls autoplay muted></video>
  <p>📦 ZIP: <a id="zip-link" download></a></p>
  <p>🎬 MP4: <a id="mp4-link" download></a></p>
  <div class="params" id="params"></div>
</div>

<script>
const form = document.getElementById('form');
const btn = document.getElementById('submit-btn');
const spinner = document.getElementById('spinner');
const error = document.getElementById('error');
const result = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true;
  spinner.classList.add('active');
  error.style.display = 'none';
  result.style.display = 'none';

  const data = new FormData(form);
  const startedAt = Date.now();

  try {
    const res = await fetch('/generate', { method: 'POST', body: data });
    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const json = await res.json();

    document.getElementById('params').textContent =
      `motion=${json.motion}  corrections=${JSON.stringify(json.corrections)}  elapsed=${elapsed}s`;

    const video = document.getElementById('video');
    const mp4Link = document.getElementById('mp4-link');
    const zipLink = document.getElementById('zip-link');

    if (json.video_name) {
      video.src = `/file/${json.video_name}`;
      mp4Link.href = `/file/${json.video_name}`;
      mp4Link.textContent = json.video_name;
    } else {
      video.style.display = 'none';
    }
    if (json.zip_name) {
      zipLink.href = `/file/${json.zip_name}`;
      zipLink.textContent = json.zip_name;
      zipLink.parentElement.style.display = 'block';
    } else {
      zipLink.parentElement.style.display = 'none';
    }
    result.style.display = 'block';
  } catch (err) {
    error.textContent = `Error: ${err.message}`;
    error.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.classList.remove('active');
  }
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') form.requestSubmit();
});
</script>
</body>
</html>
"""


@app.function(
    image=image,
    volumes={"/vol_out": output_vol},
    timeout=3600,
    scaledown_window=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    """Minimal FastAPI UI: HTML form → POST /generate → Generator.remote()."""
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        opts = "".join(
            f'<option value="{m}"{" selected" if m == "GEM" else ""}>{m}</option>'
            for m in _MOTION_CHOICES
        )
        return _INDEX_HTML.replace("__MOTION_OPTIONS__", opts)

    @web_app.post("/generate")
    async def generate_endpoint(
        image: UploadFile = File(...),
        motion: str = Form("GEM"),
        enable_oac: bool = Form(False),
        cheek_on: bool = Form(False),
        cheek_y: float = Form(1.0),
        nose_on: bool = Form(False),
        nose_x: float = Form(1.0),
        nose_z: float = Form(1.0),
        jaw_on: bool = Form(False),
        jaw_y: float = Form(1.0),
        eye_on: bool = Form(False),
        eye_x: float = Form(1.0),
    ):
        img_bytes = await image.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload")
        corrections = {
            "cheek": {"enabled": cheek_on, "y_scale": cheek_y, "margin": 0.002},
            "nose":  {"enabled": nose_on,  "x_scale": nose_x, "z_scale": nose_z, "margin": 0.001},
            "jaw":   {"enabled": jaw_on,   "y_scale": jaw_y, "margin": 0.002},
            "eye":   {"enabled": eye_on,   "x_scale": eye_x, "margin": 0.002},
        }
        try:
            video_name, zip_name = Generator().generate.remote(
                img_bytes, motion, enable_oac, corrections
            )
        except Exception as e:
            import traceback
            raise HTTPException(
                status_code=500,
                detail=f"Generator error: {e}\n{traceback.format_exc()}",
            )
        output_vol.reload()
        return JSONResponse({
            "video_name": video_name,
            "zip_name": zip_name,
            "motion": motion,
            "corrections": corrections,
        })

    @web_app.get("/file/{filename}")
    async def get_file(filename: str):
        output_vol.reload()
        path = f"/vol_out/{filename}"
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        if filename.endswith(".mp4"):
            return FileResponse(path, media_type="video/mp4")
        if filename.endswith(".zip"):
            return FileResponse(
                path, media_type="application/zip", filename=filename
            )
        return FileResponse(path, filename=filename)

    return web_app


# ==================== CLI Test (bypass UI, test Generator directly) ====================

@app.function(image=image)
def get_sample_image(name: str) -> bytes:
    """Fetch a sample image from /app/assets/sample_input/ (baked from git clone)."""
    import os
    for ext in [".png", ".jpg", ".jpeg"]:
        path = f"/app/assets/sample_input/{name}{ext}"
        if os.path.isfile(path):
            return open(path, "rb").read()
    raise FileNotFoundError(f"Sample '{name}' not found in /app/assets/sample_input/")


@app.local_entrypoint()
def test(image: str = "sample"):
    """Test the Generator pipeline without UI. Logs appear in terminal.

    Usage:
        modal run modal_app.py                    # uses messi.png (from Modal container)
        modal run modal_app.py --image status     # uses status.png
        modal run modal_app.py --image C:\\path\\to\\local.jpg  # local file

    Available samples (from assets/sample_input/ in repo, fetched from Modal container):
        messi, status, james, cluo, dufu, libai, barbara, pop, musk, speed, zhouxingchi
    """
    import os

    print("=== Testing Generator pipeline (no UI) ===")

    # Resolve image: local file path, or sample name (fetched from Modal container)
    if os.path.isfile(image):
        with open(image, "rb") as f:
            img_bytes = f.read()
        print(f"Input: {image} (local file, {len(img_bytes):,} bytes)")
    else:
        print(f"Fetching sample '{image}' from Modal container /app/assets/sample_input/...")
        try:
            img_bytes = get_sample_image.remote(image)
        except Exception as e:
            print(f"ERROR: {e}")
            print("Available: messi, status, james, cluo, dufu, libai, barbara, pop, musk, speed, zhouxingchi")
            return
        print(f"Input: /app/assets/sample_input/{image}.* ({len(img_bytes):,} bytes)")
    print("Calling Generator().generate.remote() with motion='GEM'...")
    print("(First call: Generator cold start ~30-60s + tracking ~30s + inference ~30s)")
    print()

    try:
        video_name, zip_name = Generator().generate.remote(
            img_bytes, "GEM", True
        )
        print(f"\n=== SUCCESS ===")
        print(f"Video: {video_name}")
        print(f"ZIP: {zip_name}")
        print()
        print("Download the result with:")
        print(f"  modal volume get lam-output {video_name} ./output/")
    except Exception as e:
        print(f"\n=== FAILED ===")
        print(f"Error: {e}")
