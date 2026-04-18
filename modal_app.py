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
        "face-detection-tflite==0.6.0",
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
        "Pillow>=10.0.0",
        "plyfile",
        "pygltflib==1.16.2",
        "pyrender==0.1.45",
        "PyYAML==6.0.1",
        "rembg==2.0.63",
        "Requests==2.32.3",
        "scipy",
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
        # Gradio
        "gradio>=5.0.0,<6.0.0", "fastapi",
    )
    # Clone LAM_mirai source code from GitHub
    # GIT_LFS_SKIP_SMUDGE=1: skip LFS download (model.safetensors LFS pointer
    # would 404; the real file comes from the Volume via symlink anyway)
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone -b {GITHUB_BRANCH} --depth 1 {GITHUB_REPO} /app",
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
    .run_commands("pip install 'numpy==1.23.0' --force-reinstall")
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


def _setup_symlinks():
    """Symlink Volume paths onto /app to match ModelScope app.py exactly."""
    mappings = [
        ("/vol/exps", "/app/exps"),
        ("/vol/pretrained_models", "/app/pretrained_models"),
        ("/vol/assets/sample_motion", "/app/assets/sample_motion"),
        ("/vol/external", "/app/external"),
        ("/vol/wheels", "/app/wheels"),
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
        print("LAM-Modal Container Setup")
        print("=" * 70)

        # 1. Symlink Volume → /app (matching ModelScope paths)
        _setup_symlinks()

        # 2. Verify model.safetensors SHA256 (abort if wrong)
        _verify_model_safetensors()

        # 3. Extract Blender
        _extract_blender()

        # 4. Install wheels + nvdiffrast (source)
        _install_wheels_and_nvdiffrast()

        # 5. Build NMS Cython extension
        _build_nms_cython()

        # 6. Set env vars (matching ModelScope app.py launch_gradio_app())
        os.chdir("/app")
        sys.path.insert(0, "/app")
        os.environ.update({
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": "./exps/releases/lam/lam-20k/step_045500/",
            "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
            "APP_TYPE": "infer.lam",
            "NUMBA_THREADING_LAYER": "forseq",
        })

        # 7. Load LAM model
        print("[LAM] Building model from checkpoint...")
        self.cfg, _ = _parse_configs()
        self.lam = _build_model(self.cfg)
        self.lam.to("cuda")
        print("[LAM] loaded to CUDA")

        # 8. Load FlameTracking (paths match ModelScope app.py lines 666-671)
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
    def generate(self, image_bytes: bytes, motion_name: str, enable_oac_file: bool):
        """Run LAM inference. Mirrors app.py core_fn() (lines 311-471)."""
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
                res["cano_gs_lst"][0].save_ply(
                    os.path.join(oac_dir, "offset.ply"),
                    rgb2sh=False, offset2xyz=True,
                )
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


# ==================== Gradio Web UI ====================

@app.function(
    image=image,
    volumes={"/vol_out": output_vol},
    timeout=3600,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    """Gradio UI served as ASGI app (gradio 5.x avoids the 4.44 jinja2 crash)."""
    import gradio as gr

    MOTION_CHOICES = [
        "Speeding_Scandal", "Look_In_My_Eyes", "D_ANgelo_Dinero",
        "Michael_Wayne_Rosen", "I_Am_Iron_Man", "Anti_Drugs",
        "Pen_Pineapple_Apple_Pen", "Taylor_Swift", "GEM",
        "The_Shawshank_Redemption",
    ]

    def predict(image_file, motion_name, enable_oac):
        if image_file is None:
            raise gr.Error("Please upload an image first.")
        with open(image_file, "rb") as f:
            img_bytes = f.read()

        video_name, zip_name = Generator().generate.remote(
            img_bytes, motion_name, enable_oac
        )

        output_vol.reload()

        video_path = f"/vol_out/{video_name}" if video_name else None
        zip_path = f"/vol_out/{zip_name}" if zip_name else None
        return video_path, zip_path

    with gr.Blocks(title="LAM on Modal (ModelScope reproduction)") as demo:
        gr.Markdown(
            "# LAM: Large Avatar Model — Modal Edition\n"
            "Exact reproduction of the ModelScope Studio pipeline. "
            "Drop an image, pick a motion, generate an animated avatar (+ optional OAC ZIP)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    type="filepath", label="Input Image",
                    sources=["upload"], height=480,
                )
                motion_choice = gr.Dropdown(
                    choices=MOTION_CHOICES,
                    value="GEM",
                    label="Motion Template",
                )
                enable_oac = gr.Checkbox(
                    label="Export ZIP for Chatting Avatar (OAC)", value=False,
                )
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                out_video = gr.Video(label="Rendered Video", autoplay=True)
                out_zip = gr.File(label="Output ZIP")

        btn.click(
            predict,
            inputs=[input_img, motion_choice, enable_oac],
            outputs=[out_video, out_zip],
        )

    # Return the gradio Blocks as an ASGI app (gradio 5.x supports this directly)
    return gr.routes.App.create_app(demo)
