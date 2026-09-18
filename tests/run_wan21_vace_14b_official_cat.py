"""运行官方 Wan2.1-VACE-14B 格斗猫测试数据。

该脚本对应
``examples/wanvideo/model_inference/Wan2.1-VACE-14B.py`` 中的三种 VACE
用法：控制视频、参考图、控制视频与参考图组合。模型从固定本地目录加载，
官方测试数据默认位于 ``data/examples/wan``。

如数据尚未下载，可在仓库根目录先执行：

    python -c "from modelscope import dataset_snapshot_download; dataset_snapshot_download(dataset_id='DiffSynth-Studio/examples_in_diffsynth', local_dir='./', allow_file_pattern=['data/examples/wan/depth_video.mp4', 'data/examples/wan/cat_fightning.jpg'])"

单卡（仍使用官方 USP 初始化方式）：

    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
        tests/run_wan21_vace_14b_official_cat.py

多卡运行全部官方案例：

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 \
        tests/run_wan21_vace_14b_official_cat.py --case all
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/mnt/data/wan-vace/models/Wan2.1-VACE-14B")
DATA_DIR = REPO_ROOT / "data/examples/wan"
OUTPUT_DIR = REPO_ROOT / "outputs/wan21_vace_official_cat"

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="官方 Wan2.1-VACE-14B 格斗猫测试")
    parser.add_argument(
        "--case",
        choices=("video", "reference", "combined", "all"),
        default="all",
        help="测试条件，默认运行官方三种案例。",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{description}不存在：{path}\n"
            "请先按脚本文件头部的 ModelScope 命令下载官方格斗猫测试数据。"
        )
    return path


def build_pipeline() -> WanVideoPipeline:
    model_dir = require_path(MODEL_DIR, "本地模型目录")
    dit_paths = sorted(
        glob.glob(str(model_dir / "diffusion_pytorch_model*.safetensors"))
    )
    if not dit_paths:
        raise FileNotFoundError(f"未找到 DiT 权重：{model_dir}")

    text_path = require_path(
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth", "T5 权重"
    )
    vae_path = require_path(model_dir / "Wan2.1_VAE.pth", "VAE 权重")
    tokenizer_path = require_path(
        model_dir / "google/umt5-xxl", "Tokenizer 目录"
    )

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=True,
        model_configs=[
            ModelConfig(path=dit_paths),
            ModelConfig(path=str(text_path)),
            ModelConfig(path=str(vae_path)),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
        redirect_common_files=False,
    )


def cases(case: str) -> tuple[str, ...]:
    return ("video", "reference", "combined") if case == "all" else (case,)


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps 必须大于 0。")

    depth_path = require_path(DATA_DIR / "depth_video.mp4", "官方控制视频")
    reference_path = require_path(
        DATA_DIR / "cat_fightning.jpg", "官方参考图"
    )
    pipe = build_pipeline()
    rank = dist.get_rank()

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"官方数据目录：{DATA_DIR}")
        print(f"输出目录：{args.output_dir.resolve()}")

    control_video = VideoData(str(depth_path), height=480, width=832)
    control_frames = control_video.raw_data()
    del control_video
    with Image.open(reference_path) as image:
        reference_image = image.convert("RGB").resize((832, 480))

    for case in cases(args.case):
        vace_inputs = {}
        if case in ("video", "combined"):
            vace_inputs["vace_video"] = control_frames
        if case in ("reference", "combined"):
            vace_inputs["vace_reference_image"] = reference_image

        video = pipe(
            prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
            negative_prompt=NEGATIVE_PROMPT,
            seed=args.seed,
            tiled=True,
            num_inference_steps=args.num_inference_steps,
            **vace_inputs,
        )

        if rank == 0:
            output_path = args.output_dir / f"wan21_vace_official_cat_{case}.mp4"
            save_video(video, str(output_path), fps=15, quality=5)
            print(f"已保存：{output_path.resolve()}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
