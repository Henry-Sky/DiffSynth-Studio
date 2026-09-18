"""Wan2.1-VACE-14B 基础推理调用。

默认使用官方小镇日落数据：
``data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B``。

多 GPU 运行示例：

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 \
    tests/run_wan21_vace_14b_example.py

只使用控制视频：

    torchrun --standalone --nproc_per_node=2 \
    tests/run_wan21_vace_14b_example.py --case video

只使用参考图：

    torchrun --standalone --nproc_per_node=2 \
    tests/run_wan21_vace_14b_example.py --case reference

全量测试：

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 \
    tests/run_wan21_vace_14b_example.py \
    --case reference video combined

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
DEFAULT_MODEL_DIR = Path("/mnt/data/wan-vace/models/Wan2.1-VACE-14B")
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B"
)
DEFAULT_OUTPUT_PREFIX = "wan21_vace_offical_town"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / DEFAULT_OUTPUT_PREFIX

DEFAULT_PROMPT = "from sunset to night, a small town, light, house, river"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.1-VACE-14B 基础推理")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--control-video", default="video1_softedge.mp4")
    parser.add_argument("--reference-image", default="reference_image.png")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"输出文件名前缀（默认：{DEFAULT_OUTPUT_PREFIX}）。",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        choices=("video", "reference", "combined"),
        default=["combined"],
        help="一个或多个 VACE 输入类型（默认：combined）。",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--vace-scale", type=float, default=1.0)
    return parser.parse_args()


def required(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在：{path}")
    return path


def build_pipeline(model_dir: Path) -> WanVideoPipeline:
    model_dir = required(model_dir, "模型目录")
    dit_paths = sorted(
        glob.glob(str(model_dir / "diffusion_pytorch_model*.safetensors"))
    )
    if not dit_paths:
        raise FileNotFoundError(f"未找到 DiT 权重：{model_dir}")

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=True,
        model_configs=[
            ModelConfig(path=dit_paths),
            ModelConfig(
                path=str(
                    required(
                        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
                        "T5 权重",
                    )
                )
            ),
            ModelConfig(
                path=str(required(model_dir / "Wan2.1_VAE.pth", "VAE 权重"))
            ),
        ],
        tokenizer_config=ModelConfig(
            path=str(required(model_dir / "google/umt5-xxl", "Tokenizer"))
        ),
        redirect_common_files=False,
    )


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height 和 width 必须是 16 的倍数。")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise ValueError("num_frames 必须满足 4n+1，例如 17 或 81。")

    try:
        data_dir = required(args.data_dir, "数据目录")
        control_path = required(data_dir / args.control_video, "控制视频")
        reference_path = required(data_dir / args.reference_image, "参考图")

        control_video = VideoData(
            str(control_path), height=args.height, width=args.width
        )
        control_video.set_length(args.num_frames)
        control_frames = control_video.raw_data()
        del control_video

        with Image.open(reference_path) as image:
            reference_image = image.convert("RGB").resize(
                (args.width, args.height)
            )

        pipe = build_pipeline(args.model_dir)
        for case in args.case:
            inputs = {}
            if case in ("video", "combined"):
                inputs["vace_video"] = control_frames
            if case in ("reference", "combined"):
                inputs["vace_reference_image"] = reference_image

            video = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                vace_scale=args.vace_scale,
                tiled=True,
                **inputs,
            )

            if dist.get_rank() == 0:
                output_path = args.output_dir / (
                    f"{args.output_prefix}_{case}.mp4"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_video(video, str(output_path), fps=args.fps, quality=5)
                print(f"已保存：{output_path.resolve()}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
