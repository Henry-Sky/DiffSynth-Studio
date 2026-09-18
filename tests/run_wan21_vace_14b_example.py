"""使用仓库官方测试数据执行 Wan2.1-VACE-14B 多 GPU 推理。

本脚本贴合 ``examples/wanvideo/model_inference/Wan2.1-VACE-14B.py``：
使用同一套 Pipeline、VACE 三种条件组合、推理参数及视频保存方式；多 GPU
部分按照官方 USP 教程通过 ``use_usp=True`` 和 ``torchrun`` 启动。

依赖安装（在仓库根目录执行）：

    pip install -e .
    pip install "xfuser[flash-attn]>=0.4.3"

两张 GPU 运行组合条件测试（默认）：

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        tests/run_wan21_vace_14b_example.py

四张 GPU 运行官方三种 VACE 条件测试：

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        tests/run_wan21_vace_14b_example.py --case all

指定输出目录、推理步数或帧数：

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        tests/run_wan21_vace_14b_example.py \
        --case combined --num-inference-steps 50 --num-frames 17 \
        --output-dir outputs/wan21_vace_example

注意：必须用 ``torchrun`` 启动，包括仅使用一张 GPU 的 USP 测试。只有 rank 0
会写入视频，其他 rank 仅参与统一序列并行计算。
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path("/mnt/data/wan-vace/models/Wan2.1-VACE-14B")
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/wan21_vace_example"

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用官方示例数据执行 Wan2.1-VACE-14B USP 多 GPU 推理。"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"本地模型目录（默认：{DEFAULT_MODEL_DIR}）。",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"官方测试数据目录（默认：{DEFAULT_DATASET_DIR}）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"rank 0 保存生成视频的目录（默认：{DEFAULT_OUTPUT_DIR}）。",
    )
    parser.add_argument(
        "--case",
        choices=("video", "reference", "combined", "all"),
        default="combined",
        help="VACE 条件：控制视频、参考图、二者组合，或依次运行全部。",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=17,
        help="生成帧数，必须满足 4n+1（默认：17，项目基线 clip 长度）。",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在：{path}")
    return path


def load_example(dataset_dir: Path) -> tuple[str, Path, Path]:
    """读取官方 metadata.csv 的首个案例，并解析 VACE 条件文件。"""
    metadata_path = require_path(dataset_dir / "metadata.csv", "元数据文件")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file), None)

    if row is None:
        raise ValueError(f"元数据中没有测试案例：{metadata_path}")

    required_columns = ("prompt", "vace_video", "vace_reference_image")
    missing_columns = [key for key in required_columns if not row.get(key)]
    if missing_columns:
        raise ValueError(
            f"元数据缺少有效字段 {missing_columns}：{metadata_path}"
        )

    control_video_path = require_path(
        dataset_dir / row["vace_video"], "VACE 控制视频"
    )
    reference_image_path = require_path(
        dataset_dir / row["vace_reference_image"], "VACE 参考图"
    )
    return row["prompt"], control_video_path, reference_image_path


def build_pipeline(model_dir: Path) -> WanVideoPipeline:
    """使用已持久化的本地权重构建官方 VACE Pipeline。"""
    model_dir = require_path(model_dir, "模型目录")
    dit_pattern = model_dir / "diffusion_pytorch_model*.safetensors"
    dit_paths = sorted(glob.glob(str(dit_pattern)))
    if not dit_paths:
        raise FileNotFoundError(
            f"未找到 diffusion_pytorch_model*.safetensors：{model_dir}"
        )

    text_encoder_path = require_path(
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
            ModelConfig(path=str(text_encoder_path)),
            ModelConfig(path=str(vae_path)),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
        redirect_common_files=False,
    )


def selected_cases(case: str) -> tuple[str, ...]:
    if case == "all":
        return "video", "reference", "combined"
    return (case,)


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("--height 和 --width 必须是 16 的倍数。")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise ValueError("--num-frames 必须满足 4n+1，例如 17 或 81。")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps 必须大于 0。")

    dataset_dir = require_path(args.dataset_dir, "测试数据目录")
    prompt, control_video_path, reference_image_path = load_example(dataset_dir)

    pipe = build_pipeline(args.model_dir)
    rank = dist.get_rank()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"测试案例：{dataset_dir / 'metadata.csv'}")
        print(f"输出目录：{args.output_dir.resolve()}")

    for case in selected_cases(args.case):
        # 与官方脚本一致：控制视频缩放到 480x832，参考图也缩放到相同尺寸。
        control_video = VideoData(
            str(control_video_path), height=args.height, width=args.width
        )
        control_video.set_length(args.num_frames)
        # VideoData 没有自定义 __iter__；Pipeline 的 ``for image in video``
        # 会绕过 set_length，通过 __getitem__ 继续读到源视频末尾。因此在
        # 传入 VACE 前显式展开受控长度的帧列表，保证条件与 num_frames 对齐。
        control_frames = control_video.raw_data()
        # raw_data 已经复制出所需帧，及时关闭底层 FFmpeg reader，避免进程
        # 退出时 BufferedWriter/FFmpeg 管道产生 BrokenPipeError。
        del control_video
        with Image.open(reference_image_path) as image:
            reference_image = image.convert("RGB").resize(
                (args.width, args.height)
            )

        vace_inputs = {}
        if case in ("video", "combined"):
            vace_inputs["vace_video"] = control_frames
        if case in ("reference", "combined"):
            vace_inputs["vace_reference_image"] = reference_image

        video = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            seed=args.seed,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            **vace_inputs,
        )

        if rank == 0:
            output_path = args.output_dir / f"wan21_vace_{case}.mp4"
            save_video(video, str(output_path), fps=args.fps, quality=5)
            print(f"已保存：{output_path.resolve()}")

    # USP 初始化了 NCCL process group；所有 rank 完成保存后显式清理，避免
    # torch.distributed 在正常退出时打印 destroy_process_group 警告。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
