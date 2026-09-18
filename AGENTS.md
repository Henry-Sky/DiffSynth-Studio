# 仓库指南

## 项目目标

本仓库用于微调 `Wan2.1-VACE-14B`，修复 3D 重建车辆注册并渲染回驾驶场景后的突兀感、材质/颜色偏差和不自然边缘。输出应尽量接近 `origin_video`，修改集中在车辆及其边界，同时保持车辆几何、姿态、轨迹和背景不变。最终模型作为 asset-forge 动态车辆注入流程的后处理与场景融合模块。

## 固定路径与代码结构

- 仓库：`/workspace/DiffSynth-Studio`
- 模型：`/mnt/data/wan-vace/models/Wan2.1-VACE-14B`
- 数据：`/mnt/data/wan-vace/datasets/vehicle_vace`
- Base 模型官方测试数据：`data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-14B/`（`video1.mp4`、`video1_softedge.mp4`、`reference_image.png` 与 `metadata.csv`）
- VACE 入口：`examples/wanvideo/model_inference/Wan2.1-VACE-14B.py`
- 训练入口：`examples/wanvideo/model_training/train.py`
- LoRA/全参模板：`examples/wanvideo/model_training/{lora,full}/Wan2.1-VACE-14B.sh`
- 核心条件处理：`diffsynth/pipelines/wan_video.py` 中的 `WanVideoUnit_VACE`

Wan2.1-VACE-14B 的测试与推理脚本必须以
`examples/wanvideo/model_inference/Wan2.1-VACE-14B.py` 为官方行为基准：
模型配置、VACE 条件字段（`vace_video`、`vace_video_mask`、
`vace_reference_image`）、提示词/负面提示词、随机种子、分辨率、帧数、
VAE tiling 和视频保存方式应保持一致；新增测试脚本只能在此基础上适配本地
数据集路径、多 GPU 启动和日志输出，并须在注释中说明任何有意差异。

模型必须从 ModelScope 国内源下载并持久化到上述模型目录；训练和推理脚本优先引用本地路径，禁止把权重、数据集或生成视频提交到 Git。

### 官方源码保护

- 默认不得修改官方/上游源码，包括 `diffsynth/` 核心实现、官方 Pipeline、模型实现及 `examples/` 中的官方示例；应在 `tests/` 或其他项目自有脚本中适配和验证。
- 只有用户明确提出修改官方源码时，才允许编辑上述文件；实施前应说明修改文件、原因、影响范围和验证方式。
- 诊断、调试或修复测试脚本时，不得顺手改动官方源码；若确实需要上游修复，必须先向用户确认并保留最小变更。

### GPU 运行环境

默认文件系统沙箱不挂载 NVIDIA 字符设备，即使宿主机已分配 GPU，`nvidia-smi` 和 `torch.cuda.is_available()` 仍会误报不可用。执行 GPU 探测、VACE 推理、训练或 CUDA 验证时，必须请求提升权限（`require_escalated`）后再运行；先用 `nvidia-smi` 与 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` 确认。当前提升权限环境可见单卡 `NVIDIA A800-SXM4-80GB`，默认沙箱中的 CUDA 报错不应归因于项目代码或依赖。

## 数据约定

所有样本必须严格逐帧对齐，并保持相同 FPS、帧数、裁剪和空间尺寸。基线采用 480×832、17 帧 clip。当前训练语义中，`video` 是监督目标，因此初始映射为：

- `video` → `origin_video`（真实 GT）
- `vace_video` → `registered_video`（待修复输入）
- `vace_video_mask` → 车辆区域 mask（白色/1 为响应修改区）
- `prompt` → 简短、稳定的场景与修复任务描述
- `vace_reference_image` → 可选参考图；无明确收益时不要强制使用

`metadata.csv` 至少包含启用字段对应的列，文件路径相对数据根目录；同时令 `--data_file_keys` 和 `--extra_inputs` 包含相同的 VACE 条件字段。`masked_video` 与“完整 registered video + mask”应分别做消融，不得在未验证 mask 极性及 `inactive/reactive` 语义前替换基线。

### 数据集目录与划分

数据根目录是 asset-forge 注册数据集的符号链接。每个 `sample-NNN/` 包含：

- `videos/origin_video.mp4`：原始驾驶场景 GT；
- `videos/registered_video.mp4`：注册并渲染后的车辆场景；
- `videos/mask_video.mp4`：车辆及融合边界的逐帧 mask；
- `registration-manifest.json`：样本 ID、相机、分辨率、FPS、渲染器和逐帧注册信息；
- `log/registration-mask-timing.json`：mask 生成与处理时序日志。

`sample-000` 至 `sample-089` 为保留集：当前仍在处理，未来用于测试，禁止写入训练 `metadata.csv`、训练缓存或训练统计。仅 `sample-090` 至 `sample-299` 可作为训练候选；加入前必须确认三路视频和 manifest 完整、逐帧对齐，并通过长度、FPS、分辨率与 mask 有效性检查。样本编号可能不连续，按实际存在且校验通过的目录生成 metadata，不得按编号补造记录。

## 实施与验证顺序

1. `pip install -e .` 安装环境，并先用原始权重跑通 VACE inference。
2. 用极少量对齐样本做 overfit，确认 registered → origin 映射、mask 极性和数据读取正确。
3. 先训练 `vace` LoRA，再扩大数据；只有 LoRA 上限明确后才考虑全参微调。
4. 固定种子与验证集，并排保存 `registered / VACE output / origin GT / mask`。重点检查车辆区域质量、边缘闪烁、颜色一致性、背景漂移和几何/轨迹保持。

训练前运行数据完整性检查；代码改动后至少执行 `python -m compileall diffsynth` 和一次 17 帧推理。每次实验记录配置、模型版本、显卡、显存、checkpoint、随机种子及定量/可视化结果，避免覆盖已有实验。

## 编码与提交规范

使用 Python 3.10+、四空格缩进和 PEP 8 命名：函数/变量用 `snake_case`，类用 `PascalCase`。优先复用现有 pipeline，不复制核心逻辑。提交摘要应简短明确；PR 需说明数据映射、训练参数、硬件、验证命令和效果对比。面向用户的文档统一使用中文；若修改上游双语文档，则同步维护 `docs/zh/` 与 `docs/en/`。

修改代码时，必须同步更新受影响的注释、docstring、文件头用法示例及手动运行命令，确保它们与实际参数、默认行为和输出路径一致；不得保留已失效的命令或条件语义。
