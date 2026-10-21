# H1 二跳边消息：服务器快照专用离线补丁

来源压缩包：`higgs_server_source_20261016_142815.tar.gz`

来源 SHA256：`fe8f4cdfd96fa140084b127c4c09515383439052468e8d958ea274f26d1bd6b4`

本补丁由上传的服务器源码制作，未覆盖电脑原项目。服务器包包含 121 个源码文件；与本机项目比较，75 个相同、18 个不同、28 个仅服务器存在。所有服务器专属改动均作为基线保留。

## 改动范围

仅修改一个已有文件 `dgd/diffusion_model_beta_h1.py`，新增三个 H1 专用模型辅助文件、一个新 model 配置和一个测试文件。完整 diff 在 `changes.diff`。

- `model=beta_h1` 及旧 checkpoint 缺少 `two_hop` 配置时，继续实例化原 GraphTransformer，无新增参数。
- `model=beta_h1_twohop` 才启用新模块；family 仍为 `beta_h1`，stage 仍为 `h1`，现有 HiGGs loader 可按 checkpoint 自带 cfg 加载。
- 新模块在去噪网络最后的隐藏边表示上聚合二跳边消息，并加一个零初始化残差。默认 16 个消息通道、64 维边/全局隐藏状态时新增 11,489 个参数。
- H2、Original Glue、共享 GraphTransformer、Beta forward/reverse、KLUB、schedule、eta、T、threshold、先验、LCC、条件适配、评价脚本与原配置均未修改。
- 这是借鉴 Local PPGN 的稠密 H1 适配，不是原论文完整方法复现。稠密计算为 O(B*C*N^3)，只用于 H1 社区图。
- 质量提升尚未验证；零初始化只保证初始预测与原网络相同，需要训练新 H1 才能学习改进。

## 安装及验证（Ubuntu，无需联网/不执行 Git）

将 `h1_twohop_patch_20261016.tar.gz` 上传到项目根目录。以下命令每次从头复制，避免重新登录后变量为空。

```bash
cd /data2/wenb/ylu/HiGGs-GBD/HiGGs-main
PY=/data2/wenb/wbai/miniconda3/envs/higgs/bin/python

tar --keep-old-files -xzf h1_twohop_patch_20261016.tar.gz

"$PY" h1_twohop_patch_20261016/apply_patch.py --root "$PWD" --check
```

先检查输出的 1 个 MODIFY 和 5 个 ADD。出现 hash mismatch/文件已存在等错误时，先保留错误信息，不强行覆盖。检查通过后：

```bash
cd /data2/wenb/ylu/HiGGs-GBD/HiGGs-main
PY=/data2/wenb/wbai/miniconda3/envs/higgs/bin/python

"$PY" h1_twohop_patch_20261016/apply_patch.py --root "$PWD" --apply
```

安装器先核验 payload 与相关原始文件的 SHA256，自动备份唯一被修改的 H1 原文件，最终打印备份目录。已全部安装且内容一致时再次执行会直接报告 Already applied。它不会加载数据或启动训练。

随后运行测试。使用标准 unittest，无须安装 pytest；这是小型合成图测试，checkpoint 只写在临时目录并在结束后清理：

```bash
cd /data2/wenb/ylu/HiGGs-GBD/HiGGs-main
PY=/data2/wenb/wbai/miniconda3/envs/higgs/bin/python

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$PY" tests/test_h1_twohop.py

PYTHONDONTWRITEBYTECODE=1 "$PY" dgd/main.py \
  model=beta_h1_twohop dataset=cora dataset.h=1 --cfg job
```

最后一条只解析配置，不训练，也不实例化数据集。

## 已完成验证

本机 PyTorch 2.0.1、Lightning 1.7.7 环境中，20 项测试通过：配置类型检查、旧权重严格加载、零初始化时预测逐元素相同、对称/零对角/padding、排列等变、节点数 1/2/5 的有限梯度、4/7 类节点损失与梯度、短 reverse 的输出契约、warm start 配置一致性、checkpoint 重载，以及原服务器 4 项 H1 语义回归。

一轮合成数据训练使用 2 个 train batch 与 2 个 val batch，train/val 的 `h1_total` 有限，validation-best 和 last checkpoint 均成功保存和读取。没有运行真实数据训练、真实数据 full reverse 或 MMD。

## 后续训练前需要读取实际 checkpoint 配置

不要直接以新配置的默认 train.batch_size=8 启动正式训练：原 Cora/Facebook checkpoint 的训练 batch_size 会影响 processed 文件路由。先核对源 checkpoint 保存的 cfg。下面只读命令不需要 GPU：

```bash
cd /data2/wenb/ylu/HiGGs-GBD/HiGGs-main
PY=/data2/wenb/wbai/miniconda3/envs/higgs/bin/python

PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "dgd"))
from higgs.higgs_utils import _load_checkpoint_cpu

paths = [
    root / "saved_models/cora/cora_h1.ckpt",
    root / "outputs/2026-07-30/10-58-34/checkpoints/fb_beta_h1_r10_e500/last.ckpt",
]
for path in paths:
    if not path.is_file():
        print("MISSING:", path)
        continue
    checkpoint = _load_checkpoint_cpu(path)
    cfg = checkpoint["hyper_parameters"]["cfg"]
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    print("\nCHECKPOINT:", path)
    print("epoch:", checkpoint.get("epoch"), "global_step:", checkpoint.get("global_step"))
    for key in ("dataset", "model", "train"):
        print(key + ":")
        print(OmegaConf.to_yaml(cfg[key], resolve=True))
PY
```

把这段输出与服务器测试结果发回后，才能基于真实 checkpoint 和 cache 路由固定微调命令，避免用新默认值无意改变数据来源。无需再次上传 checkpoint 或数据。

## warm start 和恢复的含义

后续训练可显式设置：

```text
model.two_hop.warm_start_checkpoint=/absolute/path/to/original_beta_h1.ckpt
```

该路径只在新训练的 `on_fit_start` 读取；新模型推理不会打开它。它只复制原 H1 权重，使用新的优化器，允许缺失的参数严格限于新模块。关键模型/数据设置、原参数形状和 train-derived prior 必须一致，否则明确失败。不会重新保存或修改源 checkpoint。

继续已经训练的 two-hop 实验应使用原项目 `general.resume` 加载新 checkpoint。`on_load_checkpoint` 会阻止再次执行旧权重 warm start。不能把旧 checkpoint 当作新结构的 optimizer-resume 文件。

本补丁不切换现有正式采样 checkpoint，也不启动 500/1000 epoch 正式训练。
