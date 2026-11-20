# 快速开始

本页介绍Python运动反馈示例、原始概率接入和MATLAB分割训练。方法与对照关系见[v3实验报告](algorithm_v3_prototype_results.md)。

条件式历史纠错的运行入口、人工检查与开发批次结果见[对齐与门控实验](algorithm_conditional_gate_study.md)。

## 安装与运行示例

```bash
git clone https://github.com/LevinJustus/rail-surface-motion-research.git
cd rail-surface-motion-research
python -m pip install -r requirements-v3.txt
python tools/run_v3_feedback_batch.py --contracts-only --out runs/v3_contracts_demo
python tools/run_v3_feedback_challenge.py --out runs/v3_challenge_demo
```

Python依赖为NumPy、SciPy、Pillow，绘图使用Matplotlib。历史运行环境及版本见相应实验报告。

| 示例 | 内容 | 主要输出 |
|---|---|---|
| 行为检查 | 位移方向、越界、运动反馈、缓存与概率范围等11项检查 | `contracts.json` |
| 人工场景 | 3个固定种子下的重复纹理、严重漏检和稳定错误历史，共9条序列 | `metrics.csv`、`motion_metrics.csv`、`summary.json` |

人工场景用于观察机制行为，输入概率由脚本构造。每次使用新的输出目录名；生成结果统一写入`runs/`。

## 接入分割概率

在仓库根目录使用：

```python
import sys
sys.path.insert(0, 'python')
from rail_motion_v3 import MotionFeedback

stream = MotionFeedback()
outputs, diagnostic = stream.step(
    rgb_uint8, raw_probability, timestamp_s=0.0
)
probability = outputs['D_feedback_strong']
```

- `rgb_uint8`：形状为256×256×3的uint8图像。
- `raw_probability`：与图像对齐的256×256缺陷概率，取值在[0,1]。
- `timestamp_s`：递增时间戳；新序列创建新的`MotionFeedback`实例。
- `odometry_px`：可选的累计观测位移，单位为图像像素。使用实际编码器时先完成物理位移到图像位移的标定。

接口返回五个实验分支：

| 输出键 | 运动选择 | 融合 |
|---|---|---|
| `B_base_conservative` | 基础候选 | 保守 |
| `C_feedback_conservative` | 历史ROI反馈 | 保守 |
| `E_base_strong` | 基础候选 | 强纠错 |
| `D_feedback_strong` | 历史ROI反馈 | 强纠错 |
| `F_uniform_strong` | 均匀区域图像核验 | 强纠错 |

`diagnostic`包含候选位移、局部评分和实际选择。比较C/B、D/E可分别观察相同融合下的反馈作用，D/F用于比较ROI与均匀核验。

## MATLAB示例

v2示例使用MATLAB R2024b和Image Processing Toolbox：

```matlab
addpath('matlab');
check_v2_compensation('runs/v2_contracts_demo');
```

输出包括18项解析检查、合成图像流和人工概率案例。Python复核与绘图入口：

```bash
python tools/audit_v2_minimal.py --run-id v2_contracts_demo --out runs/v2_audit_demo
python tools/plot_v2_artificial_examples.py --run runs/v2_contracts_demo --out runs/v2_plots_demo
```

依赖见`requirements-v2.txt`，公式与接口见[v2设计](algorithm_design_v2.md)。

## 训练分割基线

先按[数据与复现](PUBLICATION_POLICY.md)获取MCnet数据，并核对固定分区。训练使用MATLAB R2024b及Deep Learning、Image Processing和Computer Vision工具箱。

```matlab
addpath('matlab');
train_full_baseline(DATA_ROOT, ...
    'configs/training_stage2_full_v2.json', 'full_baseline_demo');
```

`DATA_ROOT`为数据根目录，运行目录使用新的名称。正式训练入口记录Git提交并检查工作树状态；运行前提交自己的代码修改。完整设置、模型保存和评价口径见[单帧基线报告](step10_full_results.md)。已有检查点可直接用于概率推理与融合实验。

## 复现实验

| 任务 | 文档 |
|---|---|
| 模型与训练对照 | [单帧基线](step10_full_results.md)、[成熟结构比较](step11_mature_results.md) |
| 生成受控运动序列 | [序列生成协议](step12_sequence_protocol.md) |
| 配准比较 | [配准协议](step13b_registration_protocol.md)、[结果](step13b_registration_results.md) |
| 概率融合与对照 | [融合协议](step13c2_fusion_protocol.md)、[开发验证](step13c4_validation_results.md) |
| v2风险诊断与消融 | [有界诊断](step13e3_training_diagnostic_results.md)、[冲突项消融](step13e4_conflict_ablation_results.md) |
| v3运动反馈 | [原型与人工挑战](algorithm_v3_prototype_results.md) |

历史批次驱动器按协议读取图像、原始概率、运动记录和评价标签。准备对应缓存后可运行相同对照；自定义输入可优先通过`MotionFeedback.step`接入。输入格式与来源见[数据与复现](PUBLICATION_POLICY.md)。

---

[项目主页](../README.md) · [研究进展](PROGRESS.md)
