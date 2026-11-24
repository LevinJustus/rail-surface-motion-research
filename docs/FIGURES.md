# 实验图表来源

仓库图表由 Matplotlib 根据实验输出绘制。人工示例使用程序构造的概率图和标签，展示特定机制和失败条件。受控序列图汇总钢轨来源图像派生序列的实验结果。

| 图表 | 输入与实验说明 | 绘图源码 |
| --- | --- | --- |
| [v2 人工机制案例](evidence/algorithm_v2_minimal/v2_artificial_examples.svg) | [人工概率、人工掩膜与融合输出](algorithm_v2_minimal_results.md) | [plot_v2_artificial_examples.py](../tools/plot_v2_artificial_examples.py) |
| [v3 反馈概览](evidence/algorithm_v3/v3_feedback_summary.svg) | [派生序列结果和人工重复纹理挑战](algorithm_v3_prototype_results.md) | [plot_v3_feedback.py](../tools/plot_v3_feedback.py) |
| [受控融合验证](evidence/step13c4/fusion_validation_summary.svg) | [验证汇总与来源配对区间](step13c4_validation_results.md) | [plot_fusion_validation.py](../tools/plot_fusion_validation.py) |
| [E3 历史纠错诊断](evidence/step13e3/v2_screening_summary.svg) | [固定检查点与钢轨来源开发序列](step13e3_training_diagnostic_results.md) | [plot_v2_screening.py](../tools/plot_v2_screening.py) |
| [E4 冲突项消融](evidence/step13e4/v2_conflict_ablation_summary.svg) | [保留和移除冲突门控的对照](step13e4_conflict_ablation_results.md) | [plot_v2_conflict_ablation.py](../tools/plot_v2_conflict_ablation.py) |
| [准确对齐配对区间](evidence/heldout_exact_alignment/oracle_alignment_effect.svg) | [保留来源准确几何对齐](heldout_exact_alignment_results.md) | [plot_oracle_alignment.py](../tools/plot_oracle_alignment.py) |
| [退化条件分层](evidence/heldout_exact_alignment/oracle_by_degradation.svg) | [准确对齐分条件结果](heldout_exact_alignment_results.md) | [plot_oracle_alignment.py](../tools/plot_oracle_alignment.py) |

## 实拍输入示例

[片段 B 第 167 帧](evidence/real_video_development/clip_B_f000167_rail_head.png)和[片段 C 第 174 帧](evidence/real_video_development/clip_C_f000174_rail_head.png)直接取自实拍视频开发实验的模型输入。裁剪、缩放和帧编号见[采集条件与输入示例](real_video_development.md#采集条件与输入示例)。

各报告列明输入范围、指标和可用的结果文件。运行方式见[快速开始](START_HERE.md)。
