# 接续开发与复现入口

复现说明：本文保留研究数值和原始SHA；未分发的运行记录链接已省略。当前文档整理不代表重新运行实验。 本文来源文件SHA-256：`74796cb853bcd3f481f8b6c2deef582226471a9c5e5c2dca36280c6468492cab`。


版本：v0.4.0。当前直接下一项：**13E3，用一个已有检查点、最多12条训练来源序列检查v2是否值得继续扩大实验。** 先确认环境与本机已有数据/检查点，不自动重训三种子或重启旧C5矩阵。

## 当前算法与训练的关系

| 组成 | 当前状态 | 是否需要训练 |
|---|---|---|
| 小型U-Net分割器 | 已有三种子研究结果，权重只留本机 | 需要训练；同机接续优先复用已有检查点 |
| v2运动候选、补偿、风险与融合 | 六个MATLAB核心、18项解析契约完成 | 固定公式，无需额外训练；参数与效果需开发验证 |
| 学习型光流/配准网络 | 当前未采用 | 若未来另行采用，训练/预训练另立方案 |

标准运动补偿本身不是已确认的新颖性。本项目的具体候选为把多位移歧义、候选截断和视野未知质量传递到像素级缺陷更新风险。v1未证明先验具有额外收益；v2也还没有真实数据上的有效性结论。

先读[算法设计](algorithm_design_v2.md)、[最小实现成果及失败](algorithm_v2_minimal_results.md)和[当前进度](PROGRESS.md)。早期C1—C4报告保留当时的方案/结果，不能覆盖本页接续顺序。

## 从干净克隆运行不依赖数据的演示

```text
git clone --branch v0.4.0 https://github.com/LevinJustus/rail-surface-motion-research.git
cd rail-surface-motion-research
```

MATLAB R2024b及Image Processing Toolbox用于本演示；真实网络训练还需要Deep Learning及相关工具箱。使用新的输出ID，入口拒绝覆盖：

```matlab
addpath('matlab');
check_v2_compensation('runs/v2_contracts_NEW');
```

该命令生成自己的随机纹理和人工概率，不下载数据、不加载权重、不训练网络。预期18项契约通过；输出contracts.json、synthetic_stream.mat、controlled_examples.mat/CSV。人工案例含当前高置信误报无法纠正的失败，不是轨面数据成绩。

可选Python独立复核/绘图：

```text
python tools/audit_v2_minimal.py --run-id v2_contracts_NEW --out runs/v2_audit_NEW
python tools/plot_v2_artificial_examples.py --run runs/v2_contracts_NEW --out runs/v2_plots_NEW
```

依赖范围见requirements-v2.txt，实际历史验收环境见结果报告。现代依赖范围不是已经完成全新安装验收的声明。所有演示输出只写入被忽略的runs目录。

## 接入已有网络概率

```matlab
cfg = jsondecode(fileread('configs/algorithm_design_v2.json'));
state = [];
observation = struct('timestamp_s',0,'odometry_px',0, ...
                     'odometry_available',false);
[outputProbability,state,diagnostic] = v2_compensation_step( ...
    rgb_uint8_256x256x3, raw_probability_256x256, observation, state, cfg);
```

RGB和概率由调用方提供；时间戳递增，不同序列传空state。没有编码器时可设观测不可用，仍可运行视觉分支。模拟里程不得写成真实编码器实测。

## 继续真实数据研究需要什么

同一台电脑的新会话应先定位现有规范研究目录、数据根目录、已有U-Net检查点及C3/C4概率缓存。不要仅为接续会话重新下载/训练；也不要将私人绝对路径写回公开文件。不同电脑或新工作树未必有这些未提交文件。

新电脑至少需要MATLAB/工具箱、从作者入口独立获取的MCnet数据，以及自行训练或通过私人渠道取得且核验身份的检查点。公开仓库没有原始数据和权重，因此GitHub链接本身不能直接恢复历史训练状态。

数据入口为[MCnet作者仓库](https://github.com/zdfcvsn/MCnet)，版本与校验记录见data/manifests。下载器把本机新元数据写入runs/download_metadata等忽略目录，不覆盖冻结记录。数据根目录结构以data/splits/conservative_group_v1.json的相对image/mask路径为准，不按原作者文件夹名重新随机划分。

确有必要从头训练时，先核验基线环境和固定分区，再按[完整单帧报告](step10_full_results.md)执行：

```matlab
train_full_baseline(DATA_ROOT, ...
    'configs/training_stage2_full_v2.json','full_baseline_NEW');
```

这是原三种子完整训练，**不属于默认下一步初筛**。正式训练入口要求清洁的Git工作树，检查点和原始图像保持本机忽略；实际成本取决于本机设备，未指定固定时长。

## 已有结果与尚未具备的复现条件

v1的C3/C4和中止的C5入口包含原本机运行依赖，例如固定检查点位置、概率缓存、C3源快照和生成序列目录。这些runs材料不随GitHub分发。其协议/安全汇总公开，但单凭克隆不能声称重新执行了整个历史矩阵；需要先验证本机材料，或另版本准备可移植入口。

v2最小契约与人工演示不依赖这些历史运行。真实来源的13E3驱动器尚待实现/验收，不能把该阶段写成已能一键运行。初筛必须保留单帧、同整数运动单点对照、原亚像素强参照与v1视觉门控，区分候选机制与采样精度；只用训练来源，不接触保留测试。

## 接续时应交付什么

先汇报环境和材料是否齐备；随后完成13E3这一项，给出实际运行、净收益与新增漏检/误报、失败及下一步。若普遍拒绝历史而仅退回单帧，不将其视为有效创新。正式扩大实验、网络重训和发布均按实际任务授权推进，论文始终单独本地保留。
