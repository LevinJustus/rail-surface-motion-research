# Rail Surface Motion

**基于轻量化深度学习与运动补偿的钢轨表面缺陷实时检测方法研究**

本项目以钢轨表面缺陷检测中的运动补偿为核心研究方向，重点研究观测运动先验约束的帧间配准与失效感知多帧融合。轻量化分割模型提供逐帧观测，通过同输入对照与消融检验运动补偿对缺陷检测稳定性的贡献；实时性作为待验证的性能目标。

本仓库包含研究代码、冻结配置、数据来源与划分、实验协议及公开结果摘要。当前开发版本 v0.5.0，新增可独立接入分割概率的Python运动反馈原型、完整反馈开关对照及人工重复纹理挑战。

本项目已完成两种单帧结构的三种子对照、平移配准和v1同条件融合开发验证。v2及去冲突项消融表明强纠错与误报传播存在权衡；v3已实现历史缺陷ROI辅助运动重排序，但在本批12条钢轨来源派生序列上没有改变运动选择，尚无反馈收益。人工挑战存在消歧成功、候选遗漏及误选反例，完整记录见[v3原型报告](docs/algorithm_v3_prototype_results.md)。最终测试、真实连续视频及实时性仍待验证。

## 从这里接续

先阅读[接续开发与复现入口](docs/START_HERE.md)。本轮已完成13E3诊断、13E4单因素消融及v3反馈原型。下一重点是解决困难场景的候选覆盖，并获取存在运动歧义的真实连续数据；不自动重训三种子或扩大已有同图矩阵。

不需要数据或权重的v3演示（依赖见requirements-v3.txt）：

```text
python tools/run_v3_feedback_batch.py --contracts-only --out runs/v3_contracts_NEW
python tools/run_v3_feedback_challenge.py --out runs/v3_challenge_NEW
```

核心接口为`python/rail_motion_v3.py`中的`MotionFeedback.step`。本次提供保守/强纠错×反馈开关及均匀图像核验对照，避免将融合变强误认为运动反馈有效；人工概率不代表U-Net预测性能。

**运动补偿是本项目的核心研究方向；具体机制的新颖性与有效性仍待验证。** 当前v2使用固定运动权重、风险和融合公式，无需单独训练；需要训练的是前端分割网络。GitHub提供源码和协议，不提供MATLAB环境、原始数据或模型权重。

不依赖真实数据或权重的最小演示：

```matlab
addpath('matlab');
check_v2_compensation('runs/v2_contracts_NEW');
```

六个MATLAB核心、18项解析契约、四帧合成RGB与三个概率案例已独立复核，最大float32概率差0。案例同时展示接近阈值恢复、错误单点补偿损伤控制，以及无法纠正当前高置信误报的失败；均不是真实轨面精度。见[算法设计](docs/algorithm_design_v2.md)与[最小实现成果](docs/algorithm_v2_minimal_results.md)。

## 已有结果

单帧对照采用固定的1396张训练图、400张验证图及共同训练配方，按验证IoU选择检查点。下表为三个预定初始化运行的均值±样本标准差，不是置信区间或最终测试成绩。

| 模型 | 参数量 | 缺陷IoU | 缺陷Dice | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 小型U-Net | 482050 | 0.4524±0.0364 | 0.6224±0.0342 | 0.5451±0.0973 | 0.7464±0.0713 |
| MobileNetV2–DeepLabv3+项目实现 | 2740082 | 0.4408±0.0200 | 0.6117±0.0194 | 0.6314±0.0189 | 0.5933±0.0214 |

成熟结构精确率较高、召回率较低；验证选点与实际末轮的比较方向不同。完整保留各个种子及best/last，不将共同配方结果解释为整个模型家族排名。详见[单帧完整结果](docs/step10_full_results.md)、[成熟结构比较](docs/step11_mature_results.md)。

受控序列使用45个训练来源代理组和15个验证来源代理组，各选一幅原图，12种固定条件共720条序列、6480帧。几何、标签、模拟里程及输入隔离通过核查；保留600个空前景帧、48条全空序列与180条里程缺测。它们是60幅源图的关联派生输入，不能当成真实连续视频，详见[序列方法](docs/step12_sequence_protocol.md)及[实际验收](docs/step12_sequence_results.md)。

## 已有单帧诊断与配准验证

本版本对两模型×三种子×best/last共12个检查点进行了阈值诊断，原阈值0.5下4800条逐图计数与历史完全一致。U-Net best的验证诊断选点IoU为0.4757±0.0156（原0.5为0.4524±0.0364）；这是同验证集选阈值的乐观诊断，不是新算法或测试增益。详见[13A报告](docs/step13a_validation_results.md)与[数字曲线](results/validation_diagnostics/)。

配准实验覆盖180条受控验证序列、1440对相邻帧，对比三种平移配准方法：

| 方法 | 有效覆盖率 | 有效估计平均网格EPE/px | 不可用或网格EPE>2px |
|---|---:|---:|---:|
| 直接模拟里程 | 94.79% | 0.550044 | 10.42% |
| 纯视觉NCC | 100% | 0.172131 | 0% |
| 运动先验约束NCC | 100% | 0.172131 | 0% |

两种NCC逐对位移完全一致，当前没有验证先验的额外收益；1440对来自15个关联来源代理组。纯平移未补偿旋转，零主失败也不表示完全对齐。独立NumPy核验全部几何及预选24对NCC通过。共享相关图开销不能用于声称先验加速；没有真实视频或板端成绩。详见[13B协议](docs/step13b_registration_protocol.md)、[结果](docs/step13b_registration_results.md)和[汇总CSV](results/registration/validation_aggregate_metrics.csv)。

## v1融合开发验证及v2边界

v1对180条验证序列执行三种子/八方法比较，77760行两阈值计数独立复现，并按15个来源组给出配对统计。统一0.5下，单帧来源宏IoU为35.947%，视觉补偿固定融合为35.995%，候选GVP为36.014%；微小均值变化未满足完整冻结规则，不能宣称核心创新已成立。见[完整报告](docs/step13c4_validation_results.md)。

v1的C5消融仅部分执行后中止，完整与部分文件在原本机保留，不标验收。当前优先推进v2小规模原理检查。默认v2三帧更新幅度界约0.1011，限制历史损伤也限制严重漏检恢复；当前强前景钳位默认不额外触发。这些是已分析的性质和代价，不是真实精度优势。

## 研究进度

- [公开进度与下一步](docs/PROGRESS.md)
- [版本变更](CHANGELOG.md)与[公开材料范围](docs/PUBLICATION_POLICY.md)

后续按当前接续说明先做v2单检查点小规模初筛，再决定正式实验预算；保留同整数单点及原亚像素强参照。每次版本更新记录具体变更、验证和失败。

## 目录与运行

```text
matlab/          分割模型、训练、评价与受控输入生成、阈值诊断与配准
tools/           数据获取、审计、配置准备与研究结果比较
configs/         冻结模型、训练和运动序列配置
data/manifests/  数据来源、版本、文件与标签审计
data/splits/     保守来源分区、批次顺序与序列源名单
docs/            公开实验方案、结果摘要与进度
results/         同口径逐种子及统计CSV
```

MATLAB部分使用R2024b及Computer Vision、Image Processing和Deep Learning Toolbox。两个网络均从头训练，不需要预训练权重；成熟结构由公共层构造。Python工具按各脚本说明准备NumPy、Pillow、SciPy。请通过Git克隆仓库，运行器会记录Git提交并要求正式训练开始时工作树干净；直接解压ZIP不是完整的训练工作区。

数据需从[MCnet作者仓库](https://github.com/zdfcvsn/MCnet)独立获取，按[基线准备记录](docs/baseline_preparation.md)验证文件与冻结分区。下载器的新增元数据写入本机忽略目录，例如`--metadata-out runs/download_metadata`，不要覆盖已冻结的`data/manifests`。原作者目录名不决定修订后的分区；不将自然保留测试用于调参。复现无需重新随机分区或改写原审计。先导计划的安全认证可在本机重建：

```text
python tools/prepare_stage1_plan.py
```

它只校验固定名单并在被忽略的本机证据目录保存认证，不启动训练。MATLAB入口示例：

```matlab
addpath('matlab');
check_data_contract;
check_baseline_environment;
generate_controlled_sequences(DATA_ROOT, ...
    'configs/motion_sequences_v2.json', 'step12_NEW_UNIQUE_RUN', 'full');
```

生成器要求新运行目录，拒绝覆盖。不要把准确几何、中心标签或评价目录传入后续待测算法。重新训练和完整实验请先阅读相应冻结协议，不能将前向检查或输入生成当作检测收益。

## 数据与结论边界

原始图像、标签、下载分卷、模型权重、生成序列及含原图的面板不随本仓库分发。带本机路径的完整训练/渲染日志、内部协作信息和本机Git历史也不公开。公开报告是注明范围的派生摘要，原始字节证据在本机保存；其SHA不能误当成公开派生文档的SHA。第三方数据的许可独立于本项目代码，见[第三方说明](THIRD_PARTY_NOTICES.md)。

安全的冻结数据元信息、配置和划分保留原字节，以维持SHA检查链。个别公开代码若仅对CPU环境探测增加容错，会在公开来源记录中列明；这不代表重新训练或新增性能结果。每次更新先检查公开材料，再以版本号与提交记录保存；无内容变化不制造空提交。

当前没有真实运动视频、实际编码器、毫米定位、内部伤损、K230板端或端到端实时性能结果。标题中的运动补偿和实时性是待验证研究目标，不提前写作已经达到的应用能力。

## 开发与版本

本项目按功能和实验阶段维护版本，通过提交记录、CHANGELOG和固定配置追踪变化，并如实记录实验结果的改善与退化。本仓库提供研究实现与复现说明，论文正文及源稿不在仓库中发布。原创代码许可证尚待确定，第三方材料遵循各自授权。

配准复现入口（先完成第12项生成；DATA_ROOT与生成运行名按实际指定）：

```matlab
inputRoot=fullfile(pwd,'runs','step12_NEW_UNIQUE_RUN','algorithm_inputs');
cfgPath=fullfile(pwd,'configs','registration_translation_v1.json');
run_registration_inputs(inputRoot,cfgPath,'contract','registration_contract_NEW');
% 先审查契约，再按冻结参数执行validation。
run_registration_inputs(inputRoot,cfgPath,'validation','registration_validation_NEW');
```

算法结束后，用NumPy/Pillow单独评价，几何目录仅传给此评价步骤：

```text
python tools/evaluate_registration.py --run runs/registration_contract_NEW --sequences runs/step12_NEW_UNIQUE_RUN
python tools/evaluate_registration.py --run runs/registration_validation_NEW --sequences runs/step12_NEW_UNIQUE_RUN
```

不可根据验证结果回调本版参数后再将同一结果称为独立测试。每个运行和evaluation目录必须使用新名称，失败产物保留。
