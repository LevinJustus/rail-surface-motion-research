# 13C5独立运行顺序与文件约定

复现说明：本文保留研究数值和原始SHA；未分发的运行记录链接已省略。当前文档整理不代表重新运行实验。 本文来源文件SHA-256：`ff17e708f3d359aaba69b19a3bfa547cc1da0d176df6a1fe30ea0e016cb97e12`。


本文件说明预定消融的执行入口，不代表全部实验已经验收。状态以主清单和最终结果报告为准。先准备C2固定输入、C3/C4概率与配准缓存、24项冻结训练阈值；第三方图像、模型、概率和掩膜只留本地，源码仓库不分发它们。

## 1. 环境与数值约定

生产及独立计数使用MATLAB R2024b和既有Image Processing Toolbox。算法没有新增网络前向或训练更新，概率复用C4的三个已保存检查点输出。概率文件为256×256、列优先、小端float32，SHA针对实际文件字节。比较阈值前显式转double，不能把阈值降成float32。

独立算子复核依赖NumPy、SciPy、Pillow；实际审计环境保存在各审计报告，运行源也单独复制。来源bootstrap要求支持PCG64和线性quantile的现代NumPy，依赖见requirements-fusion-statistics.txt。统计工具与旧审计环境分开运行，不假定旧NumPy支持新接口。作图仅读取汇总CSV，导出PNG/SVG，CSV可导入Origin。

## 2. 解析与训练来源契约

所有输出ID必须未使用，入口拒绝覆盖。以下为结构示例，需替换为自己的新ID和本地已完成的C4目录；不应复制历史run ID后覆盖旧证据。

```matlab
addpath('matlab');
check_ablation_synthetic('runs/NEW_SYNTHETIC_ID');
run_ablation_inputs('training',20261114,'NEW_TRAINING_ID','','');
```

独立审计不打开训练标签：

```text
python tools/audit_fusion_ablation.py --run-id NEW_TRAINING_ID --synthetic-run NEW_SYNTHETIC_ID --out runs/NEW_TRAINING_AUDIT_ID
```

审计核验全部引用SHA及全量恒等关系，固定首字典序来源、12条件、帧2/6、79变体独立复算；NCC选峰另从RGB复算。通过后才写训练run的contracts.json，验证入口需要该文件。解析检查和训练契约不能替代正式检测评价。

## 3. 三种子验证生产

```matlab
run_ablation_inputs('validation',20261114,'NEW_VALIDATION_14_ID','','NEW_TRAINING_ID');
run_ablation_inputs('validation',20261115,'NEW_VALIDATION_15_ID','NEW_VALIDATION_14_ID','NEW_TRAINING_ID');
run_ablation_inputs('validation',20261116,'NEW_VALIDATION_16_ID','NEW_VALIDATION_14_ID','NEW_TRAINING_ID');
```

第二、第三种子复用第一种子的配准候选和观测扰动缓存；原始概率来自各自的C4检查点。算法只接收已到达前缀。W1/W5使用同一冻结公式，观察历史长度作用；不限长度R仅在当前9帧序列内递归，序列间重置。

predictions.jsonl保留全部逻辑方法行及S/无污染参照，payloads.jsonl登记新增内容寻址载荷。多个逻辑行引用同SHA不意味着新增样本、独立重复或重新推理。恢复实验目录时要保留被引用的C3/C4原文件，不能只复制blobs目录。

## 4. 独立计数、算子复核与来源统计

每个种子的完整预测索引冻结后，另进程评价：

```matlab
evaluate_ablation_counts('NEW_VALIDATION_14_ID','NEW_COUNTS_14_ID');
```

```text
python tools/audit_fusion_ablation.py --run-id NEW_VALIDATION_14_ID --evaluation-run NEW_COUNTS_14_ID --out runs/NEW_AUDIT_14_ID
```

另两个种子同样执行。评价前核验全部概率引用；评价只打开固定验证掩膜，准确生成几何和保留测试不用。独审逐行重算全部两阈值计数及污染变化，元数据与固定来源/条件表核对。

现代Python运行来源统计及第二统计审计：

```text
python tools/summarize_fusion_ablation.py --audit-runs NEW_AUDIT_14_ID NEW_AUDIT_15_ID NEW_AUDIT_16_ID --out runs/NEW_SUMMARY_ID
python tools/audit_ablation_statistics.py --summary runs/NEW_SUMMARY_ID --out runs/NEW_STATISTICS_AUDIT_ID
python tools/summarize_ablation_registration.py --prediction-run NEW_VALIDATION_14_ID --out runs/NEW_REGISTRATION_DIAGNOSTIC_ID
python tools/plot_fusion_ablation.py --summary runs/NEW_SUMMARY_ID --out runs/NEW_PLOT_ID
```

bootstrap复用C4的15来源排序及10000×15抽样索引，不重新选有利抽样；所有预定比较均报告，区间未作多重校正，只作描述性证据。污染以各自无污染输出为参照，帧4—9为后续传播评价，不能把第3帧故障输入下的单帧S当正常对照。

## 5. 复现与结论边界

本项是同图派生的开发验证消融。缓存、内容寻址、共享NCC和审计耗时不能换算完整系统FPS；无标定/实测里程不能换算毫米误差或每米误报。人工反转概率检验历史传播，不代表真实缺陷错误分布。有限窗口清空污染按设计发生，不能单独作为新颖性依据。

tools/run_c5_evaluation_queue.ps1是本机当前运行的一次性依赖队列，不是跨平台安装器，也不创建定时任务；独立复现优先使用上述显式入口及自己的环境路径。GitHub/Word同步保持暂停，论文不纳入此阶段源码材料。
