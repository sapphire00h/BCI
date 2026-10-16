# MOABB + Braindecode 运动想象 EEG 实验

本项目使用本地 **BCI Competition IV 2a** 数据集（MOABB：`BNCI2014_001`），比较 EEG 分类模型在跨会话任务中的表现。每名受试者独立训练：使用 T 会话的 288 个 Trial 训练，使用 E 会话的 288 个 Trial 测试。

## 项目文件

| 文件 | 作用 |
| --- | --- |
| `train_predict.ipynb` | 配置、训练、评估及结果输出 |
| `data_adapter.py` | 本地数据检查、MOABB 数据读取和模型输入维度适配 |
| `eegnet_model.py` | 自定义 EEGNet、EEGTCNet 和 EEGNex 模型 |

`other/` 存放历史或辅助代码，不参与主 Notebook 的运行。

## 环境与数据

项目使用 Anaconda 的 `moabb` 环境。运行 Notebook 时，选择该环境作为内核，并确保其中已安装代码使用的 MOABB、Braindecode、PyTorch、scikit-learn、Skorch、NumPy、Matplotlib 和 Jupyter。

默认从项目根目录的以下位置读取数据：

```text
Data/MNE-bnci-data/~bci/database/001-2014/
├── A01T.mat
├── A01E.mat
├── ...
├── A09T.mat
└── A09E.mat
```

Notebook 使用当前工作目录作为项目根目录，因此应**从项目根目录打开并运行**。缺少所选受试者的 T 或 E 文件时，程序会报错。

## 运行方法

1. 在 `train_predict.ipynb` 中设置 `DATA_CONFIG["subjects"]` 和 `DATA_CONFIG["data_root"]`。
2. 设置 `MODEL_NAME`，一次选择一个模型。
3. 按需调整 `TRAIN_CONFIG`、`OUTPUT_MODE` 和 `SAVE_CHECKPOINT`。
4. 使用 `moabb` 内核，从上到下运行 Notebook 单元。

当前默认选择 `CustomEEGTCNet`，受试者为 1–9，使用 CUDA、Adam 优化器、交叉熵损失和 750 个 Epoch。可选模型包括三个自定义模型，以及 Braindecode 的 `BDEEGNet`、`ATCNet`、`TCFormer`、`Deep4Net` 和 `ShallowFBCSPNet`。

## 数据处理与评估

当前 `MotorImagery` 配置**不额外进行高通、低通滤波或重采样**。MOABB 将数据切分为 Trial；Pipeline 随后在训练 Trial 上拟合按通道、按时间点的标准化器，并将同一标准化器应用于测试 Trial。`adapt_input()` 根据模型要求，将数据调整为三维或四维输入。

评估采用固定的单向 **T→E** 划分：

- `0train`：288 个训练 Trial；
- `1test`：288 个测试 Trial；
- 不设置内部验证集；
- 每名受试者独立训练一次，E 会话在训练结束后评估。

训练结束时，程序恢复**训练 Loss 最低**的 Epoch 的模型权重。该选择依据是训练集 Loss，不是验证集表现。

## 结果

默认结果保存在 `results/<模型名>/`。`OUTPUT_MODE="experiment"` 时，改用 `EXPERIMENT_NAME` 指定目录名。输出包括：

- `report.txt`：实验参数、逐受试者指标，以及逐受试者均值和标准差；
- `metrics.txt`：逐折、逐受试者及总体指标和每轮训练 Loss，采用 `key=value` 格式；
- `sub<编号>/`：训练 Loss 曲线、训练集与测试集混淆矩阵；
- MOABB 评估结果文件；
- 可选模型权重：仅在 `SAVE_CHECKPOINT=True` 时保存。

`report.txt` 展示 accuracy、macro-F1、kappa 和测试 Loss，数值保留四位小数；`metrics.txt` 还记录 balanced accuracy。

## 当前适用范围

数据读取器、本地文件路径、`0train/1test` 会话名及每会话 288 个 Trial 的检查，均针对 `BNCI2014_001` 编写。更换其他数据集时，需要相应调整 `data_adapter.py`、Notebook 中的范式与划分，以及报告中的数据集和会话说明。

Notebook 末尾的多模型历史结果对比函数尚未实现。