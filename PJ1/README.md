# PJ1（codes）README

本 README 基于当前 `codes/` 目录代码状态编写，并结合你在 `codes/mynn/` 中已完成的实现。

## 1. 项目概述

本项目基于自定义的 NumPy 神经网络框架（`mynn`），在 MNIST 数据集上完成手写数字分类模型的训练与评估。

## 2. 环境依赖

本项目运行前需要先安装以下第三方包：

- `numpy`
- `matplotlib`
- `tqdm`
- `Pillow`

推荐直接使用下面的命令安装：

```bash
pip install numpy matplotlib tqdm Pillow
```

如果你使用 Conda，也可以：

```bash
conda install numpy matplotlib tqdm pillow
```

当前支持的模型：

- `mlp`：多层感知机
- `cnn`：轻量卷积神经网络

当前已具备的功能：

- 训练模型并保存最佳权重
- 在 MNIST 测试集上评估已保存模型
- 一次性对比 MLP 与 CNN 的测试准确率并出图
- 绘制混淆矩阵、误分类样本、模型权重/卷积核可视化

## 3. 目录与脚本说明

### 3.1 `codes/` 当前目录结构

- `.vscode/`：VS Code 工作区配置目录
- `best_models/`：保存训练得到的最佳模型权重
    - `best_models/with_lr_delay/`：使用学习率调度训练得到的模型
    - `best_models/without_lr_delay/`：未使用学习率调度训练得到的模型
        - `cnn/`：CNN 模型权重
        - `mlp/`：MLP 模型权重
- `dataset/`：数据集目录
    - `dataset/MNIST/`：MNIST 原始数据文件
- `draw_tools/`：绘图辅助脚本目录
    - `draw_tools/draw.py`：手写数字涂写/交互绘图工具
    - `draw_tools/plot.py`：训练曲线绘图工具
- `figs/`：实验图像输出目录
- `mynn/`：自定义神经网络框架核心实现
- `dataset_explore.ipynb`：数据探索 Notebook
- `compare_models.py`：对比 MLP 和 CNN 的测试准确率
- `compare_lr_modes.py`：对比有/无学习率调度时的测试准确率
- `test_model.py`：加载已训练模型并在测试集上评估
- `test_train.py`：训练模型并保存最佳权重
- `weight_visualization.py`：混淆矩阵、误分类样本和权重/卷积核可视化
- `idx.pickle`：训练集打乱索引缓存文件
- `README.md`：当前说明文档

### 3.2 主要脚本功能

- `test_train.py`：负责读取 MNIST、划分训练/验证集、构建模型、训练并保存最佳权重
- `test_model.py`：负责加载指定模型并输出测试集准确率
- `compare_models.py`：对比 MLP 与 CNN 的测试准确率，并保存对比图
- `compare_lr_modes.py`：对比有/无学习率调度两组实验的测试准确率，并保存图像
- `weight_visualization.py`：生成混淆矩阵、误分类样本和模型参数可视化结果
- `draw_tools/draw.py`：交互式绘图相关工具
- `draw_tools/plot.py`：训练过程中的损失/准确率曲线绘图工具
- `mynn/`：包含层、优化器、调度器、训练循环和指标计算等核心实现

模型与图像输出：

 
- 默认最佳模型（不使用学习率调度）：`best_models/without_lr_delay/mlp/best_model.pickle`、`best_models/without_lr_delay/cnn/best_model.pickle`
- 可视化图像输出目录：`figs/without/<model>/` 或 `figs/with/<model>/`

## 4. `mynn` 中你已完成的实现

根据当前代码，以下模块已实现并可运行：

### 4.1 `mynn/op.py`

- `Linear.forward` 与 `Linear.backward`
- `conv2D` 前向与反向（im2col 风格前向 + 对应反向梯度传播）
- `Flatten` 前向与反向
- 带可选 softmax 的 `MultiCrossEntropyLoss`
- `softmax` 辅助函数

说明：

- `L2Regularization` 目前仍是占位（`pass`）
- 权重衰减当前在可训练层（`Linear`/`conv2D`）中通过 `weight_decay` 与 `weight_decay_lambda` 控制

### 4.2 `mynn/models.py`

- `Model_MLP`：`Linear + ReLU` 堆叠，支持按层设置权重衰减
- `Model_CNN`：`conv -> relu -> conv(stride=2) -> relu -> flatten -> fc`
- MLP/CNN 的模型保存与加载

### 4.3 `mynn/optimizer.py`

- `SGD` 已实现

### 4.4 `mynn/lr_scheduler.py`

- `StepLR` 已实现并在训练脚本中使用

### 4.5 `mynn/runner.py` 与 `mynn/metric.py`

- `RunnerM`：mini-batch 训练/评估循环、周期性评估日志、最佳模型保存
- `accuracy`：MNIST 分类 top-1 准确率

## 5. 数据说明

MNIST 文件应位于：

- `dataset/MNIST/train-images-idx3-ubyte.gz`
- `dataset/MNIST/train-labels-idx1-ubyte.gz`
- `dataset/MNIST/t10k-images-idx3-ubyte.gz`
- `dataset/MNIST/t10k-labels-idx1-ubyte.gz`

`test_train.py` 当前使用固定随机种子（`309`），并采用如下划分：

- 打乱后前 10000 条作为验证集
- 剩余样本作为训练集

## 6. 快速开始

建议在 `codes/` 目录下执行以下命令。

1. 训练 MLP / CNN

```bash
python test_train.py --model mlp
python test_train.py --model cnn
```

说明：以上默认命令不启用学习率调度，模型默认保存到 `best_models/without_lr_delay/<model>/best_model.pickle`。

常用参数：

- `--epochs`（不传时自动设置：启用调度时 MLP=15、CNN=15；关闭调度时=15）
- `--log-iters`（默认 `100`）
- `--save-dir`（自定义模型保存目录）
- `--use-lr-scheduler`（启用学习率调度，默认关闭）
- `--no-lr-scheduler`（关闭学习率调度）

学习率衰减对比实验可使用不同保存目录，例如：

```bash
python test_train.py --model mlp --epochs 15 --no-lr-scheduler --save-dir ./best_models/without_lr_delay/mlp
python test_train.py --model cnn --epochs 15 --no-lr-scheduler --save-dir ./best_models/without_lr_delay/cnn

python test_train.py --model mlp --epochs 15 --use-lr-scheduler --save-dir ./best_models/with_lr_delay/mlp
python test_train.py --model cnn --epochs 15 --use-lr-scheduler --save-dir ./best_models/with_lr_delay/cnn
```

2. 测试已保存模型

```bash
#默认测试不启用学习率调度
python test_model.py --model mlp
python test_model.py --model cnn

# 指定从有/无学习率调度实验目录读取默认模型
python test_model.py --model mlp --lr with
python test_model.py --model mlp --lr without
```

也可手动指定模型路径：

```bash
python test_model.py --model cnn --model-path ./best_models/with_lr_delay/cnn/best_model.pickle
```

3. 对比 MLP 与 CNN 测试准确率

```bash
#默认对比不启用学习率调度
python compare_models.py

# 指定对比哪一组实验结果
python compare_models.py --lr with
python compare_models.py --lr without
```

输出图像：

- `figs/compare_accuracy.png`

另外，使用 `compare_lr_modes.py` 脚本可以直接比较有/无学习率调度对准确率的影响并保存图：

```bash
python compare_lr_modes.py --save-figs
```

输出图像：

- `figs/compare_lr_modes.png`

4. 可视化与错误分析

```bash
#默认可视化不启用学习率调度
python weight_visualization.py --model mlp
python weight_visualization.py --model cnn

# 指定从有/无学习率调度实验目录读取默认模型
python weight_visualization.py --model mlp --lr with
python weight_visualization.py --model mlp --lr without
```

常见输出：

- `<model>_confusion_matrix.png`
- `<model>_misclassified_examples.png`
- `mlp_weights_layer_*.png` 或 `cnn_kernels_layer_*.png`

输出图像：

- 默认不启用学习率调度时：`figs/without/<model>/`
- 启用学习率调度时：`figs/with/<model>/`

## 7. 学习率衰减对比实验设置

本项目当前按两组实验进行对比：

1. 无学习率衰减（`without-lr-delay`）

- 做法：使用 `--no-lr-scheduler` 关闭学习率调度。
- 结果保存目录：`figs/without/<model>/`

2. 有学习率调度（`with-lr-delay`）

- 做法：使用当前代码中的 `test_train.py` 参数设定进行训练（会在训练过程中触发学习率调度）。
- 结果保存目录：`figs/with/<model>/`

当前代码中的默认训练配置（来自 `test_train.py`）：

- MLP：
    - 优化器：`SGD(init_lr=0.06)`
    - 学习率调度：`StepLR(step_size=6, gamma=0.5)`
    - 默认 epoch：启用调度时 `15`；关闭调度时 `15`
    - 权重衰减系数：`[1e-4, 1e-4]`
- CNN：
    - 优化器：`SGD(init_lr=0.06)`
    - 学习率调度：`StepLR(step_size=6, gamma=0.5)`
    - 默认 epoch：启用调度时 `15`；关闭调度时 `15`
    - 权重衰减系数：`[1e-4, 1e-4, 1e-4]`

建议：完成训练后，将对应曲线图或对比图移动/保存到上述两个目录，便于写实验报告时直接对照。



