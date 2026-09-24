# RLSM-IMMTrack 使用说明

完整安装、命令、指标定义和结果见 [英文 README](README.md)。

这是在 SUTrack-B224 上加入 IMM 置信度闭环、未来验证的双层外观记忆、隔离影子分支的推理代码。包内包含 RLSM、SafeCL 和原版 SUTrack 的统一运行入口，不包含数据集、模型权重或本地虚拟环境。

## 快速开始

1. 按英文 README 创建 Python 3.9 环境，安装 PyTorch 1.11.0 + CUDA 11.3 及 requirements.txt。
2. 下载官方 SUTrack-B224 的 SUTRACK_ep0180.pth.tar，校验 SHA-256，放到 checkpoints/。
3. 准备 UAV2UAV-2 或 A2ATracking 数据，先运行 validate。
4. 使用 evaluate 开始评测；默认运行所有序列、所有对齐帧，不再默认为五个序列。
5. 使用 score 离线重算指标，使用 plot 绘制三类曲线。

~~~bash
python -m rlsm_immtrack evaluate --tracker rlsm --dataset-root data/UAV2UAV-2 --alignment common --checkpoint checkpoints/SUTRACK_ep0180.pth.tar --output runs/uav_rlsm
python -m rlsm_immtrack evaluate --tracker rlsm --dataset-root data/A2ATracking --checkpoint checkpoints/SUTRACK_ep0180.pth.tar --output runs/a2a_rlsm
~~~

将 --tracker 改为 sutrack 或 safecl，并使用新的输出目录，即可运行对应基线。中断后重复原命令并添加 --resume；已完成且通过校验的序列跳过，未完成序列从第一帧重新开始。

## 结果与注意事项

- UAV2UAV：RLSM AUC 为 70.32%，SUTrack 为 70.07%，SafeCL 为 70.19%。
- A2A：RLSM AUC 为 80.63%，SUTrack 为 80.62%，差异很小，不能解释为显著提升。
- 上述为历史完整实验结果；本次发布整理另外进行代码一致性、评分一致性和短序列 GPU 验证。
- UAV2UAV 的部分序列图片数与标注数不同，--alignment common 显式复现原实验的共同长度截断；它不能修复中间缺帧。
- 新运行入口统一记录完整 track() 调用耗时，历史 FPS 的计时范围不同，不能直接混为一谈。
- 原创代码许可证尚待作者确认；第三方 MIT 许可已保留。正式公开前请补充 LICENSE 和论文作者/引用信息。

更多细节见 [验证报告](docs/VALIDATION.md) 和 [配置说明](docs/CONFIGURATION.md)。
