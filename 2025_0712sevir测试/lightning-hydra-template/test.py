#!/usr/bin/env python3
"""
最简单的SEVIR训练脚本 - 绕过所有配置文件
直接运行：CUDA_VISIBLE_DEVICES=1 python simple_train.py
"""

import os
import sys

sys.path.append('src')

import lightning as L
from data.sevir_datamodule import SEVIRDataModule
from models.convlstm_module import VILNowcastingModel


def main():
    # 设置随机种子
    L.seed_everything(42)

    # 数据模块 - 最简配置
    datamodule = SEVIRDataModule(
        data_dir="/data0/cjl/code/nowcasting/2025_0712sevir测试/data",
        vil_data_dir="/data0/cjl/code/nowcasting/2025_0712sevir测试/data/vil",
        catalog_path="/data0/cjl/code/nowcasting/2025_0712sevir测试/data/CATALOG.csv",
        batch_size=1,
        num_workers=1,
        input_frames=5,
        output_frames=3,
        pin_memory=False
    )

    # 模型 - 最小配置
    model = VILNowcastingModel(
        input_channels=1,
        hidden_dims=[16, 16],
        kernel_sizes=[[3, 3], [3, 3]],
        output_frames=3,
        learning_rate=0.001
    )

    # 训练器 - 最简配置
    trainer = L.Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        precision=16,
        fast_dev_run=2,  # 只跑2个batch测试
        enable_checkpointing=False,
        logger=False
    )

    print("🚀 开始训练...")
    print(f"数据路径: /data0/cjl/code/nowcasting/2025_0712sevir测试/data/vil")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1000:.1f}K")

    # 开始训练
    trainer.fit(model, datamodule)

    print("✅ 测试完成！如果看到这行说明代码运行正常。")


if __name__ == "__main__":
    main()