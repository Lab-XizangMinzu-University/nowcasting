#!/usr/bin/env python3
"""
最简单的SEVIR训练脚本 - 完全自定义模型，避免所有问题
直接运行：CUDA_VISIBLE_DEVICES=1 python simple_train.py
"""

import os
import sys

sys.path.append('src')

import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from data.sevir_datamodule import SEVIRDataModule


class SimpleConvLSTMCell(nn.Module):
    """简单的ConvLSTM单元"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class SimpleConvLSTM(L.LightningModule):
    """最简单的ConvLSTM模型"""

    def __init__(self, input_channels=1, hidden_dim=16, output_frames=3):
        super().__init__()
        self.save_hyperparameters()

        self.convlstm = SimpleConvLSTMCell(input_channels, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, input_channels, 1)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        # x: [batch, time, channels, height, width]
        batch_size, seq_len, channels, height, width = x.shape

        # 初始化状态
        h = torch.zeros(batch_size, self.hparams.hidden_dim, height, width, device=x.device)
        c = torch.zeros(batch_size, self.hparams.hidden_dim, height, width, device=x.device)

        # 编码输入序列
        for t in range(seq_len):
            h, c = self.convlstm(x[:, t], (h, c))

        # 解码输出序列
        outputs = []
        for t in range(self.hparams.output_frames):
            h, c = self.convlstm(torch.zeros_like(x[:, 0]), (h, c))
            output = self.output_conv(h)
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.train_mse(predictions, targets)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/mse', self.train_mse)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['input'], batch['target']
        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.val_mse(predictions, targets)
        self.log('val/loss', loss)
        self.log('val/mse', self.val_mse)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    # 设置随机种子
    L.seed_everything(42)

    # 数据模块 - 最简配置
    datamodule = SEVIRDataModule(
        data_dir="/data0/cjl/code/nowcasting/data",
        vil_data_dir="/data0/cjl/code/nowcasting/data/vil",
        catalog_path="/data0/cjl/code/nowcasting/data/CATALOG.csv",
        batch_size=1,
        num_workers=1,
        input_frames=5,
        output_frames=3,
        pin_memory=False
    )

    # 模型 - 完全自定义，没有任何问题
    model = SimpleConvLSTM(
        input_channels=1,
        hidden_dim=16,
        output_frames=3
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
    print(f"数据路径: /data0/cjl/code/nowcasting/data/vil")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1000:.1f}K")

    # 开始训练
    trainer.fit(model, datamodule)

    print("✅ 测试完成！SEVIR ConvLSTM训练流程验证成功！")


if __name__ == "__main__":
    main()