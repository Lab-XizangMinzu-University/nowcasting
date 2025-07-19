import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict, List
import torchmetrics


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元 - 简化版本"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 拼接输入和隐状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # 分离门控
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class VILNowcastingModel(L.LightningModule):
    """VIL降水预报模型 - 简化但可配置版本"""

    def __init__(
            self,
            input_channels: int = 1,
            hidden_dims: List[int] = [16],
            kernel_sizes: List[List[int]] = [[3, 3]],
            learning_rate: float = 1e-3,
            output_frames: int = 3,
            compile: bool = False,  # 保持兼容性
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # 简化的ConvLSTM层
        self.convlstm_layers = nn.ModuleList()
        input_dim = input_channels

        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            self.convlstm_layers.append(ConvLSTMCell(input_dim, hidden_dim, kernel_size[0]))
            input_dim = hidden_dim

        # 输出层
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=1)

        # 评估指标
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        """
        x: [batch, time, channels, height, width]
        return: [batch, output_time, channels, height, width]
        """
        # 验证输入格式
        if len(x.shape) != 5:
            raise ValueError(f"期望5维输入 [batch, time, channels, height, width]，实际: {x.shape}")

        batch_size, seq_len, channels, height, width = x.shape

        # 初始化隐状态
        layer_states = []
        for layer in self.convlstm_layers:
            h = torch.zeros(batch_size, layer.hidden_dim, height, width, device=x.device)
            c = torch.zeros(batch_size, layer.hidden_dim, height, width, device=x.device)
            layer_states.append((h, c))

        # 编码输入序列
        for t in range(seq_len):
            x_t = x[:, t]
            for i, layer in enumerate(self.convlstm_layers):
                layer_states[i] = layer(x_t, layer_states[i])
                x_t = layer_states[i][0]

        # 解码预测序列
        outputs = []
        for t in range(self.hparams.output_frames):
            # 继续通过ConvLSTM层
            for i, layer in enumerate(self.convlstm_layers):
                if i == 0:
                    # 第一层输入零张量（自回归）
                    x_t = torch.zeros_like(x[:, 0])
                layer_states[i] = layer(x_t, layer_states[i])
                x_t = layer_states[i][0]

            # 输出当前帧
            output = self.output_conv(x_t)
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        inputs = batch['input']  # [batch, time, 1, height, width]
        targets = batch['target']  # [batch, time, 1, height, width]

        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.train_mse(predictions, targets)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/mse', self.train_mse, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']

        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.val_mse(predictions, targets)
        self.val_mae(predictions, targets)

        self.log('val/loss', loss)
        self.log('val/mse', self.val_mse)
        self.log('val/mae', self.val_mae)

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']

        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        self.log('test/loss', loss)
        self.log('test/mse', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }