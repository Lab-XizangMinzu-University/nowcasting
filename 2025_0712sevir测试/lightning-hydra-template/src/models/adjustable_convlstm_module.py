# src/models/adjustable_convlstm_module.py
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import torchmetrics
import math


class EnhancedConvLSTMCell(nn.Module):
    """增强版ConvLSTM单元，支持多种配置选项"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int = 3,
            bias: bool = True,
            dropout: float = 0.0,
            use_attention: bool = False,
            use_residual: bool = False,
            activation: str = 'tanh',
            normalization: str = 'none'  # 'none', 'batch', 'layer', 'group'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.use_attention = use_attention
        self.use_residual = use_residual

        # 主要的卷积层（输入门、遗忘门、输出门、候选值）
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # 归一化层
        if normalization == 'batch':
            self.norm = nn.BatchNorm2d(4 * self.hidden_dim)
        elif normalization == 'layer':
            self.norm = nn.GroupNorm(1, 4 * self.hidden_dim)
        elif normalization == 'group':
            self.norm = nn.GroupNorm(4, 4 * self.hidden_dim)
        else:
            self.norm = nn.Identity()

        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = torch.tanh

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 4, 1, 1),
                nn.Sigmoid()
            )

        # 残差连接的投影层（如果维度不匹配）
        if use_residual and input_dim != hidden_dim:
            self.residual_proj = nn.Conv2d(input_dim, hidden_dim, 1)
        else:
            self.residual_proj = None

    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = cur_state

        # 拼接输入和隐状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        combined_conv = self.norm(combined_conv)

        # 分离四个门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 计算门控值
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = self.activation(cc_g)  # 候选值

        # 更新细胞状态
        c_next = f * c_cur + i * g

        # 计算隐状态
        h_next = o * self.activation(c_next)

        # 应用注意力机制
        if self.use_attention:
            attention_weights = self.attention(h_next)
            h_next = h_next * attention_weights

        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(input_tensor)
            else:
                residual = input_tensor
            h_next = h_next + residual

        # 应用dropout
        h_next = self.dropout(h_next)

        return h_next, c_next


class AdjustableConvLSTM(L.LightningModule):
    """可高度调整的ConvLSTM模型"""

    def __init__(
            self,
            # 基础架构参数
            input_channels: int = 1,
            hidden_dims: List[int] = [32, 64],
            kernel_sizes: List[int] = [3, 3],
            num_layers: Optional[int] = None,  # 如果指定，会覆盖hidden_dims

            # 序列参数
            input_frames: int = 5,
            output_frames: int = 3,

            # 正则化参数
            dropout: float = 0.1,
            normalization: str = 'group',  # 'none', 'batch', 'layer', 'group'

            # 增强功能
            use_attention: bool = False,
            use_residual: bool = True,
            use_skip_connections: bool = True,
            activation: str = 'tanh',

            # 解码策略
            decoder_type: str = 'autoregressive',  # 'autoregressive', 'teacher_forcing', 'scheduled'
            teacher_forcing_ratio: float = 0.5,

            # 损失函数
            loss_type: str = 'mse',  # 'mse', 'mae', 'huber', 'combined'
            loss_weights: List[float] = [1.0],  # 多帧预测的权重

            # 优化参数
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            compile: bool = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # 处理层数配置
        if num_layers is not None:
            # 如果指定了层数，创建统一的隐藏维度
            hidden_dims = [hidden_dims[0] * (2 ** i) for i in range(num_layers)]
            kernel_sizes = [kernel_sizes[0]] * num_layers

        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes

        # 构建ConvLSTM层
        self.convlstm_layers = nn.ModuleList()
        input_dim = input_channels

        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            # 逐层递减dropout
            layer_dropout = dropout * (1 - i / len(hidden_dims))

            self.convlstm_layers.append(
                EnhancedConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=layer_dropout,
                    use_attention=use_attention and i == len(hidden_dims) - 1,  # 只在最后一层使用注意力
                    use_residual=use_residual,
                    activation=activation,
                    normalization=normalization
                )
            )
            input_dim = hidden_dim

        # 跳跃连接
        if use_skip_connections and len(hidden_dims) > 1:
            total_hidden = sum(hidden_dims)
            self.skip_projection = nn.Conv2d(total_hidden, hidden_dims[-1], 1)
        else:
            self.skip_projection = None

        # 输出层
        self.output_layers = self._build_output_layers(hidden_dims[-1], input_channels)

        # 位置编码（用于时序信息）
        self.pos_encoding = nn.Parameter(
            torch.randn(1, output_frames, 1, 1, 1) * 0.1,
            requires_grad=True
        )

        # 评估指标
        self._setup_metrics()

    def _build_output_layers(self, hidden_dim: int, output_channels: int) -> nn.Module:
        """构建输出层"""
        return nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, output_channels, 1),
            nn.Sigmoid()  # 假设输出范围在[0,1]
        )

    def _setup_metrics(self):
        """设置评估指标"""
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, x: torch.Tensor, return_all_frames: bool = False) -> torch.Tensor:
        """
        x: [batch, time, channels, height, width]
        return: [batch, output_time, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = x.shape

        # 初始化隐状态
        layer_states = []
        for layer in self.convlstm_layers:
            h = torch.zeros(batch_size, layer.hidden_dim, height, width, device=x.device)
            c = torch.zeros(batch_size, layer.hidden_dim, height, width, device=x.device)
            layer_states.append((h, c))

        # 编码阶段 - 处理输入序列
        skip_features = [] if self.skip_projection is not None else None

        for t in range(seq_len):
            x_t = x[:, t]
            layer_outputs = []

            for i, layer in enumerate(self.convlstm_layers):
                layer_states[i] = layer(x_t, layer_states[i])
                x_t = layer_states[i][0]

                if skip_features is not None:
                    layer_outputs.append(x_t)

            # 收集跳跃连接特征
            if skip_features is not None and t == seq_len - 1:  # 最后一个时间步
                skip_features = layer_outputs

        # 解码阶段 - 生成预测序列
        outputs = []
        last_input = x[:, -1]  # 使用最后一帧作为初始输入

        for t in range(self.hparams.output_frames):
            # 选择输入策略
            if self.hparams.decoder_type == 'autoregressive':
                x_t = last_input
            elif self.hparams.decoder_type == 'teacher_forcing' and self.training:
                # 在训练时使用teacher forcing
                use_teacher_forcing = torch.rand(1) < self.hparams.teacher_forcing_ratio
                if use_teacher_forcing and hasattr(self, '_target_sequence'):
                    x_t = self._target_sequence[:, min(t, self._target_sequence.size(1) - 1)]
                else:
                    x_t = last_input
            else:
                x_t = last_input

            # 通过ConvLSTM层
            layer_outputs = []
            for i, layer in enumerate(self.convlstm_layers):
                layer_states[i] = layer(x_t, layer_states[i])
                x_t = layer_states[i][0]
                layer_outputs.append(x_t)

            # 应用跳跃连接
            if self.skip_projection is not None:
                # 合并所有层的特征
                combined_features = torch.cat(layer_outputs, dim=1)
                x_t = self.skip_projection(combined_features) + x_t

            # 生成输出
            output = self.output_layers(x_t)

            # 添加位置编码
            output = output + self.pos_encoding[0, t]

            outputs.append(output)
            last_input = output  # 自回归

        result = torch.stack(outputs, dim=1)

        if return_all_frames:
            return result, layer_outputs
        return result

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失函数"""
        if self.hparams.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.hparams.loss_type == 'mae':
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif self.hparams.loss_type == 'huber':
            loss = F.huber_loss(predictions, targets, reduction='none')
        elif self.hparams.loss_type == 'combined':
            mse_loss = F.mse_loss(predictions, targets, reduction='none')
            mae_loss = F.l1_loss(predictions, targets, reduction='none')
            loss = 0.7 * mse_loss + 0.3 * mae_loss
        else:
            loss = F.mse_loss(predictions, targets, reduction='none')

        # 应用时间权重
        if len(self.hparams.loss_weights) == predictions.size(1):
            weights = torch.tensor(self.hparams.loss_weights, device=predictions.device)
            weights = weights.view(1, -1, 1, 1, 1)
            loss = loss * weights

        return loss.mean()

    def training_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']

        # 存储目标序列用于teacher forcing
        if self.hparams.decoder_type == 'teacher_forcing':
            self._target_sequence = targets

        predictions = self(inputs)
        loss = self._compute_loss(predictions, targets)

        # 添加正则化
        if self.hparams.weight_decay > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.hparams.weight_decay
            loss = loss + l2_loss

        # 更新指标
        self.train_mse(predictions, targets)
        self.train_loss(loss)

        # 记录日志
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']

        predictions = self(inputs)
        loss = self._compute_loss(predictions, targets)

        # 更新指标
        self.val_mse(predictions, targets)
        self.val_mae(predictions, targets)
        self.val_loss(loss)

        # SSIM (只对第一帧计算，避免内存问题)
        if batch_idx < 10:  # 只在前几个batch计算SSIM
            self.val_ssim(predictions[:, 0], targets[:, 0])

        # 记录日志
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mse', self.val_mse, on_step=False, on_epoch=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True)
        if batch_idx < 10:
            self.log('val/ssim', self.val_ssim, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        inputs = batch['input']
        targets = batch['target']

        predictions = self(inputs)
        loss = self._compute_loss(predictions, targets)  # 使用模型的损失函数

        # 计算额外的测试指标
        test_mse = F.mse_loss(predictions, targets)  # 添加 F. 前缀
        test_mae = F.l1_loss(predictions, targets)  # 添加 F. 前缀

        # 记录测试日志
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mse', test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mae', test_mae, on_step=False, on_epoch=True)

        return loss