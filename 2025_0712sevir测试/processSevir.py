import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置文件锁定
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


class SEVIRVILModelProcessor:
    def __init__(self, data_root, catalog_path=None):
        """
        SEVIR VIL模型输入预处理器

        Args:
            data_root: SEVIR数据根目录
            catalog_path: 目录文件路径
        """
        self.data_root = Path(data_root)
        self.vil_path = self.data_root / "vil"
        self.catalog_path = Path(catalog_path) if catalog_path else None

        # 数据标准化器
        self.scaler = None
        self.scaler_type = None

        # 获取所有VIL文件
        self.vil_files = self.get_vil_files()

        print(f"Found {len(self.vil_files)} VIL files")

    def get_vil_files(self):
        """获取所有VIL文件路径"""
        vil_files = []
        for year_dir in sorted(self.vil_path.glob("*")):
            if year_dir.is_dir():
                h5_files = list(year_dir.glob("*.h5"))
                vil_files.extend(h5_files)
        return sorted(vil_files)

    def preprocess_vil_data(self, vil_data, method='log_normalize'):
        """
        预处理VIL数据

        Args:
            vil_data: 原始VIL数据 (N, H, W, T) 或 (H, W, T)
            method: 预处理方法

        Returns:
            processed_data: 预处理后的数据
        """
        # 处理负值和缺失值
        vil_data = np.nan_to_num(vil_data, nan=0.0, posinf=0.0, neginf=0.0)
        vil_data = np.maximum(vil_data, 0)  # 确保非负

        if method == 'log_normalize':
            # 对数变换 + 归一化
            processed = np.log1p(vil_data)  # log(1 + x)
            processed = self._normalize_data(processed)

        elif method == 'normalize':
            # 直接归一化到[0,1]
            processed = self._normalize_data(vil_data)

        elif method == 'standardize':
            # 标准化 (z-score)
            processed = self._standardize_data(vil_data)

        elif method == 'dbz_conversion':
            # 转换为dBZ并归一化
            processed = 10 * np.log10(vil_data + 1e-8)
            processed = self._normalize_data(processed)

        elif method == 'threshold_normalize':
            # 阈值处理 + 归一化
            threshold = np.percentile(vil_data, 95)
            processed = np.clip(vil_data, 0, threshold)
            processed = processed / threshold

        return processed

    def _normalize_data(self, data):
        """归一化到[0,1]范围"""
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)

    def _standardize_data(self, data):
        """标准化(z-score)"""
        data_mean = np.mean(data)
        data_std = np.std(data)
        return (data - data_mean) / (data_std + 1e-8)

    def fit_scaler(self, sample_data, scaler_type='minmax'):
        """
        拟合数据缩放器

        Args:
            sample_data: 样本数据用于拟合缩放器
            scaler_type: 缩放器类型 ('minmax', 'standard')
        """
        # 将数据展平用于拟合
        flattened_data = sample_data.reshape(-1, 1)

        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()

        self.scaler.fit(flattened_data)
        self.scaler_type = scaler_type

        print(f"Fitted {scaler_type} scaler on {len(flattened_data)} samples")

    def transform_data(self, data):
        """使用已拟合的缩放器转换数据"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")

        original_shape = data.shape
        flattened_data = data.reshape(-1, 1)
        scaled_data = self.scaler.transform(flattened_data)
        return scaled_data.reshape(original_shape)

    def create_sequences(self, vil_data, input_length=12, output_length=12, stride=1):
        """
        创建时间序列

        Args:
            vil_data: VIL数据 (N, H, W, T) 或 (H, W, T)
            input_length: 输入序列长度
            output_length: 输出序列长度
            stride: 步长

        Returns:
            inputs: 输入序列 (N, H, W, input_length)
            targets: 目标序列 (N, H, W, output_length)
        """
        if len(vil_data.shape) == 3:
            vil_data = vil_data[np.newaxis, ...]  # 添加batch维度

        N, H, W, T = vil_data.shape

        inputs = []
        targets = []

        for n in range(N):
            for t in range(0, T - input_length - output_length + 1, stride):
                # 输入序列
                input_seq = vil_data[n, :, :, t:t + input_length]
                # 目标序列
                target_seq = vil_data[n, :, :, t + input_length:t + input_length + output_length]

                inputs.append(input_seq)
                targets.append(target_seq)

        return np.array(inputs), np.array(targets)

    def prepare_model_data(self, max_files=None, input_length=12, output_length=12,
                           preprocessing_method='log_normalize', validation_split=0.2,
                           test_split=0.1, stride=1, subsample_rate=1):
        """
        准备模型训练数据

        Args:
            max_files: 最大文件数量
            input_length: 输入序列长度
            output_length: 输出序列长度
            preprocessing_method: 预处理方法
            validation_split: 验证集比例
            test_split: 测试集比例
            stride: 序列创建步长
            subsample_rate: 子采样率(每n个事件取1个)

        Returns:
            data_dict: 包含训练、验证、测试数据的字典
        """
        print("Loading and preprocessing VIL data...")

        all_inputs = []
        all_targets = []

        files_to_process = self.vil_files[:max_files] if max_files else self.vil_files

        # 首先收集一些样本数据用于拟合缩放器
        sample_data = []
        sample_count = 0

        for file_path in tqdm(files_to_process[:3], desc="Collecting samples for scaler"):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_events = len(f['id'])
                    for i in range(0, n_events, max(1, n_events // 10)):  # 取每个文件的10个样本
                        vil = f['vil'][i]
                        sample_data.append(vil)
                        sample_count += 1
                        if sample_count >= 100:  # 最多100个样本
                            break
                if sample_count >= 100:
                    break
            except Exception as e:
                print(f"Error sampling from {file_path}: {e}")
                continue

        # 拟合缩放器
        if sample_data:
            sample_array = np.array(sample_data)
            processed_samples = self.preprocess_vil_data(sample_array, preprocessing_method)
            self.fit_scaler(processed_samples, scaler_type='minmax')

        # 处理所有文件
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_events = len(f['id'])

                    # 子采样
                    event_indices = range(0, n_events, subsample_rate)

                    for i in event_indices:
                        vil = f['vil'][i]

                        # 预处理
                        processed_vil = self.preprocess_vil_data(vil, preprocessing_method)

                        # 使用缩放器进一步处理
                        if self.scaler:
                            processed_vil = self.transform_data(processed_vil)

                        # 创建序列
                        inputs, targets = self.create_sequences(
                            processed_vil, input_length, output_length, stride
                        )

                        all_inputs.extend(inputs)
                        all_targets.extend(targets)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # 转换为numpy数组
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)

        print(f"Total sequences created: {len(all_inputs)}")
        print(f"Input shape: {all_inputs.shape}")
        print(f"Target shape: {all_targets.shape}")

        # 数据分割
        n_total = len(all_inputs)
        n_test = int(n_total * test_split)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_test - n_val

        # 随机打乱
        indices = np.random.permutation(n_total)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        data_dict = {
            'train_inputs': all_inputs[train_idx],
            'train_targets': all_targets[train_idx],
            'val_inputs': all_inputs[val_idx],
            'val_targets': all_targets[val_idx],
            'test_inputs': all_inputs[test_idx],
            'test_targets': all_targets[test_idx],
            'scaler': self.scaler,
            'preprocessing_method': preprocessing_method,
            'input_length': input_length,
            'output_length': output_length,
            'data_stats': {
                'n_total': n_total,
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test,
                'input_shape': all_inputs.shape[1:],
                'target_shape': all_targets.shape[1:]
            }
        }

        return data_dict

    def save_processed_data(self, data_dict, save_path):
        """保存处理后的数据"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"Processed data saved to: {save_path}")

    def load_processed_data(self, load_path):
        """加载处理后的数据"""
        with open(load_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 恢复缩放器
        self.scaler = data_dict['scaler']
        self.scaler_type = 'minmax' if isinstance(self.scaler, MinMaxScaler) else 'standard'

        print(f"Loaded processed data from: {load_path}")
        return data_dict


class SEVIRVILDataset(Dataset):
    """PyTorch数据集类"""

    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]

        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_seq).permute(2, 0, 1)  # (T, H, W)
        target_tensor = torch.FloatTensor(target_seq).permute(2, 0, 1)  # (T, H, W)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor


def create_data_loaders(data_dict, batch_size=32, num_workers=4):
    """创建PyTorch数据加载器"""

    # 创建数据集
    train_dataset = SEVIRVILDataset(
        data_dict['train_inputs'],
        data_dict['train_targets']
    )

    val_dataset = SEVIRVILDataset(
        data_dict['val_inputs'],
        data_dict['val_targets']
    )

    test_dataset = SEVIRVILDataset(
        data_dict['test_inputs'],
        data_dict['test_targets']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    DATA_ROOT = "/data0/cjl/code/nowcasting/data"
    processor = SEVIRVILModelProcessor(DATA_ROOT)

    # 准备模型数据
    data_dict = processor.prepare_model_data(
        max_files=5,  # 限制文件数量用于测试
        input_length=12,  # 输入12个时间步(1小时)
        output_length=12,  # 预测12个时间步(1小时)
        preprocessing_method='log_normalize',
        validation_split=0.2,
        test_split=0.1,
        stride=6,  # 步长6，减少数据量
        subsample_rate=2  # 每2个事件取1个
    )

    # 打印数据统计信息
    print("\n=== 数据统计 ===")
    stats = data_dict['data_stats']
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 保存处理后的数据
    processor.save_processed_data(data_dict, "processed_sevir_data.pkl")

    # 创建PyTorch数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict,
        batch_size=16,
        num_workers=2
    )

    print(f"\n=== 数据加载器信息 ===")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")

    # 测试数据加载
    for batch_inputs, batch_targets in train_loader:
        print(f"\n=== 批次数据形状 ===")
        print(f"输入形状: {batch_inputs.shape}")  # (batch_size, input_length, H, W)
        print(f"目标形状: {batch_targets.shape}")  # (batch_size, output_length, H, W)
        break

    # 示例：如何使用处理后的数据训练模型
    print("\n=== 模型训练示例 ===")
    print("# 现在你可以使用train_loader来训练你的模型:")
    print("# for epoch in range(num_epochs):")
    print("#     for batch_inputs, batch_targets in train_loader:")
    print("#         # batch_inputs: (batch_size, input_length, H, W)")
    print("#         # batch_targets: (batch_size, output_length, H, W)")
    print("#         # 在这里进行模型训练")
    print("#         pass")