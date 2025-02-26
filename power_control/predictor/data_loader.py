import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from config import MODEL_CONFIG

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, past_hour_features, cur_datetime_features, dayback_features, targets):
        self.past_hour_features = torch.FloatTensor(past_hour_features)
        self.cur_datetime_features = torch.FloatTensor(cur_datetime_features)
        self.dayback_features = torch.FloatTensor(dayback_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'past_hour': self.past_hour_features[idx],
            'cur_datetime': self.cur_datetime_features[idx],
            'dayback': self.dayback_features[idx],
            'target': self.targets[idx]
        }

class DataLoader:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.dataset_dir = os.path.join(self.config['data_dir'], 'processed_datasets')
        self._load_scalers()
    
    def _load_scalers(self):
        """加载数据缩放器"""
        scaler_path = os.path.join(self.dataset_dir, "dataset_scalers.pkl")
        scalers = joblib.load(scaler_path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        self.dayback_scaler = scalers['dayback_scaler']
    
    def load_data(self, split: str = 'all'):
        """
        加载指定的数据集划分
        
        参数:
            split (str): 要加载的数据集划分('train', 'val', 'test', 'all')
            
        返回:
            dict: 包含加载的数据集的字典
        """
        try:
            data_dict = {}
            splits = [split] if split != 'all' else ['train', 'val', 'test']
            
            for current_split in splits:
                # 加载特征数据
                X_key = f'X_{current_split}'
                X_data = []
                for i in range(3):
                    filename = f"dataset_{X_key}_part{i}.npy"
                    filepath = os.path.join(self.dataset_dir, filename)
                    X_data.append(np.load(filepath))
                data_dict[X_key] = X_data
                
                # 加载目标值
                y_key = f'y_{current_split}'
                filename = f"dataset_{y_key}.npy"
                filepath = os.path.join(self.dataset_dir, filename)
                data_dict[y_key] = np.load(filepath)
            
            print(f"成功加载{split}数据集")
            return data_dict
            
        except Exception as e:
            raise RuntimeError(f"加载数据集时出错: {str(e)}")
    
    def create_data_loaders(self, batch_size: int, split: str = 'all') -> dict:
        """
        创建数据加载器
        
        参数:
            batch_size (int): 批次大小
            split (str): 要加载的数据集划分
            
        返回:
            dict: 包含数据加载器的字典
        """
        data_dict = self.load_data(split)
        loaders = {}
        
        splits = [split] if split != 'all' else ['train', 'val', 'test']
        for current_split in splits:
            dataset = TimeSeriesDataset(
                data_dict[f'X_{current_split}'][0],
                data_dict[f'X_{current_split}'][1],
                data_dict[f'X_{current_split}'][2],
                data_dict[f'y_{current_split}']
            )
            
            loaders[current_split] = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(current_split == 'train'),
                num_workers=4,
                pin_memory=True
            )
        
        return loaders
    
    def inverse_transform_y(self, y_scaled):
        """反转目标值的缩放"""
        return self.target_scaler.inverse_transform(y_scaled) 