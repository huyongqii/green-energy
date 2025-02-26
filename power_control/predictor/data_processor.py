import os
import joblib
import holidays
import pandas as pd
import numpy as np
from config import MODEL_CONFIG
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.dayback_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.cn_holidays = holidays.CN()
        
        self.pst_hour_feature_names = [
            'running_jobs',
            'waiting_jobs',
            'nb_computing',
            'avg_req_cpu_occupancy_rate',
            'avg_nodes_per_job',
            'avg_cpus_per_job',
            'avg_runtime_minutes'
        ]
        
        self.feature_size = len(self.pst_hour_feature_names)
        self.dataset_dir = os.path.join(self.config['data_dir'], 'processed_datasets')
        os.makedirs(self.dataset_dir, exist_ok=True)

    def process_and_save_data(self):
        """处理数据并保存"""
        print("开始处理数据...")
        
        # 加载原始数据
        data = pd.read_csv(self.config['data_path'])
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # 准备时间序列数据
        past_hour_features, cur_datetime_features, dayback_features, target_values = \
            self.prepare_time_series_data(data)
        
        # 确保目标值非负
        target_values = np.maximum(target_values, 0)
        
        # 划分数据集
        train_size = int(len(target_values) * 0.7)
        val_size = int(len(target_values) * 0.15)
        
        # 创建数据字典
        data_dict = {
            'X_train': [
                past_hour_features[:train_size],
                cur_datetime_features[:train_size],
                dayback_features[:train_size]
            ],
            'y_train': target_values[:train_size],
            
            'X_val': [
                past_hour_features[train_size:train_size + val_size],
                cur_datetime_features[train_size:train_size + val_size],
                dayback_features[train_size:train_size + val_size]
            ],
            'y_val': target_values[train_size:train_size + val_size],
            
            'X_test': [
                past_hour_features[train_size + val_size:],
                cur_datetime_features[train_size + val_size:],
                dayback_features[train_size + val_size:]
            ],
            'y_test': target_values[train_size + val_size:]
        }
        
        # 特征缩放
        for split in ['train', 'val', 'test']:
            if split == 'train':
                # 在训练集上拟合并转换
                data_dict[f'X_{split}'][0] = self.feature_scaler.fit_transform(
                    data_dict[f'X_{split}'][0].reshape(-1, self.feature_size)
                ).reshape(data_dict[f'X_{split}'][0].shape)
                
                data_dict[f'X_{split}'][2] = self.dayback_scaler.fit_transform(
                    data_dict[f'X_{split}'][2]
                )
                
                data_dict[f'y_{split}'] = self.target_scaler.fit_transform(
                    data_dict[f'y_{split}']
                )
            else:
                # 在验证集和测试集上只进行转换
                data_dict[f'X_{split}'][0] = self.feature_scaler.transform(
                    data_dict[f'X_{split}'][0].reshape(-1, self.feature_size)
                ).reshape(data_dict[f'X_{split}'][0].shape)
                
                data_dict[f'X_{split}'][2] = self.dayback_scaler.transform(
                    data_dict[f'X_{split}'][2]
                )
                
                data_dict[f'y_{split}'] = self.target_scaler.transform(
                    data_dict[f'y_{split}']
                )
        
        # 保存处理好的数据集
        self.save_processed_data(data_dict)
        print("数据处理完成并已保存")

    def prepare_time_series_data(self, df):
        """准备时间序列数据"""
        # 1. 添加数据验证
        self._validate_input_data(df)
        
        # 2. 处理异常值
        df = self._handle_outliers(df)
        
        lookback = self.config['lookback_minutes']
        forecast_horizon = self.config['forecast_minutes']
        
        past_hour_sequences = []
        cur_datetime_feature_vectors = []
        dayback_feature_vectors = []
        target_values = []
        
        timestamps = pd.to_datetime(df['datetime'])
        
        for i in range(len(df) - lookback - forecast_horizon + 1):
            # 1. 提取历史序列数据
            past_hour_data = df[self.pst_hour_feature_names].iloc[i:(i + lookback)].values
            
            # 2. 获取目标时间点
            target_start = i + lookback
            target_end = target_start + forecast_horizon
            target_value = df['nb_computing'].iloc[target_end-1]
            
            target_time = timestamps[target_start]
            
            # 3. 生成时间特征
            cur_datetime_features = self._create_time_features(
                timestamps[target_start], 
                timestamps[target_end - 1]
            )
            
            # 4. 获取历史模式特征
            dayback_features = self._get_dayback_features(
                df, timestamps, target_time, target_start, 'nb_computing'
            )
            
            past_hour_sequences.append(past_hour_data)
            cur_datetime_feature_vectors.append(cur_datetime_features)
            dayback_feature_vectors.append(dayback_features)
            target_values.append(target_value)
        
        return (np.array(past_hour_sequences),
                np.array(cur_datetime_feature_vectors),
                np.array(dayback_feature_vectors),
                np.array(target_values).reshape(-1, 1))

    def _create_time_features(self, start_time, end_time):
        """创建时间范围的特征向量"""
        is_weekend = float(start_time.dayofweek >= 5)
        is_holiday = float(self.is_holiday(start_time.date()))
        
        hours = pd.date_range(start_time, end_time, freq='1min').hour
        period_counts = np.zeros(6)
        for hour in hours:
            period = self.get_day_period(hour)
            period_counts[period] += 1
        
        main_period = np.argmax(period_counts)
        
        return [
            is_weekend,         # 是否周末 (0/1)
            is_holiday,         # 是否节假日 (0/1)
            main_period / 6.0,  # 主要时间段 (归一化到 0-1)
        ]

    def get_day_period(self, hour):
        """将一天分为不同时段"""
        if 5 <= hour < 9:
            return 0  # 早晨
        elif 9 <= hour < 12:
            return 1  # 上午
        elif 12 <= hour < 14:
            return 2  # 中午
        elif 14 <= hour < 18:
            return 3  # 下午
        elif 18 <= hour < 24:
            return 4  # 晚上
        else:
            return 5  # 深夜

    def is_holiday(self, date):
        """判断是否为节假日"""
        return date in self.cn_holidays

    def _get_dayback_features(self, df, timestamps, target_time, current_idx, target_col):
        """获取历史模式特征"""
        pattern_features = []
        window_minutes = 30
        
        for days_back in [1, 3, 5, 7]:
            minutes_back = days_back * 24 * 60
            historical_center_idx = current_idx - minutes_back
            
            if historical_center_idx >= window_minutes and historical_center_idx + window_minutes < len(df):
                # 获取历史时间段的数据
                historical_window = df[target_col].iloc[
                    historical_center_idx - window_minutes:
                    historical_center_idx + window_minutes
                ]
                
                # 增加更多统计特征
                pattern_features.extend([
                    float(historical_window.min()),     # 最小值
                    float(historical_window.max()),     # 最大值
                ])
            else:
                # 使用当前值填充
                current_value = float(df[target_col].iloc[current_idx])
                pattern_features.extend([current_value] * 2)
        
        return np.array(pattern_features, dtype=np.float32)

    def _validate_input_data(self, df):
        """验证输入数据的完整性和有效性"""
        # 检查必要的列是否存在
        required_columns = self.pst_hour_feature_names + ['datetime', 'nb_computing']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 检查时间戳的连续性
        timestamps = pd.to_datetime(df['datetime'])
        time_diff = timestamps.diff().dropna()
        if not (time_diff == pd.Timedelta(minutes=1)).all():
            print("警告: 时间序列不连续，可能影响预测效果")
        
        # 检查数值的有效性
        for col in self.pst_hour_feature_names + ['nb_computing']:
            if df[col].isnull().any():
                print(f"警告: 列 {col} 存在空值")
            if (df[col] < 0).any():
                print(f"警告: 列 {col} 存在负值")

    def _handle_outliers(self, df):
        """处理异常值"""
        df_clean = df.copy()
        
        for col in self.pst_hour_feature_names:
            if col in ['running_jobs', 'waiting_jobs', 'nb_computing']:
                # 对于作业数量，只检查负值
                outliers = df[col] < 0
                if outliers.any():
                    print(f"列 {col} 发现 {outliers.sum()} 个负值")
                    # 将负值设为0
                    df_clean.loc[outliers, col] = 0
                    
            # elif col == 'avg_req_cpu_occupancy_rate':
            #     # 对于利用率，检查是否在合理范围内
            #     outliers = (df[col] < 0) | (df[col] > 100)
            #     if outliers.any():
            #         print(f"列 {col} 发现 {outliers.sum()} 个异常值")
            #         # 使用移动平均替换异常值
            #         window_size = 5
            #         moving_avg = df[col].rolling(
            #             window=window_size,
            #             center=True,
            #             min_periods=1
            #         ).mean()
            #         df_clean.loc[outliers, col] = moving_avg[outliers]
            
            print(f"列 {col} 的范围: [{df_clean[col].min()}, {df_clean[col].max()}]")
        
        return df_clean

    def save_processed_data(self, data_dict: dict, prefix: str = 'dataset'):
        """保存处理好的数据集和缩放器"""
        try:
            # 保存数据集
            for key, data in data_dict.items():
                if isinstance(data, list):
                    for i, feature_data in enumerate(data):
                        filename = f"{prefix}_{key}_part{i}.npy"
                        filepath = os.path.join(self.dataset_dir, filename)
                        np.save(filepath, feature_data)
                else:
                    filename = f"{prefix}_{key}.npy"
                    filepath = os.path.join(self.dataset_dir, filename)
                    np.save(filepath, data)
            
            # 保存缩放器
            scalers = {
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'dayback_scaler': self.dayback_scaler
            }
            scaler_path = os.path.join(self.dataset_dir, f"{prefix}_scalers.pkl")
            joblib.dump(scalers, scaler_path)
            
            print(f"数据集和缩放器已保存到: {self.dataset_dir}")
            
        except Exception as e:
            raise RuntimeError(f"保存数据集时出错: {str(e)}")

def main():
    """主函数用于数据处理"""
    processor = DataProcessor()
    processor.process_and_save_data()

if __name__ == '__main__':
    main()