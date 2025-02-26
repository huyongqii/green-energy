import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODEL_CONFIG
from model import NodePredictorNN
from data_loader import DataLoader

class Evaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = DataLoader()
        self.config = MODEL_CONFIG
        self.model = None

    def load_model(self):
        """从固定目录加载预训练模型"""
        model_dir = self.config['model_dir']
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录未找到: {model_dir}")
        
        model_path = os.path.join(model_dir, 'checkpoint.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 创建新的模型实例
        self.model = NodePredictorNN(
            feature_size=7
        ).to(self.device)
        
        # 加载模型，设置 weights_only=True
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功从 {model_path} 加载模型")

    def evaluate(self) -> tuple:
        """评估模型并生成可视化结果"""
        print("开始评估模型")

        # 1. 加载模型
        self.load_model()

        # 2. 获取测试数据加载器
        test_loaders = self.data_loader.create_data_loaders(
            batch_size=self.config['batch_size'],
            split='test'
        )
        test_loader = test_loaders['test']
        
        predictions = []
        targets = []
        
        print("\n=== 模型预测过程调试信息 ===")
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移到设备上
                past_hour = batch['past_hour'].to(self.device)
                cur_datetime = batch['cur_datetime'].to(self.device)
                dayback = batch['dayback'].to(self.device)
                target = batch['target']
                
                # 预测
                output = self.model(past_hour, cur_datetime, dayback)
                
                # 转换回CPU并保存结果
                predictions.extend(output.cpu().numpy())
                targets.extend(target.numpy())
        
        # 转换为numpy数组
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 反向转换缩放
        predictions = self.data_loader.inverse_transform_y(predictions)
        targets = self.data_loader.inverse_transform_y(targets)
        
        # 计算评估指标
        metrics = self._calculate_metrics(predictions, targets)
        
        # 生成可视化
        self._generate_visualizations(predictions, targets)
        
        return metrics, predictions, targets

    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        # 基础指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 计算SMAPE
        smape = np.mean(2.0 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets))) * 100
        
        # 计算不同误差范围内的准确率
        errors = np.abs(predictions - targets)
        accuracy_metrics = {}
        for i in range(20):  # 0-10的误差范围
            accuracy_metrics[f'accuracy_within_{i}'] = (errors <= i).mean() * 100
        
        # 误差分布统计
        error_distribution = {
            'mean_error': np.mean(predictions - targets),
            'mean_abs_error': np.mean(errors),
            'median_error': np.median(predictions - targets),
            'error_std': np.std(predictions - targets),
            'max_error': np.max(errors)
        }
        
        # 合并所有指标
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'smape': smape,
            **accuracy_metrics,
            'error_distribution': error_distribution
        }
        
        return metrics

    def _generate_visualizations(self, predictions, targets, save_dir='visualization_results'):
        """Generate visualization plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Time series comparison plot
        plt.figure(figsize=(15, 6))
        plt.plot(targets, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title('Computing Nodes Prediction Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Computing Nodes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'time_series.png'))
        plt.close()
        
        # 2. Scatter plot of predictions vs actual values
        plt.figure(figsize=(10, 10))
        plt.scatter(targets, predictions, alpha=0.5)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
        plt.close()
        
        print(f"Visualization results saved to: {save_dir}")

def main():
    evaluator = Evaluator()
    metrics, predictions, targets = evaluator.evaluate()

    print("\n=== 测试结果 ===")
    print("\n基础指标:")
    print(f"均方误差 (MSE): {metrics['mse']:.4f}")
    print(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
    print(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
    print(f"R² 分数: {metrics['r2']:.4f}")
    print(f"对称平均绝对百分比误差 (SMAPE): {metrics['smape']:.2f}%")
    
    print("\n准确率指标:")
    for i in range(20):
        print(f"误差在 ±{i} 节点内: {metrics[f'accuracy_within_{i}']:.2f}%")
    
    dist = metrics['error_distribution']
    print("\n误差分布:")
    print(f"平均误差: {dist['mean_error']:.2f} 节点")
    print(f"平均绝对误差: {dist['mean_abs_error']:.2f} 节点")
    print(f"中位数误差: {dist['median_error']:.2f} 节点")
    print(f"误差标准差: {dist['error_std']:.2f} 节点")
    print(f"最大误差: {dist['max_error']:.2f} 节点")

if __name__ == '__main__':
    main()