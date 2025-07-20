#!/usr/bin/env python3
"""
🚀 純PyTorch模型比較系統
Random Forest vs LSTM vs Transformer - 全部使用PyTorch實現
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# PyTorch相關導入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch可用")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch不可用，將只使用Random Forest")
    # 創建空的替代類
    class nn:
        class Module: pass
        class LSTM: pass
        class Linear: pass
        class Dropout: pass
        class TransformerEncoderLayer: pass
        class TransformerEncoder: pass
        class MSELoss: pass
        class Parameter: pass

# PyTorch模型定義
if TORCH_AVAILABLE:
    class PyTorchLSTM(nn.Module):
        """PyTorch LSTM預測模型"""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(PyTorchLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            # LSTM層
            lstm_out, _ = self.lstm(x)
            
            # 取最後一個時間步的輸出
            last_output = lstm_out[:, -1, :]
            
            # Dropout和全連接層
            output = self.dropout(last_output)
            output = self.fc(output)
            
            return output

    class PyTorchTransformer(nn.Module):
        """PyTorch Transformer預測模型"""
        
        def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, dropout=0.1):
            super(PyTorchTransformer, self).__init__()
            self.d_model = d_model
            
            # 輸入投影
            self.input_projection = nn.Linear(input_size, d_model)
            
            # 位置編碼
            self.pos_embedding = nn.Parameter(torch.randn(1000, d_model))
            
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # 輸出層
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, 1)
            
        def forward(self, x):
            # 輸入投影
            x = self.input_projection(x)
            
            # 添加位置編碼
            seq_len = x.size(1)
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
            x = x + pos_emb
            
            # Transformer處理
            x = self.transformer(x)
            
            # 全局平均池化
            x = x.mean(dim=1)
            
            # 輸出層
            x = self.dropout(x)
            x = self.fc(x)
            
            return x

else:
    # 如果PyTorch不可用，創建空的替代類
    class PyTorchLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available")
    
    class PyTorchTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available")

class PyTorchModelComparison:
    """純PyTorch模型比較系統"""
    
    def __init__(self, pool_name='3pool', sequence_length=24):
        self.pool_name = pool_name
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # 數據相關
        self.data = None
        self.processed_data = None
        
        # 特徵相關
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        print(f"🚀 初始化純PyTorch模型比較系統 - {pool_name}")
        
    def load_data(self, file_path=None):
        """載入歷史數據"""
        
        if file_path is None:
            file_path = f"free_historical_cache/{self.pool_name}_comprehensive_free_historical_365d.csv"
        
        try:
            self.data = pd.read_csv(file_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            print(f"✅ 數據載入成功: {len(self.data)} 條記錄")
            print(f"📅 時間範圍: {self.data['timestamp'].min()} 到 {self.data['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            return False
    
    def create_features(self):
        """特徵工程 - 創建預測特徵"""
        
        print("🔧 開始特徵工程...")
        
        df = self.data.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. 滯後特徵 (Lag Features)
        for lag in [1, 6, 24, 168]:  # 1個點、6個點、24個點、168個點
            df[f'virtual_price_lag_{lag}'] = df['virtual_price'].shift(lag)
        
        # 2. 移動平均特徵 (Moving Average)
        for window in [24, 168, 672]:  # 6小時、7天、28天
            df[f'virtual_price_ma_{window}'] = df['virtual_price'].rolling(window).mean()
        
        # 3. 波動率特徵 (Volatility)
        for window in [24, 168]:
            df[f'virtual_price_std_{window}'] = df['virtual_price'].rolling(window).std()
            df[f'virtual_price_cv_{window}'] = df[f'virtual_price_std_{window}'] / df[f'virtual_price_ma_{window}']
        
        # 4. 價格變化特徵 (Price Change)
        df['virtual_price_change'] = df['virtual_price'].pct_change()
        df['virtual_price_change_abs'] = df['virtual_price_change'].abs()
        
        # 5. 流動性特徵 (Liquidity Features)
        df['total_supply_change'] = df['total_supply'].pct_change()
        df['total_supply_ma_24'] = df['total_supply'].rolling(24).mean()
        
        # 6. 餘額特徵 (Balance Features)
        token_columns = [col for col in df.columns if col.endswith('_balance')]
        if len(token_columns) >= 2:
            df['balance_ratio'] = df[token_columns[0]] / df[token_columns[1]]
            df['balance_imbalance'] = df[token_columns].std(axis=1) / df[token_columns].mean(axis=1)
        
        # 7. 時間特徵 (Time Features)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # 8. 技術指標特徵 (Technical Indicators)
        # RSI (相對強弱指數)
        df['price_change_positive'] = df['virtual_price_change'].apply(lambda x: x if x > 0 else 0)
        df['price_change_negative'] = df['virtual_price_change'].apply(lambda x: -x if x < 0 else 0)
        df['rsi_14'] = 100 - (100 / (1 + df['price_change_positive'].rolling(14).mean() / 
                                         df['price_change_negative'].rolling(14).mean()))
        
        # 9. 目標變數 (Target Variable)
        df['target_24h'] = df['virtual_price'].shift(-24)  # 預測24個點後的價格 (6小時後)
        df['target_return_24h'] = (df['target_24h'] / df['virtual_price'] - 1) * 100  # 收益率%
        
        # 刪除缺失值
        self.processed_data = df.dropna().reset_index(drop=True)
        
        print(f"✅ 特徵工程完成")
        print(f"📊 處理後數據: {len(self.processed_data)} 條記錄")
        
        return self.processed_data
    
    def prepare_data(self):
        """準備不同模型的訓練數據"""
        
        # 選擇特徵欄
        exclude_cols = ['timestamp', 'pool_address', 'pool_name', 'source', 'target_24h', 'target_return_24h', 'virtual_price']
        self.feature_columns = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        X = self.processed_data[self.feature_columns].fillna(0)
        y = self.processed_data['target_return_24h'].fillna(0)
        
        # 時間序列分割 (前80%訓練，後20%測試)
        split_index = int(len(X) * 0.8)
        
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index] 
        self.y_test = y.iloc[split_index:]
        
        print(f"📊 訓練集: {len(self.X_train)} 樣本")
        print(f"📊 測試集: {len(self.X_test)} 樣本")
        
    def create_sequences_for_pytorch(self, X, y, sequence_length):
        """為PyTorch模型創建序列數據"""
        
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def train_random_forest(self):
        """訓練Random Forest模型"""
        
        print("\n🌳 訓練Random Forest模型...")
        
        # 標準化數據
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # 訓練模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, self.y_train)
        
        # 預測
        train_pred = rf_model.predict(X_train_scaled)
        test_pred = rf_model.predict(X_test_scaled)
        
        # 存儲模型和結果
        self.models['Random Forest'] = rf_model
        self.scalers['Random Forest'] = scaler
        self.results['Random Forest'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mae': mean_absolute_error(self.y_train, train_pred),
            'test_mae': mean_absolute_error(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
            'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(self.y_train)) * 100,
            'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(self.y_test)) * 100
        }
        
        print("✅ Random Forest訓練完成")
        
    def train_pytorch_lstm(self):
        """使用PyTorch訓練LSTM模型"""
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，跳過LSTM訓練")
            return
            
        print("\n🔄 訓練PyTorch LSTM模型...")
        
        # 準備序列數據
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # 創建序列
        X_train_seq, y_train_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_train_scaled), self.y_train, self.sequence_length
        )
        X_test_seq, y_test_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_test_scaled), self.y_test, self.sequence_length
        )
        
        # 轉換為PyTorch張量
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.FloatTensor(y_test_seq).view(-1, 1)
        
        # 創建數據載入器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 創建模型
        input_size = X_train_tensor.shape[2]
        lstm_model = PyTorchLSTM(input_size, hidden_size=64, num_layers=2)
        
        # 訓練參數
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        # 訓練模型
        lstm_model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(train_loader):.4f}")
        
        # 預測
        lstm_model.eval()
        with torch.no_grad():
            train_pred = lstm_model(X_train_tensor).numpy().flatten()
            test_pred = lstm_model(X_test_tensor).numpy().flatten()
        
        # 存儲結果
        self.models['LSTM (PyTorch)'] = lstm_model
        self.scalers['LSTM (PyTorch)'] = scaler
        self.results['LSTM (PyTorch)'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mae': mean_absolute_error(y_train_seq, train_pred),
            'test_mae': mean_absolute_error(y_test_seq, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_seq, test_pred)),
            'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(y_train_seq)) * 100,
            'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(y_test_seq)) * 100
        }
        
        print("✅ PyTorch LSTM訓練完成")
    
    def train_pytorch_transformer(self):
        """使用PyTorch訓練Transformer模型"""
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，跳過Transformer訓練")
            return
            
        print("\n🤖 訓練PyTorch Transformer模型...")
        
        # 準備序列數據 (與LSTM相同)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # 創建序列
        X_train_seq, y_train_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_train_scaled), self.y_train, self.sequence_length
        )
        X_test_seq, y_test_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_test_scaled), self.y_test, self.sequence_length
        )
        
        # 轉換為PyTorch張量
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.FloatTensor(y_test_seq).view(-1, 1)
        
        # 創建數據載入器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 創建模型
        input_size = X_train_tensor.shape[2]
        transformer_model = PyTorchTransformer(input_size, d_model=64, nhead=8, num_layers=3)
        
        # 訓練參數
        criterion = nn.MSELoss()
        optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
        
        # 訓練模型
        transformer_model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = transformer_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(train_loader):.4f}")
        
        # 預測
        transformer_model.eval()
        with torch.no_grad():
            train_pred = transformer_model(X_train_tensor).numpy().flatten()
            test_pred = transformer_model(X_test_tensor).numpy().flatten()
        
        # 存儲結果
        self.models['Transformer (PyTorch)'] = transformer_model
        self.scalers['Transformer (PyTorch)'] = scaler
        self.results['Transformer (PyTorch)'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mae': mean_absolute_error(y_train_seq, train_pred),
            'test_mae': mean_absolute_error(y_test_seq, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_seq, test_pred)),
            'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(y_train_seq)) * 100,
            'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(y_test_seq)) * 100
        }
        
        print("✅ PyTorch Transformer訓練完成")
    
    def compare_models(self):
        """比較所有模型的性能"""
        
        print("\n📊 模型性能比較")
        print("=" * 80)
        
        # 創建比較表格
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test MAE': results['test_mae'],
                'Test RMSE': results['test_rmse'],
                'Test Direction Accuracy (%)': results['test_direction_acc'],
                'Train Direction Accuracy (%)': results['train_direction_acc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Direction Accuracy (%)', ascending=False)
        
        print(comparison_df.round(3).to_string(index=False))
        
        return comparison_df
    
    def visualize_predictions(self, last_n_points=200):
        """可視化所有模型的預測結果"""
        
        if not self.results:
            print("❌ 沒有訓練好的模型")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 為每個模型創建子圖
        n_models = len(self.results)
        cols = 2
        rows = (n_models + 1) // 2
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            plt.subplot(rows, cols, idx + 1)
            
            # 獲取測試集預測結果
            test_pred = results['test_pred']
            
            # 根據模型類型調整實際值
            if 'PyTorch' in model_name:
                # 深度學習模型使用序列數據，需要調整
                actual_values = self.y_test.iloc[self.sequence_length:].tail(last_n_points).values
                pred_values = test_pred[-last_n_points:]
            else:
                # Random Forest使用完整測試集
                actual_values = self.y_test.tail(last_n_points).values
                pred_values = test_pred[-last_n_points:]
            
            # 確保長度一致
            min_length = min(len(actual_values), len(pred_values))
            actual_values = actual_values[-min_length:]
            pred_values = pred_values[-min_length:]
            
            # 繪制預測 vs 實際
            plt.plot(actual_values, label='實際收益率', alpha=0.7)
            plt.plot(pred_values, label='預測收益率', alpha=0.7)
            
            plt.title(f'{model_name}\n準確率: {results["test_direction_acc"]:.1f}%')
            plt.ylabel('收益率 (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.pool_name}_pytorch_comparison_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self):
        """繪制性能比較圖表"""
        
        if not self.results:
            print("❌ 沒有訓練好的模型")
            return
            
        # 準備數據
        model_names = list(self.results.keys())
        test_accuracies = [self.results[name]['test_direction_acc'] for name in model_names]
        test_maes = [self.results[name]['test_mae'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 準確率比較
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)]
        bars1 = ax1.bar(model_names, test_accuracies, color=colors)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='隨機水準(50%)')
        ax1.set_title('📊 模型方向準確率比較 (全PyTorch)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('準確率 (%)')
        ax1.set_ylim(40, max(test_accuracies) + 5)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, acc in zip(bars1, test_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # MAE比較
        bars2 = ax2.bar(model_names, test_maes, color=colors)
        ax2.set_title('📈 模型平均絕對誤差比較', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE (%)')
        ax2.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, mae in zip(bars2, test_maes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mae:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.pool_name}_pytorch_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_comparison(self):
        """運行完整的模型比較"""
        
        print("🚀 開始完整的PyTorch模型比較實驗...")
        print("=" * 80)
        
        # 1. 載入和準備數據
        if not self.load_data():
            return
        
        self.create_features()
        self.prepare_data()
        
        # 2. 訓練所有可用模型
        self.train_random_forest()
        
        if TORCH_AVAILABLE:
            self.train_pytorch_lstm()
            self.train_pytorch_transformer()
        
        # 3. 比較模型性能
        comparison_df = self.compare_models()
        
        # 4. 可視化結果
        self.visualize_predictions()
        self.plot_performance_comparison()
        
        # 5. 生成報告
        self.generate_comparison_report(comparison_df)
        
        print("\n🎉 PyTorch模型比較實驗完成！")
        return comparison_df
    
    def generate_comparison_report(self, comparison_df):
        """生成詳細的比較報告"""
        
        report = []
        report.append(f"# 🚀 {self.pool_name} 池子純PyTorch模型比較報告")
        report.append("=" * 60)
        report.append(f"生成時間: {pd.Timestamp.now()}")
        report.append(f"框架: 純PyTorch實現 (Random Forest + PyTorch LSTM + PyTorch Transformer)")
        report.append(f"數據期間: 365天")
        report.append(f"預測目標: 未來6小時Virtual Price收益率")
        report.append("")
        
        # 最佳模型
        if len(comparison_df) > 0:
            best_model = comparison_df.iloc[0]
            report.append("🏆 最佳表現模型:")
            report.append(f"模型: {best_model['Model']}")
            report.append(f"測試準確率: {best_model['Test Direction Accuracy (%)']:.2f}%")
            report.append(f"測試MAE: {best_model['Test MAE']:.4f}%")
            report.append("")
            
            # 模型排名
            report.append("📊 模型排名 (按準確率):")
            for idx, row in comparison_df.iterrows():
                rank = idx + 1
                model = row['Model']
                acc = row['Test Direction Accuracy (%)']
                mae = row['Test MAE']
                report.append(f"{rank}. {model}: {acc:.2f}% (MAE: {mae:.4f})")
            
            report.append("")
            report.append("💡 建議:")
            
            if best_model['Test Direction Accuracy (%)'] > 70:
                report.append("✅ 最佳模型表現優秀，建議用於實際預測")
            elif best_model['Test Direction Accuracy (%)'] > 60:
                report.append("⚖️ 最佳模型表現中等，建議謹慎使用")
            else:
                report.append("⚠️ 所有模型表現一般，建議改進特徵工程")
        
        # 保存報告
        report_content = "\n".join(report)
        with open(f'{self.pool_name}_pytorch_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 比較報告已保存: {self.pool_name}_pytorch_comparison_report.txt")

def demo_pytorch_model_comparison():
    """演示純PyTorch模型比較"""
    
    print("🚀 Curve Virtual Price預測 - 純PyTorch模型比較演示")
    print("=" * 80)
    
    # 創建比較器
    comparator = PyTorchModelComparison(pool_name='3pool', sequence_length=24)
    
    # 運行完整比較
    comparison_df = comparator.run_complete_comparison()
    
    if comparison_df is not None:
        print(f"\n🎯 實驗總結:")
        print(f"最佳模型: {comparison_df.iloc[0]['Model']}")
        print(f"最高準確率: {comparison_df.iloc[0]['Test Direction Accuracy (%)']:.2f}%")
        print(f"框架: 純PyTorch實現 (統一框架)")
        
        # 保存比較結果
        comparison_df.to_csv(f'3pool_pytorch_comparison_results.csv', index=False)
        print("💾 比較結果已保存到CSV文件")

if __name__ == "__main__":
    demo_pytorch_model_comparison() 