#!/usr/bin/env python3
"""
🚀 純PyTorch + 量子機器學習模型比較系統
Random Forest vs LSTM vs Transformer vs QNN - 包含量子神經網絡
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# XGBoost導入
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost可用")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost不可用，將跳過XGBoost模型")

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

# PennyLane量子機器學習導入
try:
    import pennylane as qml
    import pennylane.numpy as pnp  # 只給PennyLane使用的numpy
    QML_AVAILABLE = True
    print("✅ PennyLane量子機器學習庫可用")
except ImportError:
    QML_AVAILABLE = False
    print("❌ PennyLane不可用，將跳過QNN模型")
    qml = None
    pnp = None

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

    class PyTorchQNN(nn.Module):
        """PyTorch + PennyLane 量子神經網絡模型"""
        
        def __init__(self, input_size, n_qubits=4, n_layers=2):
            super(PyTorchQNN, self).__init__()
            self.input_size = input_size
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            
            # 檢查PennyLane是否可用
            if not QML_AVAILABLE:
                raise ImportError("PennyLane not available for QNN")
            
            # 創建量子設備
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            # 使用PCA降維替代神經網絡預處理
            # input_size 現在應該等於 n_qubits * 2 (PCA降維後的特徵數)
            assert input_size == n_qubits * 2, f"PCA降維後特徵數應為{n_qubits * 2}，但得到{input_size}"
            print(f"✅ QNN直接使用PCA降維後的{input_size}個特徵")
            
            # 量子電路
            @qml.qnode(self.dev, interface="torch")
            def quantum_circuit(inputs, weights):
                # 改進的數據編碼：使用更多輸入信息
                # 將inputs分成兩組，分別用於角度編碼和相位編碼
                n_inputs = len(inputs)
                angle_inputs = inputs[:n_inputs//2]  # 前半部分用於角度
                phase_inputs = inputs[n_inputs//2:]  # 後半部分用於相位
                
                # 角度編碼：RY門編碼主要特徵
                for i in range(self.n_qubits):
                    qml.RY(angle_inputs[i % len(angle_inputs)], wires=i)
                
                # 相位編碼：RZ門編碼次要特徵
                for i in range(self.n_qubits):
                    qml.RZ(phase_inputs[i % len(phase_inputs)], wires=i)
                
                # 變分量子電路
                for layer in range(self.n_layers):
                    # 單量子比特旋轉
                    for i in range(self.n_qubits):
                        qml.RX(weights[layer, i, 0], wires=i)
                        qml.RY(weights[layer, i, 1], wires=i)
                        qml.RZ(weights[layer, i, 2], wires=i)
                    
                    # 糾纏層：更豐富的糾纏模式
                    # 線性糾纏
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    
                    # 環形糾纏 (如果有3個以上量子比特)
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
                    
                    # 如果是多層，添加額外的糾纏
                    if layer < self.n_layers - 1 and self.n_qubits >= 4:
                        # 交叉糾纏
                        qml.CNOT(wires=[0, 2])
                        qml.CNOT(wires=[1, 3])
                
                # 測量所有量子比特
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            self.quantum_circuit = quantum_circuit
            
            # 量子權重參數
            self.q_weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.1
            )
            
            # 改進的經典後處理層
            self.post_net = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),  # 先擴展
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(n_qubits * 2, n_qubits),  # 再收縮
                nn.ReLU(),
                nn.Linear(n_qubits, 1)  # 最終輸出
            )
            
        def forward(self, x):
            batch_size = x.size(0)
            
            # 確保輸入是Float32類型
            x = x.float()
            
            # 如果是序列數據，取最後一個時間步
            if len(x.shape) == 3:
                x = x[:, -1, :]  # (batch_size, seq_len, features) -> (batch_size, features)
            
            # 直接使用PCA降維後的特徵，無需額外預處理
            pca_features = x  # x已經是PCA降維後的特徵
            
            # 量子電路處理
            quantum_results = []
            for i in range(batch_size):
                # 重塑權重參數用於量子電路，確保Float32類型
                weights = self.q_weights.reshape(self.n_layers, self.n_qubits, 3).float()
                
                # 量子電路前向傳播，直接使用PCA特徵
                q_out = self.quantum_circuit(pca_features[i].float(), weights)
                
                # 將量子電路輸出轉換為PyTorch張量並確保Float32
                if isinstance(q_out, (list, tuple)):
                    q_out_tensor = torch.stack([torch.as_tensor(val, dtype=torch.float32) for val in q_out])
                else:
                    q_out_tensor = torch.as_tensor(q_out, dtype=torch.float32)
                
                quantum_results.append(q_out_tensor)
            
            quantum_out = torch.stack(quantum_results).float()
            
            # 經典後處理
            output = self.post_net(quantum_out)
            
            return output

    class PyTorchQSVM_QNN(nn.Module):
        """QSVM-QNN混合模型：量子支持向量機 + 量子神經網絡"""
        
        def __init__(self, input_size=4, n_qubits=4, n_layers=2, qsvm_features=4):
            super(PyTorchQSVM_QNN, self).__init__()
            self.input_size = input_size  # PCA降維後的4個特徵
            self.n_qubits = n_qubits  
            self.n_layers = n_layers
            self.qsvm_features = qsvm_features  # QSVM輸出特徵數，等於input_size
            
            # 檢查PennyLane是否可用
            if not QML_AVAILABLE:
                raise ImportError("PennyLane not available for QSVM-QNN")
                
            print(f"🌌 初始化QSVM-QNN混合模型：{input_size}個PCA特徵 → QSVM({input_size}量子比特) → QNN({n_qubits}量子比特)")
            
            # QSVM量子設備（使用input_size個量子比特處理PCA特徵）
            self.qsvm_dev = qml.device("default.qubit", wires=input_size)
            
            # QNN量子設備  
            self.qnn_dev = qml.device("default.qubit", wires=n_qubits)
            
            # QSVM量子特徵映射電路
            @qml.qnode(self.qsvm_dev, interface="torch")
            def qsvm_feature_map(x):
                # 數據編碼到量子態
                for i in range(len(x)):
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i]**2, wires=i)  # 非線性特徵映射
                
                # 糾纏層以捕獲特徵關聯
                for i in range(len(x)-1):
                    qml.CNOT(wires=[i, i+1])
                if len(x) > 2:
                    qml.CNOT(wires=[len(x)-1, 0])
                    
                # 測量所有量子比特
                return [qml.expval(qml.PauliZ(i)) for i in range(len(x))]
            
            self.qsvm_feature_map = qsvm_feature_map
            
            # QNN部分（接收QSVM處理後的特徵）
            @qml.qnode(self.qnn_dev, interface="torch") 
            def qnn_circuit(qsvm_features, weights):
                # 將QSVM特徵編碼到QNN
                for i in range(self.n_qubits):
                    qml.RY(qsvm_features[i % len(qsvm_features)], wires=i)
                
                # 變分量子電路
                for layer in range(self.n_layers):
                    # 旋轉門
                    for i in range(self.n_qubits):
                        qml.RX(weights[layer, i, 0], wires=i)
                        qml.RY(weights[layer, i, 1], wires=i)
                        qml.RZ(weights[layer, i, 2], wires=i)
                    
                    # 糾纏層
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
                
                # 測量
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            self.qnn_circuit = qnn_circuit
            
            # QNN可訓練參數
            self.qnn_weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.1
            )
            
            # 最終輸出層
            self.output_net = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(n_qubits * 2, 1)
            )
            
        def forward(self, x):
            batch_size = x.size(0)
            
            # 確保輸入是Float32類型
            x = x.float()
            
            # 如果是序列數據，取最後一個時間步
            if len(x.shape) == 3:
                x = x[:, -1, :]  # (batch_size, seq_len, features) -> (batch_size, features)
            
            # 第一階段：QSVM特徵映射
            qsvm_results = []
            for i in range(batch_size):
                # QSVM處理
                qsvm_out = self.qsvm_feature_map(x[i])
                if isinstance(qsvm_out, (list, tuple)):
                    qsvm_tensor = torch.stack([torch.as_tensor(val, dtype=torch.float32) for val in qsvm_out])
                else:
                    qsvm_tensor = torch.as_tensor(qsvm_out, dtype=torch.float32)
                qsvm_results.append(qsvm_tensor)
            
            qsvm_features = torch.stack(qsvm_results).float()
            
            # 第二階段：QNN處理QSVM輸出
            qnn_results = []
            for i in range(batch_size):
                weights = self.qnn_weights.reshape(self.n_layers, self.n_qubits, 3).float()
                qnn_out = self.qnn_circuit(qsvm_features[i], weights)
                
                if isinstance(qnn_out, (list, tuple)):
                    qnn_tensor = torch.stack([torch.as_tensor(val, dtype=torch.float32) for val in qnn_out])
                else:
                    qnn_tensor = torch.as_tensor(qnn_out, dtype=torch.float32)
                qnn_results.append(qnn_tensor)
            
            qnn_output = torch.stack(qnn_results).float()
            
            # 最終輸出
            output = self.output_net(qnn_output)
            
            return output

else:
    # 如果PyTorch不可用，創建空的替代類
    class PyTorchLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available")
    
    class PyTorchTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available")
    
    class PyTorchQNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch or PennyLane not available")
            
    class PyTorchQSVM_QNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch or PennyLane not available")

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
        
    def train_xgboost(self):
        """訓練XGBoost模型"""
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost不可用，跳過XGBoost訓練")
            return
            
        print("\n🚀 訓練XGBoost模型...")
        
        # 標準化數據
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # 創建XGBoost模型 (新版本語法)
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,           # 更多樹以提高性能
            max_depth=6,                # 適中深度避免過擬合
            learning_rate=0.1,          # 學習率
            subsample=0.8,              # 行採樣比例
            colsample_bytree=0.8,       # 特徵採樣比例
            reg_alpha=0.1,              # L1正則化
            reg_lambda=1.0,             # L2正則化
            random_state=42,
            n_jobs=-1,                  # 使用所有CPU核心
            tree_method='hist',         # 更快的樹構建方法
            objective='reg:squarederror' # 回歸目標
        )
        
        # 訓練模型 (簡化版本，無早停)
        xgb_model.fit(
            X_train_scaled, self.y_train,
            verbose=False              # 不顯示詳細訓練過程
        )
        
        # 預測
        train_pred = xgb_model.predict(X_train_scaled)
        test_pred = xgb_model.predict(X_test_scaled)
        
        # 存儲模型和結果
        self.models['XGBoost'] = xgb_model
        self.scalers['XGBoost'] = scaler
        self.results['XGBoost'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mae': mean_absolute_error(self.y_train, train_pred),
            'test_mae': mean_absolute_error(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
            'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(self.y_train)) * 100,
            'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(self.y_test)) * 100
        }
        
        # 顯示特徵重要性前5名
        try:
            feature_importance = xgb_model.feature_importances_
            top_features = np.argsort(feature_importance)[-5:][::-1]
            print(f"📊 XGBoost前5重要特徵: {[f'特徵{i}' for i in top_features]}")
            print(f"📈 對應重要性分數: {[f'{feature_importance[i]:.3f}' for i in top_features]}")
        except Exception as e:
            print(f"⚠️ 特徵重要性顯示失敗: {e}")
        
        print("✅ XGBoost訓練完成")
        
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
    
    def train_pytorch_qnn(self):
        """使用PyTorch + PennyLane訓練量子神經網絡模型"""
        
        if not TORCH_AVAILABLE or not QML_AVAILABLE:
            print("❌ PyTorch或PennyLane不可用，跳過QNN訓練")
            return
            
        print("\n🌌 訓練PyTorch量子神經網絡(QNN)模型...")
        
        # 準備數據 - 使用PCA降維替代神經網絡預處理
        print("🔧 使用PCA降維進行特徵預處理...")
        
        # 標準化原始特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # PCA降維到適合4個量子比特的特徵數 (8個特徵用於雙重編碼)
        target_features = 4 * 2  # 4個量子比特 × 2 (角度+相位編碼)
        pca = PCA(n_components=target_features)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # 顯示PCA信息
        explained_variance_ratio = pca.explained_variance_ratio_
        total_variance = sum(explained_variance_ratio)
        print(f"📊 PCA降維: {self.X_train.shape[1]}特徵 → {target_features}特徵")
        print(f"📈 保留方差比例: {total_variance:.3f} ({total_variance*100:.1f}%)")
        print(f"🔝 前3個主成分方差比例: {explained_variance_ratio[:3]}")
        
        # 創建序列數據（用於與其他模型保持一致）
        X_train_seq, y_train_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_train_pca), self.y_train, self.sequence_length
        )
        X_test_seq, y_test_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_test_pca), self.y_test, self.sequence_length
        )
        
        # 轉換為PyTorch張量，確保Float32類型
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)
        
        # 創建數據載入器 - 極小批次以提高量子計算穩定性
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 單樣本批次，最大穩定性
        
        # 創建改進的QNN模型 - 更好的特徵編碼
        input_size = X_train_tensor.shape[2]
        qnn_model = PyTorchQNN(input_size, n_qubits=4, n_layers=2)  # 增加到4量子比特，2層
        
        # 訓練參數 (量子模型使用更保守的設置)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(qnn_model.parameters(), lr=0.005)  # 調整學習率
        
        # 訓練模型
        qnn_model.train()
        print("🌌 訓練改進的QNN: 4量子比特 + 2層電路，更強的特徵編碼能力...")
        
        successful_epochs = 0
        for epoch in range(100):  # 減少epoch數量
            total_loss = 0
            batch_count = 0
            epoch_success = True
            
            for batch_X, batch_y in train_loader:
                try:
                    optimizer.zero_grad()
                    outputs = qnn_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 檢查loss是否有效
                    if torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        batch_count += 1
                    else:
                        print(f"⚠️ Epoch {epoch+1}: 跳過無效loss")
                        continue
                        
                except Exception as e:
                    print(f"⚠️ Epoch {epoch+1}: {str(e)[:50]}...")
                    epoch_success = False
                    break
            
            if epoch_success and batch_count > 0:
                successful_epochs += 1
                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.4f}, 成功批次: {batch_count}")
            
            # 如果連續失敗，提前終止
            if epoch > 10 and successful_epochs < epoch * 0.3:
                print(f"⚠️ 量子訓練不穩定，提前終止於Epoch {epoch+1}")
                break
        
        # 預測
        qnn_model.eval()
        try:
            with torch.no_grad():
                train_pred = qnn_model(X_train_tensor).numpy().flatten()
                test_pred = qnn_model(X_test_tensor).numpy().flatten()
            
            # 存儲結果（包含PCA變換器）
            self.models['QNN (PyTorch+PennyLane)'] = qnn_model
            self.scalers['QNN (PyTorch+PennyLane)'] = {'scaler': scaler, 'pca': pca}
            self.results['QNN (PyTorch+PennyLane)'] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_mae': mean_absolute_error(y_train_seq, train_pred),
                'test_mae': mean_absolute_error(y_test_seq, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test_seq, test_pred)),
                'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(y_train_seq)) * 100,
                'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(y_test_seq)) * 100
            }
            
            print("✅ PyTorch量子神經網絡訓練完成")
            
        except Exception as e:
            print(f"❌ QNN預測階段出錯: {e}")
            print("⚠️ QNN模型可能需要更多調優")
    
    def train_pytorch_qsvmqnn(self):
        """使用PyTorch + PennyLane訓練QSVM-QNN混合模型"""
        
        if not TORCH_AVAILABLE or not QML_AVAILABLE:
            print("❌ PyTorch或PennyLane不可用，跳過QSVM-QNN訓練")
            return
            
        print("\n🌌 訓練PyTorch QSVM-QNN混合模型...")
        
        # 準備數據 - 使用PCA降維替代神經網絡預處理
        print("🔧 使用PCA降維進行特徵預處理...")
        
        # 標準化原始特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # PCA降維到4個特徵（用戶要求）
        target_features = 8  # 降維到4個特徵進入QSVM
        pca = PCA(n_components=target_features)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # 顯示PCA信息
        explained_variance_ratio = pca.explained_variance_ratio_
        total_variance = sum(explained_variance_ratio)
        print(f"📊 PCA降維: {self.X_train.shape[1]}特徵 → {target_features}特徵 (進入QSVM)")
        print(f"📈 保留方差比例: {total_variance:.3f} ({total_variance*100:.1f}%)")
        print(f"🔝 主成分方差比例: {explained_variance_ratio}")
        
        # 創建序列數據（用於與其他模型保持一致）
        X_train_seq, y_train_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_train_pca), self.y_train, self.sequence_length
        )
        X_test_seq, y_test_seq = self.create_sequences_for_pytorch(
            pd.DataFrame(X_test_pca), self.y_test, self.sequence_length
        )
        
        # 轉換為PyTorch張量，確保Float32類型
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)
        
        # 創建數據載入器 - 極小批次以提高量子計算穩定性
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 單樣本批次，最大穩定性
        
        # 創建QSVM-QNN混合模型：4個PCA特徵 → QSVM → QNN
        input_size = X_train_tensor.shape[2]  # 應該是4
        qsvmqnn_model = PyTorchQSVM_QNN(input_size=input_size, n_qubits=4, n_layers=2)
        
        # 訓練參數 (量子模型使用更保守的設置，使用HuberLoss)
        criterion = nn.HuberLoss(delta=1.0)  # HuberLoss對異常值更魯棒
        optimizer = optim.Adam(qsvmqnn_model.parameters(), lr=0.005)  # 調整學習率
        
        # 訓練模型
        qsvmqnn_model.train()
        print("🌌 訓練QSVM-QNN: QSVM特徵映射 + QNN學習 + HuberLoss魯棒訓練...")
        
        successful_epochs = 0
        for epoch in range(100):  # 減少epoch數量
            total_loss = 0
            batch_count = 0
            epoch_success = True
            
            for batch_X, batch_y in train_loader:
                try:
                    optimizer.zero_grad()
                    outputs = qsvmqnn_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 檢查loss是否有效
                    if torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        batch_count += 1
                    else:
                        print(f"⚠️ Epoch {epoch+1}: 跳過無效loss")
                        continue
                        
                except Exception as e:
                    print(f"⚠️ Epoch {epoch+1}: {str(e)[:50]}...")
                    epoch_success = False
                    break
            
            if epoch_success and batch_count > 0:
                successful_epochs += 1
                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.4f}, 成功批次: {batch_count}")
            
            # 如果連續失敗，提前終止
            if epoch > 10 and successful_epochs < epoch * 0.3:
                print(f"⚠️ 量子訓練不穩定，提前終止於Epoch {epoch+1}")
                break
        
        # 預測
        qsvmqnn_model.eval()
        try:
            with torch.no_grad():
                train_pred = qsvmqnn_model(X_train_tensor).numpy().flatten()
                test_pred = qsvmqnn_model(X_test_tensor).numpy().flatten()
            
            # 存儲結果（包含PCA變換器）
            self.models['QSVM-QNN (PyTorch+PennyLane)'] = qsvmqnn_model
            self.scalers['QSVM-QNN (PyTorch+PennyLane)'] = {'scaler': scaler, 'pca': pca}
            self.results['QSVM-QNN (PyTorch+PennyLane)'] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_mae': mean_absolute_error(y_train_seq, train_pred),
                'test_mae': mean_absolute_error(y_test_seq, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test_seq, test_pred)),
                'train_direction_acc': np.mean(np.sign(train_pred) == np.sign(y_train_seq)) * 100,
                'test_direction_acc': np.mean(np.sign(test_pred) == np.sign(y_test_seq)) * 100
            }
            
            print("✅ PyTorch QSVM-QNN訓練完成")
            
        except Exception as e:
            print(f"❌ QSVM-QNN預測階段出錯: {e}")
            print("⚠️ QSVM-QNN模型可能需要更多調優")
    
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
        self.train_xgboost()  # 新增XGBoost訓練
        
        if TORCH_AVAILABLE:
            self.train_pytorch_lstm()
            self.train_pytorch_transformer()
            
            # 訓練量子神經網絡 (如果PennyLane可用)
            if QML_AVAILABLE:
                self.train_pytorch_qnn()
                self.train_pytorch_qsvmqnn() # 新增QSVM-QNN訓練
            else:
                print("⚠️ 跳過QNN訓練，需要安裝PennyLane: pip install pennylane")
        
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