# 🌌 QNN量子神經網絡功能指南

> **新增功能**: 為Curve Virtual Price預測系統添加量子神經網絡(QNN)模型

---

## 🎯 **功能概覽**

### **新增模型**
✅ **QNN (Quantum Neural Network)** - 基於PennyLane + PyTorch的量子神經網絡  
✅ **混合架構** - 經典預處理 + 量子電路 + 經典後處理  
✅ **變分優化** - 量子參數與經典參數聯合訓練

### **技術特性**
- **量子比特數**: 4個模擬量子比特
- **量子電路**: 2層變分量子電路 (VQC)  
- **量子門**: RX/RY/RZ旋轉門 + CNOT糾纏門
- **測量方式**: PauliZ期望值測量
- **混合訓練**: 經典-量子參數聯合優化

---

## 🏗️ **系統架構**

### **QNN模型架構**
```python
PyTorchQNN架構:
├── 經典預處理層: Linear(25 → 24) + Tanh激活
├── 量子電路層: 4個量子比特 + 2層變分電路
│   ├── 數據編碼: RY(classical_data[i % len])
│   ├── 變分層: RX(θ₁) → RY(θ₂) → RZ(θ₃) 
│   ├── 糾纏層: CNOT(i, i+1) 環形連接
│   └── 測量: ⟨PauliZ⟩ 期望值
├── 經典後處理層: Linear(4 → 2) + ReLU + Linear(2 → 1)
└── 訓練: Adam優化器, lr=0.01
```

### **量子電路示意**
```
q₀: ──RY(x₀)─┬─RX(θ₀₀)─RY(θ₀₁)─RZ(θ₀₂)─●─────────┬─⟨Z⟩──
              │                           │         │
q₁: ──RY(x₁)─┼─RX(θ₁₀)─RY(θ₁₁)─RZ(θ₁₂)─X──●──────┼─⟨Z⟩──
              │                              │      │
q₂: ──RY(x₂)─┼─RX(θ₂₀)─RY(θ₂₁)─RZ(θ₂₂)─────X──●──┼─⟨Z⟩──
              │                                 │   │
q₃: ──RY(x₃)─┴─RX(θ₃₀)─RY(θ₃₁)─RZ(θ₃₂)────────X──┴─⟨Z⟩──
```

---

## 💻 **代碼實現**

### **核心類定義**
```python
class PyTorchQNN(nn.Module):
    """PyTorch + PennyLane 量子神經網絡模型"""
    
    def __init__(self, input_size, n_qubits=4, n_layers=2):
        # 量子設備初始化
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 經典預處理網絡
        self.pre_net = nn.Sequential(
            nn.Linear(input_size, n_qubits * n_layers * 3),
            nn.Tanh()
        )
        
        # 量子電路定義
        @qml.qnode(self.dev, interface="torch")  
        def quantum_circuit(inputs, weights):
            # 數據編碼 + 變分電路 + 測量
            # ... (詳見源碼)
            
        # 經典後處理網絡  
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, n_qubits // 2),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(n_qubits // 2, 1)
        )
```

### **訓練函數**
```python
def train_pytorch_qnn(self):
    """使用PyTorch + PennyLane訓練量子神經網絡模型"""
    
    # 數據準備 (使用相同的25個特徵)
    X_train_seq, y_train_seq = self.create_sequences_for_pytorch(...)
    
    # QNN模型創建
    qnn_model = PyTorchQNN(input_size, n_qubits=4, n_layers=2)
    
    # 訓練設置 (較高學習率，較少epoch)
    optimizer = optim.Adam(qnn_model.parameters(), lr=0.01)
    
    # 訓練循環 (含異常處理)
    for epoch in range(50):
        # ... 量子訓練循環
```

---

## 🔧 **安裝配置**

### **依賴安裝**
```bash
# 基本依賴
pip install pandas numpy matplotlib scikit-learn torch

# 量子機器學習依賴  
pip install pennylane>=0.28.0
pip install pennylane-lightning>=0.28.0

# 或者一次安裝所有
pip install -r requirements.txt
```

### **環境檢查**
```python
# 檢查QNN是否可用
python -c "
try:
    import torch, pennylane
    print('✅ QNN環境準備就緒')
except ImportError as e:
    print(f'❌ 缺少依賴: {e}')
"
```

---

## 🚀 **使用方法**

### **完整模型比較 (包含QNN)**
```python
from pytorch_model_comparison import PyTorchModelComparison

# 創建比較器
comparator = PyTorchModelComparison(pool_name='3pool')

# 運行完整比較 (4個模型)
results = comparator.run_complete_comparison()
# 輸出: Random Forest + LSTM + Transformer + QNN

# 查看QNN結果
if 'QNN (PyTorch+PennyLane)' in results:
    qnn_accuracy = results['QNN (PyTorch+PennyLane)']['test_direction_acc'] 
    print(f'QNN準確率: {qnn_accuracy:.2f}%')
```

### **單獨訓練QNN**
```python  
# 只訓練QNN模型
comparator.train_pytorch_qnn()

# 檢查QNN結果
if 'QNN (PyTorch+PennyLane)' in comparator.results:
    print("QNN訓練成功!")
else:
    print("QNN訓練失敗或被跳過")
```

---

## 📊 **性能預期**

### **訓練參數對比**

| 模型 | 訓練時間 | 學習率 | Epochs | 批次大小 |
|------|----------|--------|--------|----------|
| Random Forest | 2分鐘 | N/A | N/A | N/A |
| LSTM | 5-10分鐘 | 0.001 | 100 | 32 |
| Transformer | 10-15分鐘 | 0.001 | 100 | 32 |
| **QNN** | **15-30分鐘** | **0.01** | **50** | **16** |

### **QNN特殊設置原因**
- **較高學習率** (0.01): 量子梯度通常較小，需要較大步長
- **較少Epochs** (50): 量子訓練計算密集，避免過擬合
- **較小批次** (16): 量子電路計算較慢，減少批次提升穩定性

---

## 🌌 **量子優勢理論**

### **量子特性**
1. **量子疊加**: 同時探索多個解空間路徑
2. **量子糾纏**: 捕獲特徵間非經典關聯
3. **量子干涉**: 增強有用信號，抑制噪聲
4. **指數空間**: 4個量子比特可表示2^4=16維狀態空間

### **適用場景**
✅ **複雜非線性**: 量子非線性可能捕獲隱藏模式  
✅ **特徵糾纏**: 多特徵間的量子關聯  
✅ **探索性研究**: 量子機器學習前沿探索
⚠️ **計算資源**: 需要足夠的計算時間和內存

---

## ⚠️ **注意事項**

### **技術限制**
- **模擬量子**: 使用經典計算機模擬，非真實量子計算機
- **噪聲干擾**: 量子訓練可能不穩定，需要多次嘗試
- **計算密集**: QNN訓練時間明顯長於經典模型

### **實驗性質**  
- **前沿技術**: 量子機器學習仍在快速發展中
- **性能不保證**: QNN可能不一定優於經典模型
- **調參敏感**: 需要仔細調整量子參數才能獲得好結果

### **故障排除**
```python
# 常見問題檢查
1. ImportError: 安裝 pip install pennylane 
2. RuntimeError: 減少批次大小或序列長度
3. 訓練不穩定: 調整學習率或增加正則化
4. 內存不足: 減少量子比特數量 (n_qubits=2)
```

---

## 🔬 **實驗建議**

### **對比實驗**
1. **基準測試**: 先運行不含QNN的3模型比較
2. **QNN測試**: 安裝PennyLane後運行4模型比較  
3. **性能對比**: 比較QNN與其他模型的準確率差異
4. **參數調優**: 嘗試不同的量子比特數和電路層數

### **參數探索**
```python
# 實驗不同QNN配置
configs = [
    {'n_qubits': 2, 'n_layers': 1},  # 簡單配置
    {'n_qubits': 4, 'n_layers': 2},  # 標準配置  
    {'n_qubits': 6, 'n_layers': 3},  # 複雜配置
]

for config in configs:
    qnn = PyTorchQNN(input_size=25, **config)
    # 訓練和評估...
```

---

## 📈 **未來發展**

### **技術方向**
- **真實量子**: 集成IBM Qiskit或Google Cirq
- **量子優化**: 量子近似最佳化算法 (QAOA) 
- **混合模型**: QNN與經典模型的ensemble組合
- **量子特徵**: 量子特徵映射和量子核方法

### **應用擴展**  
- **多資產**: 擴展到股票、債券、商品預測
- **風險管理**: 量子VaR計算和投資組合最佳化
- **高頻交易**: 量子加速的實時決策系統

---

**🌟 QNN為Curve預測系統帶來了量子機器學習的前沿探索能力，代表了金融AI的未來方向！**

*最後更新: 2024-07-20 | 版本: 1.0.0* 