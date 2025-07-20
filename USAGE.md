# 🚀 Curve預測系統 - 簡化版使用指南

> **核心功能**: Curve Finance虛擬價格預測與模型比較

---

## 📁 **項目結構**

```
Quantum_curve_predict/
├── free_historical_data.py      # 數據收集和處理核心
├── pytorch_model_comparison.py  # 純PyTorch模型比較系統  
├── free_historical_cache/       # 歷史數據緩存
├── requirements.txt             # 依賴包列表
├── README.md                    # 詳細說明文檔
└── USAGE.md                     # 本使用指南
```

---

## 🚀 **快速開始**

### **1. 安裝依賴**
```bash
pip install -r requirements.txt
```

### **2. 收集數據**
```bash
# 使用數據收集系統
python free_historical_data.py
```

### **3. 運行模型比較**
```bash  
# 運行PyTorch模型比較
python pytorch_model_comparison.py
```

---

## 💻 **核心功能**

### **🗂️ 數據收集 (`free_historical_data.py`)**

**功能**:
- 收集Curve Finance池子歷史數據
- 支援37個主要池子
- 自動緩存和數據管理
- 批量數據處理

**使用方法**:
```python
from free_historical_data import CurveFreeHistoricalDataManager

# 創建數據管理器
manager = CurveFreeHistoricalDataManager()

# 收集單個池子數據
data = manager.get_comprehensive_free_data(
    pool_name="3pool",
    days=365
)

# 批量收集多個池子
batch_data = manager.get_all_main_pools_data(days=90)
```

### **🤖 模型比較 (`pytorch_model_comparison.py`)**

**功能**:
- Random Forest基準模型
- PyTorch LSTM深度學習模型  
- PyTorch Transformer注意力模型
- 自動性能比較和可視化

**使用方法**:
```python
from pytorch_model_comparison import PyTorchModelComparison

# 創建比較器
comparator = PyTorchModelComparison(
    pool_name='3pool',
    sequence_length=24
)

# 運行完整比較
results = comparator.run_complete_comparison()
```

---

## 📊 **預期結果**

### **🎯 模型性能基準**

| 模型 | 預期準確率 | 訓練時間 | 說明 |
|------|-----------|----------|------|
| **Random Forest** | 66-75% | 2分鐘 | 快速穩健基準 |
| **LSTM (PyTorch)** | 70-80% | 5-10分鐘 | 序列模式學習 |
| **Transformer** | 72-82% | 10-15分鐘 | 注意力機制 |

### **📈 輸出文件**

運行後會生成：
- 性能比較圖表 (`.png`)
- 詳細結果報告 (`.txt`) 
- 數據表格 (`.csv`)

---

## ⚙️ **系統需求**

### **最低需求**
- Python 3.7+
- 8GB RAM
- 2GB 可用磁盤空間

### **推薦配置**
- Python 3.9+
- 16GB RAM
- GPU支持 (可選，加速深度學習)

---

## 🔧 **故障排除**

### **常見問題**

**Q: PyTorch安裝失敗**
```bash
# CPU版本安裝
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本安裝 (如果有CUDA)
pip install torch torchvision torchaudio
```

**Q: 數據收集失敗** 
- 檢查網絡連接
- 確認API端點可用
- 重試機制會自動處理暫時性錯誤

**Q: 內存不足**
```python
# 減少批次大小或序列長度
comparator = PyTorchModelComparison(sequence_length=12)  # 降到12
```

---

## 🎯 **使用建議**

### **數據收集**
1. **開始時**: 收集7-30天數據進行測試
2. **生產環境**: 使用365天完整數據
3. **定期更新**: 每日或每週更新數據

### **模型選擇**
1. **快速測試**: 使用Random Forest
2. **高精度需求**: 嘗試Transformer模型
3. **平衡選擇**: LSTM在速度和精度間平衡

### **投資建議**
- 準確率>70%: 可考慮實際應用
- 準確率60-70%: 謹慎使用，結合其他指標  
- 準確率<60%: 僅供參考，不建議投資依據

---

## 📚 **延伸功能**

### **自訂擴展**
1. 添加新的池子地址到數據收集器
2. 調整深度學習模型架構
3. 實現自訂特徵工程
4. 集成外部數據源

### **進階用法**
```python
# 自訂特徵工程
comparator.create_features()
comparator.prepare_data()

# 單獨訓練特定模型
comparator.train_random_forest()
comparator.train_pytorch_lstm()
comparator.train_pytorch_transformer()

# 自訂可視化
comparator.visualize_predictions(last_n_points=500)
```

---

**🎉 現在您擁有了一個精簡、高效的Curve虛擬價格預測系統！** 