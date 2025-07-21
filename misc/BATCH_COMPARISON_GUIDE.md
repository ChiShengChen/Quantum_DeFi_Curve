# 🚀 全數據集批量模型比較指南

## 問題解答

**您的問題**: `all_datasets_detailed_results.csv` 只看到 Random Forest 和 XGBoost 結果，沒有 Transformer、QNN 等模型。

**原因**: PyTorch CUDA 庫冲突，導致所有基於 PyTorch 的模型被跳過。

## 📊 數據集統計

- **總數據集**: 84個 (每個池子3種版本: batch, comprehensive, self_built)
- **池數量**: 28個不同的 Curve Finance 池
- **數據範圍**: 365天歷史數據
- **特徵數**: 25+ 工程特徵

## 🔧 解決方案

### 方案 A: 完整模型比較 (推薦)

使用 `curve_transformer` 環境運行所有6個模型：

```bash
# 1. 自動檢測環境並選擇最佳方案
python run_complete_batch.py

# 2. 或直接在curve_transformer環境中運行
conda activate curve_transformer
python batch_model_comparison.py
```

**包含模型**:
- ✅ Random Forest
- ✅ XGBoost  
- ✅ LSTM (PyTorch)
- ✅ Transformer (PyTorch)
- ✅ QNN (PyTorch + PennyLane)
- ✅ QSVM-QNN (PyTorch + PennyLane)

### 方案 B: 部分模型比較 (當前可用)

在當前環境運行可用模型：

```bash
python batch_model_comparison.py
```

**包含模型**:
- ✅ Random Forest
- ✅ XGBoost
- ❌ LSTM (PyTorch問題)
- ❌ Transformer (PyTorch問題) 
- ❌ QNN (PyTorch問題)
- ❌ QSVM-QNN (PyTorch問題)

## 📈 輸出結果

### 詳細結果文件

1. **`all_datasets_detailed_results.csv`**
   - 每個數據集、每個模型的詳細性能
   - 包含: dataset, pool_name, model, test_mae, test_rmse, test_direction_acc 等

2. **`all_datasets_average_performance.csv`**
   - 按模型聚合的平均性能統計
   - 包含均值、標準差、數據集計數

3. **`models_performance_summary.csv`**  
   - 模型性能排名摘要
   - 便於快速比較不同模型

### 中間結果文件

- **`intermediate_results_X_datasets.csv`**: 每處理5個數據集保存一次

## 🎯 預期結果示例

如果所有模型都能運行，您會看到類似這樣的結果：

```
🏆 模型性能排名 (按測試方向準確率):
  1. XGBoost: 70.59% (±1.13%, n=84)
  2. Random Forest: 68.63% (±1.13%, n=84) 
  3. LSTM (PyTorch): 65.45% (±2.34%, n=84)
  4. Transformer (PyTorch): 63.21% (±3.12%, n=84)
  5. QSVM-QNN (PyTorch+PennyLane): 58.76% (±4.23%, n=84)
  6. QNN (PyTorch+PennyLane): 56.34% (±3.45%, n=84)
```

## 🔍 數據集詳細說明

### 數據範圍
- **時間跨度**: 365天 (2024-07-20 到 2025-07-20)
- **採樣頻率**: 每6小時一個數據點
- **原始數據點**: ~1460 條/數據集
- **處理後數據**: ~765 條/數據集 (去除NaN和滯後期)

### 訓練/測試分割
- **訓練集**: 80% (~612 樣本)
- **測試集**: 20% (~153 樣本)  
- **時間序列分割**: 按時間順序，避免數據洩漏

### 特徵工程
1. **滯後特徵**: 1, 6, 24, 168 個時間點
2. **移動平均**: 24, 168, 672 窗口
3. **波動率指標**: 標準差、變異係數
4. **技術指標**: RSI, 價格變化率
5. **流動性特徵**: 總供應量變化
6. **餘額特徵**: 代幣餘額比率
7. **時間特徵**: 小時、星期、月份

### 目標變數
- **預測目標**: 24個時間點後的虛擬價格 (6小時後)
- **評估指標**:
  - MAE (平均絕對誤差)
  - RMSE (均方根誤差)  
  - 方向準確率 (漲跌方向預測準確性)

## 🚀 立即開始

```bash
# 克隆或進入項目目錄
cd /media/meow/Transcend/Quantum_curve_predict

# 運行完整解決方案
python run_complete_batch.py
```

系統會自動：
1. 檢測可用環境
2. 選擇最佳運行方案  
3. 處理所有84個數據集
4. 生成綜合性能報告

預計運行時間: 2-4小時 (取決於模型數量和硬體性能)

## ❓ 故障排除

### PyTorch CUDA問題
```bash
# 重新安裝PyTorch (無CUDA版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 環境問題
```bash
# 檢查conda環境
conda env list
conda activate curve_transformer
```

### 記憶體不足
```bash
# 降低批量大小或使用較少數據集進行測試
python run_complete_batch.py --max-datasets 10
``` 