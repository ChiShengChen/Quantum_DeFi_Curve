#!/usr/bin/env python3
"""
🚀 批量模型比較系統 - 全數據集綜合性能評估
對所有365d.csv歷史數據進行模型比較並計算平均性能
"""

import os
import glob
import pandas as pd
import numpy as np
from pytorch_model_comparison import PyTorchModelComparison
import warnings
warnings.filterwarnings('ignore')

class BatchModelComparison:
    """批量模型比較系統"""
    
    def __init__(self, data_dir="/media/meow/Transcend/Quantum_curve_predict/free_historical_cache"):
        self.data_dir = data_dir
        self.results_all = []  # 存儲所有數據集結果
        self.failed_datasets = []  # 存儲失敗的數據集
        self.quantum_only = False  # 量子專用模式標誌
        self.realtime_write = False  # 實時寫入模式標誌
        
    def set_quantum_only_mode(self, enabled=True):
        """設置量子專用模式"""
        self.quantum_only = enabled
        if enabled:
            print("🌌 啟用量子專用模式 - 僅訓練QNN和QSVM-QNN模型")
        else:
            print("🔧 標準模式 - 訓練所有可用模型")
    
    def set_realtime_write(self, enabled=True):
        """設置實時寫入模式"""
        self.realtime_write = enabled
        if enabled:
            print("💾 啟用實時寫入模式 - 每個模型訓練完成後立即保存結果")
        else:
            print("📦 批量寫入模式 - 每5個數據集保存一次中間結果")
            
    def save_single_result(self, result_row):
        """保存單個結果到CSV(實時寫入模式)"""
        if not self.realtime_write:
            return
            
        import pandas as pd
        import os
        
        # 保存到實時結果文件 (不重複添加到results_all，因為主流程已經添加了)
        realtime_file = "realtime_results.csv"
        
        # 如果文件不存在，創建並寫入標題
        if not os.path.exists(realtime_file):
            df = pd.DataFrame([result_row])
            df.to_csv(realtime_file, index=False)
            print(f"💾 創建實時結果文件: {realtime_file}")
        else:
            # 追加模式寫入
            df = pd.DataFrame([result_row])
            df.to_csv(realtime_file, mode='a', header=False, index=False)
            
        print(f"✅ 實時保存: {result_row['dataset']} - {result_row['model']}")
    
    def get_all_datasets(self):
        """獲取所有365d.csv數據集文件"""
        pattern = os.path.join(self.data_dir, "*365d.csv")
        dataset_files = glob.glob(pattern)
        
        # 提取數據集名稱（去除路徑和後綴）
        datasets = []
        for file_path in dataset_files:
            filename = os.path.basename(file_path)
            # 提取池名稱（去除後綴）
            dataset_name = filename.replace('_365d.csv', '')
            datasets.append({
                'name': dataset_name,
                'file_path': file_path
            })
            
        return datasets
    
    def extract_pool_name(self, dataset_name):
        """從數據集名稱提取池名稱"""
        # 處理不同的命名模式
        if '_batch_historical' in dataset_name:
            return dataset_name.replace('_batch_historical', '')
        elif '_comprehensive_free_historical' in dataset_name:
            return dataset_name.replace('_comprehensive_free_historical', '')
        elif '_self_built_historical' in dataset_name:
            return dataset_name.replace('_self_built_historical', '')
        else:
            return dataset_name
    
    def run_single_dataset_comparison(self, dataset_info):
        """對單個數據集運行模型比較"""
        dataset_name = dataset_info['name']
        file_path = dataset_info['file_path']
        pool_name = self.extract_pool_name(dataset_name)
        
        print(f"\n{'='*80}")
        print(f"🚀 處理數據集: {dataset_name}")
        print(f"📁 文件路徑: {file_path}")
        print(f"🏊 池名稱: {pool_name}")
        print(f"{'='*80}")
        
        try:
            # 創建模型比較器，增加錯誤處理
            print("🔧 初始化模型比較器...")
            comparator = PyTorchModelComparison(pool_name=pool_name)
            
            # 直接讀取指定的CSV文件而不是動態下載
            if os.path.exists(file_path):
                print(f"📊 載入數據集: {file_path}")
                # 使用load_data方法載入數據
                success = comparator.load_data(file_path)
                if success:
                    print(f"✅ 數據載入成功: {len(comparator.data)} 條記錄")
                
                # 運行完整比較
                try:
                    # 準備數據
                    comparator.create_features()
                    comparator.prepare_data()
                    
                    # 訓練所有可用模型 (可通過參數控制)
                    if not hasattr(self, 'quantum_only') or not self.quantum_only:
                        comparator.train_random_forest()
                        comparator.train_xgboost()  # XGBoost
                    
                        # 分別嘗試每個PyTorch模型
                        pytorch_models = [
                            ('LSTM', comparator.train_pytorch_lstm),
                            ('Transformer', comparator.train_pytorch_transformer),
                        ]
                        
                        for model_name, train_func in pytorch_models:
                            try:
                                import torch
                                test_tensor = torch.tensor([1.0])  # 測試PyTorch是否真的可用
                                
                                print(f"🔥 嘗試訓練{model_name}模型...")
                                results_before = len(comparator.results)
                                train_func()
                                results_after = len(comparator.results)
                                
                                # 檢查是否真的添加了結果
                                if results_after > results_before:
                                    print(f"✅ {model_name}模型訓練成功")
                                else:
                                    print(f"⚠️ {model_name}模型訓練被跳過")
                            except Exception as e:
                                print(f"⚠️ 跳過{model_name}模型: {e}")
                    
                    # 量子模型 (獨立嘗試)
                    print("🌌 開始量子模型訓練...")
                    quantum_models = [
                        ('QNN', comparator.train_pytorch_qnn),
                        ('QSVM-QNN', comparator.train_pytorch_qsvmqnn),
                    ]
                    
                    for model_name, train_func in quantum_models:
                        try:
                            import pennylane as qml
                            # 檢查PyTorch是否可用 (量子模型需要PyTorch)
                            import torch
                            test_tensor = torch.tensor([1.0])  # 測試PyTorch是否真的可用
                            
                            print(f"🔮 嘗試訓練{model_name}模型...")
                            results_before = len(comparator.results)
                            train_func()
                            results_after = len(comparator.results)
                            
                            # 檢查是否真的添加了結果
                            if results_after > results_before:
                                print(f"✅ {model_name}模型訓練成功")
                            else:
                                print(f"⚠️ {model_name}模型訓練被跳過")
                        except Exception as e:
                            print(f"⚠️ 跳過{model_name}模型: {e}")
                    
                    # 收集結果
                    results = comparator.results
                    if results:
                        # 為每個模型添加數據集信息
                        for model_name, metrics in results.items():
                            result_row = {
                                'dataset': dataset_name,
                                'pool_name': pool_name, 
                                'model': model_name,
                                'test_mae': metrics['test_mae'],
                                'test_rmse': metrics['test_rmse'],
                                'test_direction_acc': metrics['test_direction_acc'],
                                'train_mae': metrics['train_mae'],
                                'train_rmse': metrics['train_rmse'], 
                                'train_direction_acc': metrics['train_direction_acc']
                            }
                            self.results_all.append(result_row)
                            self.save_single_result(result_row) # 實時保存
                        
                        print(f"✅ {dataset_name} 完成！收集了 {len(results)} 個模型結果")
                    else:
                        print(f"⚠️ {dataset_name} 沒有生成結果")
                        self.failed_datasets.append(dataset_name)
                        
                except Exception as e:
                    print(f"❌ {dataset_name} 模型訓練失敗: {e}")
                    self.failed_datasets.append(dataset_name)
            else:
                print(f"❌ 文件不存在: {file_path}")
                self.failed_datasets.append(dataset_name)
                
        except Exception as e:
            print(f"❌ {dataset_name} 處理失敗: {e}")
            self.failed_datasets.append(dataset_name)
    
    def run_batch_comparison(self, max_datasets=None):
        """運行批量模型比較"""
        print("🌟 開始批量模型比較 - 全數據集綜合評估")
        print("="*80)
        
        # 獲取所有數據集
        datasets = self.get_all_datasets()
        total_datasets = len(datasets)
        
        print(f"📊 發現 {total_datasets} 個數據集文件")
        
        if max_datasets:
            datasets = datasets[:max_datasets]
            print(f"🎯 限制處理前 {max_datasets} 個數據集")
        
        # 處理每個數據集
        for i, dataset_info in enumerate(datasets, 1):
            print(f"\n⏳ 進度: [{i}/{len(datasets)}] - {dataset_info['name']}")
            self.run_single_dataset_comparison(dataset_info)
            
            # 每處理5個數據集保存一次中間結果
            if i % 5 == 0 and self.results_all:
                print(f"💾 保存中間結果 (已處理{i}個數據集)...")
                self.save_intermediate_results(i)
        
        # 保存詳細結果
        if self.results_all:
            self.save_detailed_results()
            self.calculate_average_performance()
        
        # 報告失敗情況
        if self.failed_datasets:
            print(f"\n⚠️ 失敗的數據集 ({len(self.failed_datasets)}):")
            for failed in self.failed_datasets:
                print(f"  - {failed}")
        
        print(f"\n🎉 批量比較完成！")
        print(f"✅ 成功: {len(self.results_all)//6 if self.results_all else 0} 個數據集")  # 假設6個模型
        print(f"❌ 失敗: {len(self.failed_datasets)} 個數據集")
    
    def save_intermediate_results(self, processed_count):
        """保存中間結果"""
        if self.results_all:
            results_df = pd.DataFrame(self.results_all)
            intermediate_file = f"intermediate_results_{processed_count}_datasets.csv"
            results_df.to_csv(intermediate_file, index=False)
            print(f"💾 中間結果已保存: {intermediate_file}")
        
    def save_detailed_results(self):
        """保存詳細結果到CSV"""
        if not self.results_all:
            print("❌ 沒有結果可保存")
            return
            
        results_df = pd.DataFrame(self.results_all)
        
        # 保存詳細結果
        detailed_file = "all_datasets_detailed_results.csv"
        results_df.to_csv(detailed_file, index=False)
        print(f"💾 詳細結果已保存: {detailed_file}")
        
        # 顯示基本統計
        print(f"\n📊 數據集統計:")
        print(f"  總記錄數: {len(results_df)}")
        print(f"  數據集數: {results_df['dataset'].nunique()}")
        print(f"  模型數: {results_df['model'].nunique()}")
        print(f"  模型類型: {list(results_df['model'].unique())}")
        
    def calculate_average_performance(self):
        """計算並顯示平均性能"""
        if not self.results_all:
            print("❌ 沒有結果可分析")
            return
            
        results_df = pd.DataFrame(self.results_all)
        
        # 按模型計算平均性能
        avg_performance = results_df.groupby('model').agg({
            'test_mae': ['mean', 'std', 'count'],
            'test_rmse': ['mean', 'std'],  
            'test_direction_acc': ['mean', 'std'],
            'train_mae': ['mean', 'std'],
            'train_rmse': ['mean', 'std'],
            'train_direction_acc': ['mean', 'std']
        }).round(4)
        
        # 簡化列名
        avg_performance.columns = ['_'.join(col).strip() for col in avg_performance.columns]
        
        # 保存平均性能結果
        avg_file = "all_datasets_average_performance.csv" 
        avg_performance.to_csv(avg_file)
        print(f"💾 平均性能已保存: {avg_file}")
        
        # 顯示排名
        print(f"\n🏆 模型性能排名 (按測試方向準確率):")
        ranking = results_df.groupby('model')['test_direction_acc'].mean().sort_values(ascending=False)
        
        for i, (model, acc) in enumerate(ranking.items(), 1):
            count = results_df[results_df['model'] == model].shape[0]
            std = results_df[results_df['model'] == model]['test_direction_acc'].std()
            print(f"  {i}. {model}: {acc:.2f}% (±{std:.2f}%, n={count})")
            
        # 創建性能摘要表
        summary_data = []
        for model in ranking.index:
            model_data = results_df[results_df['model'] == model]
            summary_data.append({
                'Model': model,
                'Avg_Test_Direction_Acc': model_data['test_direction_acc'].mean(),
                'Std_Test_Direction_Acc': model_data['test_direction_acc'].std(),
                'Avg_Test_MAE': model_data['test_mae'].mean(),
                'Std_Test_MAE': model_data['test_mae'].std(),
                'Avg_Test_RMSE': model_data['test_rmse'].mean(),
                'Std_Test_RMSE': model_data['test_rmse'].std(),
                'Dataset_Count': len(model_data)
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_file = "models_performance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"💾 性能摘要已保存: {summary_file}")
        
        return avg_performance, summary_df


def main():
    """主函數 - 運行批量比較"""
    print("🌟 啟動全數據集批量模型比較系統")
    
    # 創建批量比較器
    batch_comparator = BatchModelComparison()
    
    # 運行全部數據集批量比較
    print("\n🚀 生產模式: 處理所有84個數據集...")
    # batch_comparator.run_batch_comparison()
    
    # 如果要測試模式，取消註釋下面這行
    batch_comparator.run_batch_comparison(max_datasets=5)


if __name__ == "__main__":
    main() 