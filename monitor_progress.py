#!/usr/bin/env python3
"""
📊 批量處理進度監控器
"""

import os
import glob
import time
import pandas as pd

def monitor_progress():
    """監控批量處理進度"""
    
    print("📊 批量模型比較進度監控器")
    print("="*60)
    
    while True:
        try:
            # 檢查中間結果文件
            intermediate_files = glob.glob("intermediate_results_*_datasets.csv")
            if intermediate_files:
                # 找到最新的中間結果文件
                latest_file = max(intermediate_files, key=os.path.getmtime)
                
                # 讀取並分析最新結果
                try:
                    df = pd.read_csv(latest_file)
                    unique_datasets = df['dataset'].nunique()
                    unique_models = df['model'].nunique()
                    total_results = len(df)
                    
                    print(f"🔄 最新進度 ({time.strftime('%H:%M:%S')}): {latest_file}")
                    print(f"  ✅ 已完成數據集: {unique_datasets}")
                    print(f"  🧠 模型類型數: {unique_models}")
                    print(f"  📝 總結果數: {total_results}")
                    
                    if unique_models > 0:
                        # 顯示當前最佳性能
                        avg_performance = df.groupby('model')['test_direction_acc'].mean().sort_values(ascending=False)
                        print(f"  🏆 當前排名:")
                        for i, (model, acc) in enumerate(avg_performance.head(3).items(), 1):
                            print(f"    {i}. {model}: {acc:.1f}%")
                    
                except Exception as e:
                    print(f"⚠️ 讀取中間結果失敗: {e}")
            else:
                print(f"⏳ 等待中間結果... ({time.strftime('%H:%M:%S')})")
            
            # 檢查最終結果
            if os.path.exists("all_datasets_detailed_results.csv"):
                final_df = pd.read_csv("all_datasets_detailed_results.csv")
                final_datasets = final_df['dataset'].nunique()
                
                print(f"\n🎉 批量處理已完成！")
                print(f"  📊 總處理數據集: {final_datasets}")
                print(f"  📋 最終結果文件: all_datasets_detailed_results.csv")
                print(f"  📈 性能摘要文件: models_performance_summary.csv")
                break
            
            print("-" * 60)
            time.sleep(30)  # 每30秒檢查一次
            
        except KeyboardInterrupt:
            print("\n⏹️ 監控已停止")
            break
        except Exception as e:
            print(f"❌ 監控錯誤: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_progress() 