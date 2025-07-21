#!/usr/bin/env python3
"""
🚀 一鍵啟動全數據集批量模型比較
自動啟用實時寫入，每個模型訓練完成立即保存結果
"""

import os
import sys
sys.path.append('.')

def main():
    print("🌟 Curve Finance 全數據集模型比較系統")
    print("="*80)
    print("📊 處理所有84個數據集 (28個池子 × 3種數據版本)")
    print("💾 實時寫入已啟用 - 每個模型訓練完成後立即保存")
    print("🕐 預計運行時間: 2-4小時")
    print()
    
    print("🔄 結果文件將持續更新:")
    print("  📝 realtime_results.csv - 實時結果")
    print("  📊 intermediate_results_X_datasets.csv - 每5個數據集備份")
    print("  📈 最終彙總報告 - 運行完成後生成")
    print()
    
    # 檢查是否存在舊的實時結果文件
    if os.path.exists("realtime_results.csv"):
        print("⚠️ 發現舊的實時結果文件 realtime_results.csv")
        choice = input("是否備份並清除舊結果? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            import time
            timestamp = int(time.time())
            backup_name = f"realtime_results_backup_{timestamp}.csv"
            os.rename("realtime_results.csv", backup_name)
            print(f"✅ 舊結果已備份為: {backup_name}")
    
    print("\n🚀 啟動完整批量比較...")
    print("💡 您可以隨時查看 realtime_results.csv 監控進度")
    
    # 導入並運行批量比較
    try:
        from batch_model_comparison import BatchModelComparison
        
        batch_comparator = BatchModelComparison()
        batch_comparator.set_realtime_write(True)
        
        print("\n" + "="*80)
        batch_comparator.run_batch_comparison()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷運行")
        print("💾 已保存的結果仍然可用:")
        if os.path.exists("realtime_results.csv"):
            print("  ✅ realtime_results.csv - 包含已訓練的模型結果")
        print("  ✅ intermediate_results_*.csv - 中間備份文件")
        
    except Exception as e:
        print(f"\n❌ 運行過程中出現錯誤: {e}")
        print("💾 檢查已保存的部分結果:")
        if os.path.exists("realtime_results.csv"):
            print("  ✅ realtime_results.csv")
        
    print("\n🎉 感謝使用 Curve Finance 模型比較系統！")

if __name__ == "__main__":
    main() 