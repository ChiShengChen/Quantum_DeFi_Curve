#!/usr/bin/env python3
"""
🚀 完整批量模型比較解決方案
支持所有84個數據集，自動檢測可用模型
"""

import subprocess
import sys
import os
from batch_model_comparison import BatchModelComparison

def check_pytorch_environment():
    """檢查PyTorch環境狀態"""
    print("🔍 檢查PyTorch環境狀態...")
    
    # 檢查base環境中的PyTorch
    print("\n📋 當前環境 (base):")
    try:
        import torch
        test_tensor = torch.tensor([1.0, 2.0])
        print(f"✅ PyTorch可用 (版本: {torch.__version__})")
        return True, "base"
    except Exception as e:
        print(f"❌ PyTorch不可用: {e}")
    
    # 檢查curve_transformer環境
    print("\n📋 檢查curve_transformer環境:")
    try:
        result = subprocess.run(['conda', 'run', '-n', 'curve_transformer', 'python', '-c', 
                               'import torch; print(f"PyTorch {torch.__version__} 可用")'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ curve_transformer環境: {result.stdout.strip()}")
            return True, "curve_transformer"
        else:
            print(f"❌ curve_transformer環境PyTorch測試失敗: {result.stderr}")
    except Exception as e:
        print(f"❌ 無法測試curve_transformer環境: {e}")
    
    return False, "none"

def run_with_curve_transformer_env():
    """在curve_transformer環境中運行完整批量比較"""
    print("🚀 在curve_transformer環境中運行完整批量比較...")
    
    # 創建一個臨時腳本來運行完整比較，並啟用實時寫入
    script_content = '''
import sys
sys.path.append('.')
from batch_model_comparison import BatchModelComparison

print("💾 啟用實時寫入模式 - 每個模型訓練完成後立即保存結果")

# 創建批量比較器並啟用實時寫入
batch_comparator = BatchModelComparison()
batch_comparator.set_realtime_write(True)

# 運行全部84個數據集
batch_comparator.run_batch_comparison()
'''
    
    with open('run_full_realtime.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    cmd = [
        'conda', 'run', '-n', 'curve_transformer',
        'python', 'run_full_realtime.py'
    ]
    
    try:
        print("💡 運行指令:", ' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        # 清理臨時文件
        try:
            os.remove('run_full_realtime.py')
        except:
            pass
            
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ 運行失敗: {e}")
        return False

def run_quantum_only_with_curve_transformer():
    """在curve_transformer環境中運行量子專用模式"""
    print("🌌 在curve_transformer環境中運行量子專用模式...")
    
    # 創建一個臨時腳本來運行量子專用模式
    script_content = '''
import sys
sys.path.append('.')
from batch_model_comparison import BatchModelComparison

# 創建批量比較器並啟用量子專用模式
batch_comparator = BatchModelComparison()
batch_comparator.set_quantum_only_mode(True)

# 運行批量比較
batch_comparator.run_batch_comparison()
'''
    
    with open('run_quantum_only.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    cmd = [
        'conda', 'run', '-n', 'curve_transformer',
        'python', 'run_quantum_only.py'
    ]
    
    try:
        print("💡 運行指令:", ' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        # 清理臨時文件
        try:
            os.remove('run_quantum_only.py')
        except:
            pass
            
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ 運行失敗: {e}")
        return False

def run_quantum_only_current_env():
    """在當前環境中運行量子專用模式"""
    print("🌌 在當前環境中運行量子專用模式...")
    
    # 創建批量比較器並啟用量子專用模式
    batch_comparator = BatchModelComparison()
    batch_comparator.set_quantum_only_mode(True)
    
    # 運行批量比較
    batch_comparator.run_batch_comparison()
    
    return True

def run_with_current_env():
    """在當前環境中運行(僅Random Forest + XGBoost)，啟用實時寫入"""
    print("🚀 在當前環境中運行批量比較...")
    print("💾 啟用實時寫入模式 - 每個模型訓練完成後立即保存結果")
    
    # 創建批量比較器並啟用實時寫入
    batch_comparator = BatchModelComparison()
    batch_comparator.set_realtime_write(True)
    
    # 運行全部84個數據集
    batch_comparator.run_batch_comparison()
    
    return True

def main():
    """主函數 - 智能選擇運行方案"""
    print("🌟 Curve Finance 全數據集模型比較系統")
    print("="*80)
    
    # 檢查環境
    pytorch_available, env_name = check_pytorch_environment()
    
    print(f"\n🎯 執行方案選擇:")
    
    if env_name == "curve_transformer":
        print("✨ 方案A: 使用curve_transformer環境運行所有模型 (實時寫入)")
        print("   包含: Random Forest, XGBoost, LSTM, Transformer, QNN, QSVM-QNN")
        print("🌌 方案B: 使用curve_transformer環境運行量子專用模式 (實時寫入)")
        print("   包含: QNN, QSVM-QNN")
        print("📋 方案C: 當前環境運行可用模型 (實時寫入)")
        print("   包含: Random Forest, XGBoost")
        
        print("\n請選擇運行方案:")
        print("  1. 方案A - 所有模型 (完整比較，實時保存)")
        print("  2. 方案B - 量子專用模式 (QNN + QSVM-QNN，實時保存)")
        print("  3. 方案C - 傳統模型 (Random Forest + XGBoost，實時保存)")
        
        while True:
            choice = input("\n請輸入選擇 (1/2/3): ").strip()
            if choice == "1":
                print("\n🚀 啟動方案A: 完整模型比較...")
                print("💾 實時寫入已啟用 - 每個模型訓練完成後立即保存到realtime_results.csv")
                success = run_with_curve_transformer_env()
                break
            elif choice == "2":
                print("\n🌌 啟動方案B: 量子專用模式...")
                print("💾 實時寫入已啟用 - 每個模型訓練完成後立即保存到realtime_results.csv")
                success = run_quantum_only_with_curve_transformer()
                break
            elif choice == "3":
                print("\n📋 啟動方案C: 傳統模型...")
                print("💾 實時寫入已啟用 - 每個模型訓練完成後立即保存到realtime_results.csv")
                success = run_with_current_env()
                break
            else:
                print("❌ 無效選擇，請輸入1、2或3")
                continue
        
        if not success and choice == "1":
            print("\n⚠️ curve_transformer環境運行失敗，切換到方案C...")
            run_with_current_env()
            
    elif env_name == "base" and pytorch_available:
        print("✨ 方案A: 當前環境運行所有模型")
        print("🌌 方案B: 當前環境運行量子專用模式")
        print("   包含: QNN, QSVM-QNN")
        
        print("\n請選擇運行方案:")
        print("  1. 方案A - 所有模型")
        print("  2. 方案B - 量子專用模式")
        
        while True:
            choice = input("\n請輸入選擇 (1/2): ").strip()
            if choice == "1":
                print("\n🚀 啟動方案A: 所有模型...")
                run_with_current_env()
                break
            elif choice == "2":
                print("\n🌌 啟動方案B: 量子專用模式...")
                run_quantum_only_current_env()
                break
            else:
                print("❌ 無效選擇，請輸入1或2")
                continue
                
    else:
        print("📋 方案A: 僅運行Random Forest + XGBoost模型")
        print("   原因: PyTorch不可用")
        print("🌌 方案B: 嘗試量子專用模式 (可能失敗)")
        
        print("\n💡 要獲得完整模型比較，請:")
        print("   1. 修復PyTorch CUDA問題，或")
        print("   2. 使用curve_transformer環境")
        
        print("\n請選擇運行方案:")
        print("  1. 方案A - 傳統模型 (Random Forest + XGBoost)")
        print("  2. 方案B - 嘗試量子模型 (可能失敗)")
        print("  0. 取消運行")
        
        while True:
            choice = input("\n請輸入選擇 (1/2/0): ").strip()
            if choice == "1":
                print("\n📋 啟動方案A: 傳統模型...")
                run_with_current_env()
                break
            elif choice == "2":
                print("\n🌌 啟動方案B: 嘗試量子模型...")
                print("⚠️ 警告: 由於PyTorch問題，此選項可能失敗")
                run_quantum_only_current_env()
                break
            elif choice == "0":
                print("❌ 用戶取消運行")
                return
            else:
                print("❌ 無效選擇，請輸入1、2或0")
                continue
    
    print("\n🎉 批量模型比較完成！")
    print("\n📊 結果文件:")
    print("   - realtime_results.csv: 實時結果 (每個模型訓練完成立即保存)")
    print("   - all_datasets_detailed_results.csv: 詳細結果 (最終彙總)")
    print("   - all_datasets_average_performance.csv: 平均性能") 
    print("   - models_performance_summary.csv: 性能摘要")
    print("   - intermediate_results_X_datasets.csv: 中間備份 (每5個數據集)")
    
    print("\n💡 建議優先查看:")
    print("   1. realtime_results.csv - 完整的實時結果")
    print("   2. models_performance_summary.csv - 快速性能排名")

if __name__ == "__main__":
    main() 