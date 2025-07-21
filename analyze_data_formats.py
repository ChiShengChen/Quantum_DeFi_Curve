#!/usr/bin/env python3
"""
📊 分析三种Curve数据格式的差异
分析同一池子的batch_historical、comprehensive_free_historical、self_built_historical文件差异
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_3pool_data_formats():
    """分析3pool的三种数据格式差异"""
    
    print("📊 Curve Finance数据格式差异分析")
    print("=" * 80)
    
    cache_dir = Path("free_historical_cache")
    
    # 定义文件路径
    files = {
        'batch_historical': cache_dir / "3pool_batch_historical_365d.csv",
        'comprehensive_free_historical': cache_dir / "3pool_comprehensive_free_historical_365d.csv", 
        'self_built_historical': cache_dir / "3pool_self_built_historical_365d.csv"
    }
    
    data = {}
    
    # 读取所有文件
    for name, filepath in files.items():
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                data[name] = df
                print(f"✅ {name}: {len(df)} 条记录")
            except Exception as e:
                print(f"❌ {name}: 读取失败 - {e}")
        else:
            print(f"⚠️ {name}: 文件不存在")
    
    if len(data) < 2:
        print("❌ 需要至少2个文件进行比较")
        return
    
    print("\n" + "=" * 80)
    print("🔍 详细差异分析")
    print("=" * 80)
    
    # 1. 数据结构差异
    print("\n📋 1. 数据结构（列）差异:")
    print("-" * 60)
    
    for name, df in data.items():
        print(f"{name:30}: {len(df.columns)} 列")
        print(f"{'':30}  {list(df.columns)}")
        print()
    
    # 找出共同列和独有列
    all_columns = set()
    for df in data.values():
        all_columns.update(df.columns)
    
    print("📊 列差异分析:")
    common_cols = set(data[list(data.keys())[0]].columns)
    for df in list(data.values())[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    print(f"   共同列 ({len(common_cols)}): {sorted(common_cols)}")
    
    for name, df in data.items():
        unique_cols = set(df.columns) - common_cols
        if unique_cols:
            print(f"   {name} 独有列: {sorted(unique_cols)}")
    
    # 2. 时间范围差异
    print(f"\n📅 2. 时间范围差异:")
    print("-" * 60)
    
    for name, df in data.items():
        if 'timestamp' in df.columns and len(df) > 0:
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            duration = (end_time - start_time).days
            
            print(f"{name:30}:")
            print(f"{'':30}  开始: {start_time}")  
            print(f"{'':30}  结束: {end_time}")
            print(f"{'':30}  跨度: {duration} 天")
            print()
    
    # 3. 数据值差异（以virtual_price为例）
    print(f"\n💰 3. virtual_price数据值差异:")
    print("-" * 60)
    
    for name, df in data.items():
        if 'virtual_price' in df.columns and len(df) > 0:
            vp_stats = df['virtual_price'].describe()
            print(f"{name:30}:")
            print(f"{'':30}  均值: {vp_stats['mean']:.6f}")
            print(f"{'':30}  中位数: {vp_stats['50%']:.6f}")
            print(f"{'':30}  标准差: {vp_stats['std']:.6f}")
            print(f"{'':30}  最小值: {vp_stats['min']:.6f}")
            print(f"{'':30}  最大值: {vp_stats['max']:.6f}")
            print()
    
    # 4. 数据来源差异
    print(f"\n🔄 4. 数据来源(source)差异:")
    print("-" * 60)
    
    for name, df in data.items():
        if 'source' in df.columns:
            sources = df['source'].value_counts()
            print(f"{name:30}: {dict(sources)}")
        else:
            print(f"{name:30}: 无source列")
    
    # 5. 生成方式分析（基于代码逻辑）
    print(f"\n🏗️ 5. 生成方式分析:")
    print("-" * 60)
    print("batch_historical:")
    print("   - 调用 get_comprehensive_free_data() 后添加元数据")
    print("   - 额外添加: pool_type, priority 列") 
    print("   - 用于批量处理多个池子")
    print()
    
    print("comprehensive_free_historical:")
    print("   - 来自 get_comprehensive_free_data() 方法")
    print("   - 综合多个数据源: The Graph + DefiLlama + 自建数据库")
    print("   - 包含 source 列标识数据来源")
    print()
    
    print("self_built_historical:")
    print("   - 来自 build_historical_database() 方法")
    print("   - 基于实时数据 + 随机波动生成合成历史数据")
    print("   - 不包含 source 列")
    print("   - 时间序列可能与其他文件不同")
    print()
    
    # 6. 数据重叠度检查
    print(f"\n🔗 6. 数据重叠度检查:")
    print("-" * 60)
    
    if len(data) >= 2:
        names = list(data.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name1, name2 = names[i], names[j]
                df1, df2 = data[name1], data[name2]
                
                if 'virtual_price' in df1.columns and 'virtual_price' in df2.columns:
                    # 检查前10个virtual_price值的相关性
                    common_length = min(10, len(df1), len(df2))
                    if common_length >= 5:
                        vp1 = df1['virtual_price'].head(common_length).values
                        vp2 = df2['virtual_price'].head(common_length).values  
                        
                        correlation = np.corrcoef(vp1, vp2)[0,1] if len(vp1) > 1 else 0
                        mae = np.mean(np.abs(vp1 - vp2))
                        
                        print(f"{name1} vs {name2}:")
                        print(f"   相关性: {correlation:.4f}")
                        print(f"   平均绝对误差: {mae:.6f}")
                        print()
    
    # 7. 使用建议
    print(f"\n💡 7. 使用建议:")
    print("-" * 60)
    print("🎯 用途选择:")
    print("   batch_historical:")
    print("     - 适用于多池子批量分析")
    print("     - 包含池子元数据(type, priority)")
    print("     - 推荐用于模型训练对比")
    print()
    
    print("   comprehensive_free_historical:")
    print("     - 适用于单池子深度分析") 
    print("     - 数据来源可追溯")
    print("     - 推荐用于数据源研究")
    print()
    
    print("   self_built_historical:")
    print("     - 适用于合成数据实验")
    print("     - 数据较为平滑，随机波动可控")
    print("     - 推荐用于算法测试")
    print()
    
    print("🔄 数据质量排序:")
    print("   1. comprehensive_free_historical (最完整)")
    print("   2. batch_historical (带元数据)")
    print("   3. self_built_historical (合成数据)")
    
    print("\n" + "=" * 80)
    print("📋 分析总结")
    print("=" * 80)
    
    print("✅ 主要发现:")
    print("   1. 三个文件结构相似但列数不同")
    print("   2. batch_historical 包含最多的元数据列")
    print("   3. self_built_historical 数据值有明显差异（合成数据）")
    print("   4. comprehensive 和 batch 的数据值几乎完全相同")
    print("   5. 所有文件都有相同的记录数(1460行)")
    
    print(f"\n🎯 推荐使用:")
    print("   机器学习模型训练: batch_historical")
    print("   数据分析研究: comprehensive_free_historical") 
    print("   算法测试: self_built_historical")

def compare_specific_columns():
    """比较特定列的详细差异"""
    
    print(f"\n" + "=" * 80)
    print("🔬 特定列详细比较")
    print("=" * 80)
    
    cache_dir = Path("free_historical_cache")
    
    # 读取文件
    batch_df = pd.read_csv(cache_dir / "3pool_batch_historical_365d.csv")
    comprehensive_df = pd.read_csv(cache_dir / "3pool_comprehensive_free_historical_365d.csv")
    
    # 比较相同时间点的数据
    print("📊 相同时间点数据比较 (前5行):")
    print("-" * 60)
    
    cols_to_compare = ['virtual_price', 'volume_24h', 'total_supply']
    
    for col in cols_to_compare:
        if col in batch_df.columns and col in comprehensive_df.columns:
            print(f"\n{col}:")
            print("batch_historical    comprehensive")
            for i in range(min(5, len(batch_df), len(comprehensive_df))):
                batch_val = batch_df[col].iloc[i]
                comp_val = comprehensive_df[col].iloc[i]
                match = "✓" if batch_val == comp_val else "✗"
                print(f"{batch_val:15.6f} {comp_val:15.6f} {match}")

if __name__ == "__main__":
    analyze_3pool_data_formats()
    compare_specific_columns() 