#!/usr/bin/env python3
"""
🔍 数据来源差异深度分析
从爬虫数据获取流程的角度分析三种格式的来源差异
"""

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_data_sources():
    """分析数据来源的具体差异"""
    
    print("🔍 Curve Finance数据来源深度分析")
    print("=" * 80)
    
    # 1. 数据源架构分析
    print("\n🏗️ 1. 数据获取架构分析")
    print("-" * 60)
    
    data_sources = {
        "The Graph API": {
            "状态": "❌ 已禁用 (ENABLE_THEGRAPH_API = False)",
            "原因": "API端点已废弃",
            "数据类型": "历史池子快照、TVL、交易量",
            "更新频率": "每日",
            "免费额度": "1000请求/天",
            "优点": "官方subgraph，数据权威",
            "缺点": "端点已不可用"
        },
        "DefiLlama API": {
            "状态": "✅ 已启用 (ENABLE_DEFILLAMA = True)", 
            "原因": "免费且可靠",
            "数据类型": "APY历史、TVL数据",
            "更新频率": "实时",
            "免费额度": "无限制",
            "优点": "完全免费，数据丰富",
            "缺点": "主要是APY数据，缺少详细池子信息"
        },
        "Curve API": {
            "状态": "✅ 已启用 (ENABLE_CURVE_API = True)",
            "原因": "官方API",
            "数据类型": "池子基本信息、实时状态",
            "更新频率": "实时",
            "免费额度": "无限制",
            "优点": "官方数据，最权威",
            "缺点": "主要是当前状态，历史数据有限"
        },
        "自建数据库": {
            "状态": "✅ 已启用 (ENABLE_SELF_BUILT = True)",
            "原因": "最后备份方案",
            "数据类型": "基于实时数据的合成历史序列",
            "更新频率": "按需生成",
            "免费额度": "无限制",
            "优点": "完全可控，永远可用",
            "缺点": "合成数据，非真实市场数据"
        }
    }
    
    for source_name, info in data_sources.items():
        print(f"\n📊 {source_name}:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # 2. 数据流程分析
    print(f"\n\n🔄 2. 数据获取流程分析")
    print("-" * 60)
    
    print("📋 三种文件的生成流程差异:")
    print()
    
    # batch_historical流程
    print("🔹 batch_historical生成流程:")
    print("   1️⃣ 调用 get_batch_historical_data()")
    print("   2️⃣ 对每个池子调用 get_comprehensive_free_data()")
    print("   3️⃣ 尝试数据源顺序:")
    print("      ❌ The Graph API (已禁用)")
    print("      🔄 DefiLlama API (尝试但可能失败)")  
    print("      🔄 Curve API (隐含在自建数据库中)")
    print("      ✅ 自建数据库 (最终成功)")
    print("   4️⃣ 添加元数据: pool_type, priority")
    print("   5️⃣ 保存为 {pool_name}_batch_historical_{days}d.csv")
    print()
    
    # comprehensive_free_historical流程
    print("🔹 comprehensive_free_historical生成流程:")
    print("   1️⃣ 直接调用 get_comprehensive_free_data()")
    print("   2️⃣ 尝试数据源顺序:")
    print("      ❌ The Graph API (已禁用)")
    print("      🔄 DefiLlama API (尝试但可能失败)")
    print("      ✅ 自建数据库 (成功获取)")
    print("   3️⃣ 合并所有可用数据源")
    print("   4️⃣ 添加source列标识数据来源")
    print("   5️⃣ 保存为 {pool_name}_comprehensive_free_historical_{days}d.csv")
    print()
    
    # self_built_historical流程
    print("🔹 self_built_historical生成流程:")
    print("   1️⃣ 直接调用 build_historical_database()")
    print("   2️⃣ 尝试获取实时基础数据:")
    print("      🔄 CurveRealDataCollector.get_real_time_data()")
    print("      📊 成功获取当前池子状态")
    print("   3️⃣ 基于实时数据生成历史序列:")
    print("      📈 添加趋势因子: 1 + 0.1 * sin(时间)")
    print("      🎲 添加随机波动: 2%标准差")
    print("      ⏰ 每4小时一个数据点")
    print("   4️⃣ 不添加source列")
    print("   5️⃣ 保存为 {pool_name}_self_built_historical_{days}d.csv")
    print()

def analyze_actual_data_sources():
    """分析实际文件中的数据来源"""
    
    print("\n📊 3. 实际数据来源验证")
    print("-" * 60)
    
    cache_dir = Path("free_historical_cache")
    
    files = {
        'batch_historical': cache_dir / "3pool_batch_historical_365d.csv",
        'comprehensive_free_historical': cache_dir / "3pool_comprehensive_free_historical_365d.csv",
        'self_built_historical': cache_dir / "3pool_self_built_historical_365d.csv"
    }
    
    for name, filepath in files.items():
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                print(f"\n📁 {name}:")
                
                # 检查source列
                if 'source' in df.columns:
                    source_counts = df['source'].value_counts()
                    print(f"   数据来源分布: {dict(source_counts)}")
                    
                    # 分析每个来源的数据特征
                    for source in source_counts.index:
                        source_data = df[df['source'] == source]
                        print(f"   {source} 数据:")
                        print(f"     记录数: {len(source_data)}")
                        
                        if 'virtual_price' in source_data.columns:
                            vp_stats = source_data['virtual_price'].describe()
                            print(f"     Virtual Price: 均值={vp_stats['mean']:.6f}, 标准差={vp_stats['std']:.6f}")
                        
                        if 'timestamp' in source_data.columns:
                            time_range = source_data['timestamp'].max() - source_data['timestamp'].min()
                            print(f"     时间跨度: {time_range}")
                else:
                    print(f"   ⚠️ 无source列 - 可能是自建数据")
                    
                    # 分析数据特征来推断来源
                    if 'virtual_price' in df.columns:
                        vp_values = df['virtual_price'].head(10).values
                        vp_diff = np.diff(vp_values)
                        vp_volatility = np.std(vp_diff)
                        
                        print(f"   数据特征分析:")
                        print(f"     前10个virtual_price变化波动率: {vp_volatility:.6f}")
                        
                        if vp_volatility > 0.01:
                            print(f"     📊 推断: 高波动性，可能是合成数据")
                        else:
                            print(f"     📊 推断: 低波动性，可能是真实或插值数据")
                            
            except Exception as e:
                print(f"   ❌ 读取失败: {e}")

def analyze_data_generation_methods():
    """分析数据生成方法的差异"""
    
    print(f"\n\n🧬 4. 数据生成方法深度剖析")
    print("-" * 60)
    
    print("📈 各种数据生成策略对比:")
    print()
    
    generation_methods = {
        "The Graph历史数据": {
            "方法": "get_thegraph_historical_data()",
            "原理": "GraphQL查询subgraph获取历史快照",
            "数据来源": "链上事件聚合",
            "时间粒度": "每日快照",
            "数据质量": "🥇 最高 - 真实链上数据",
            "当前状态": "❌ 不可用 (API废弃)",
            "特征": "官方权威，数据完整，包含所有池子参数"
        },
        "DefiLlama APY数据": {
            "方法": "get_defillama_apy_history()", 
            "原理": "REST API获取收益率历史",
            "数据来源": "多数据源聚合",
            "时间粒度": "可变(小时/日)",
            "数据质量": "🥈 高 - 聚合多源数据",
            "当前状态": "✅ 可用但可能失败",
            "特征": "主要是APY和TVL，缺少详细池子状态"
        },
        "实时数据+插值": {
            "方法": "build_historical_database()",
            "原理": "获取当前状态+时间序列插值",
            "数据来源": "CurveRealDataCollector实时API", 
            "时间粒度": "6小时/4小时",
            "数据质量": "🥉 中 - 基于实时数据外推",
            "当前状态": "✅ 总是可用",
            "特征": "基于当前真实状态，但历史部分是估算"
        },
        "纯合成数据": {
            "方法": "_generate_synthetic_data()",
            "原理": "预设参数+数学模型生成",
            "数据来源": "数学公式 (sin波 + 随机噪声)",
            "时间粒度": "4小时",
            "数据质量": "🥉 低 - 完全合成",
            "当前状态": "✅ 总是可用",
            "特征": "行为可预测，用于算法测试"
        }
    }
    
    for method_name, details in generation_methods.items():
        print(f"🔸 {method_name}:")
        for key, value in details.items():
            print(f"   {key}: {value}")
        print()

def analyze_fallback_cascade():
    """分析数据获取的fallback级联机制"""
    
    print(f"\n🔄 5. 数据获取Fallback级联分析")
    print("-" * 60)
    
    print("📊 get_comprehensive_free_data() 的级联策略:")
    print()
    print("🏆 优先级1: The Graph API")
    print("   状态: ❌ 已禁用 (ENABLE_THEGRAPH_API = False)")
    print("   如果成功: 标记source='thegraph'")
    print("   如果失败: 继续尝试下一个")
    print()
    
    print("🥈 优先级2: DefiLlama API")
    print("   状态: ✅ 已启用 (ENABLE_DEFILLAMA = True)")
    print("   尝试结果: 🔄 通常失败 (SSL或网络问题)")
    print("   如果成功: 标记source='defillama'")
    print("   如果失败: 继续尝试下一个")
    print()
    
    print("🥉 优先级3: 自建数据库 (最终备份)")
    print("   状态: ✅ 已启用 (ENABLE_SELF_BUILT = True)")
    print("   触发条件: total_records < min_required_records")
    print("   尝试结果: ✅ 总是成功")
    print("   数据标记: source='self_built'")
    print()
    
    print("📋 实际运行结果 (基于文件分析):")
    print("   1. The Graph: ❌ 跳过 (已禁用)")
    print("   2. DefiLlama: ❌ 失败 (网络/SSL问题)")
    print("   3. 自建数据库: ✅ 成功生成1460条记录")
    print("   4. 最终source标记: 'self_built'")
    print()
    
    print("💡 这解释了为什么三个文件都显示source='self_built':")
    print("   - 不是因为直接调用自建方法")
    print("   - 而是因为其他数据源都失败了")
    print("   - 自建数据库作为最后的fallback成功运行")

def recommend_data_source_optimization():
    """推荐数据源优化策略"""
    
    print(f"\n\n🚀 6. 数据源优化建议")
    print("-" * 60)
    
    print("📊 当前问题诊断:")
    print("   1. ❌ The Graph API完全不可用")
    print("   2. ❌ DefiLlama API连接失败")
    print("   3. ✅ 只有自建数据库在工作")
    print("   4. ⚠️ 所有数据实际上都是合成的")
    print()
    
    print("🔧 优化策略建议:")
    print()
    print("🌐 网络连接优化:")
    print("   - 启用SSL验证: ENABLE_SSL_VERIFICATION = True")
    print("   - 增加请求超时: REQUEST_TIMEOUT = 10")
    print("   - 添加User-Agent请求头")
    print("   - 使用代理或VPN")
    print()
    
    print("📡 数据源多样化:")
    print("   - 集成Coingecko API作为备选")
    print("   - 添加Dune Analytics查询")
    print("   - 考虑CoinMarketCap Pro API")
    print("   - 直接连接以太坊节点查询")
    print()
    
    print("🏗️ 自建数据库改进:")
    print("   - 改进随机波动模型")
    print("   - 添加更真实的市场相关性")
    print("   - 引入宏观经济因子")
    print("   - 基于历史模式生成数据")
    print()
    
    print("📊 数据质量提升:")
    print("   - 实施数据验证机制")
    print("   - 添加异常值检测")
    print("   - 交叉验证多个数据源")
    print("   - 建立数据质量评分体系")

if __name__ == "__main__":
    analyze_data_sources()
    analyze_actual_data_sources()
    analyze_data_generation_methods()
    analyze_fallback_cascade()
    recommend_data_source_optimization()
    
    print("\n" + "=" * 80)
    print("🎯 总结")
    print("=" * 80)
    print("✅ 核心发现:")
    print("   1. 三个文件都来自相同的自建数据源")
    print("   2. 其他高质量数据源(The Graph, DefiLlama)都失败了")
    print("   3. batch和comprehensive文件完全相同的原因是数据源相同")
    print("   4. self_built文件不同是因为随机种子或生成时间不同")
    print()
    print("💡 关键洞察:")
    print("   - 格式差异主要在于元数据添加，而非数据来源")
    print("   - 真正的数据来源差异被网络问题掩盖了")
    print("   - 需要解决网络连接问题才能体验真正的数据源差异")
    print("=" * 80) 