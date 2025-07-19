#!/usr/bin/env python3
"""
天数配置使用示例
展示如何在free_historical_data.py中切换不同的时间跨度
"""

# 修改free_historical_data.py中的天数配置示例
def show_config_examples():
    """展示配置修改方法"""
    
    print("🔧 如何修改 free_historical_data.py 的天数配置")
    print("=" * 60)
    
    print("\n📝 方法1: 直接修改配置变量")
    print("编辑 free_historical_data.py 文件，找到这部分：")
    print()
    print("# 当前使用的配置 - 修改这里来改变所有方法的默认值")
    print("CURRENT_DAYS_SETTING = FULL_YEAR_DAYS  # ← 改成你想要的配置")
    print()
    print("可选配置:")
    print("- QUICK_TEST_DAYS (7天)")
    print("- MEDIUM_RANGE_DAYS (90天)") 
    print("- FULL_YEAR_DAYS (365天)")
    print("- 或直接写数字，如: CURRENT_DAYS_SETTING = 180")
    
    print("\n📝 方法2: 使用程序内切换")
    print("在程序中调用 switch_days_config() 函数：")
    print()
    print("from free_historical_data import switch_days_config")
    print("switch_days_config('quick')    # 切换到7天")
    print("switch_days_config('medium')   # 切换到90天")
    print("switch_days_config('full')     # 切换到365天")
    
    print("\n📝 方法3: 自定义天数")
    print("可以在调用方法时覆盖默认值：")
    print()
    print("manager = FreeHistoricalDataManager()")
    print("df = manager.get_comprehensive_free_data(")
    print("    pool_address='0x...', ")
    print("    pool_name='3pool',")
    print("    days=180  # ← 直接指定天数")
    print(")")
    
    print("\n📊 不同配置的特点对比:")
    print("┌─────────────┬───────┬─────────────┬─────────────────────┐")
    print("│ 配置名称    │ 天数  │ 数据量      │ 适用场景            │")
    print("├─────────────┼───────┼─────────────┼─────────────────────┤")
    print("│ quick       │ 7天   │ ~168条记录  │ 快速测试、调试      │")
    print("│ medium      │ 90天  │ ~2160条记录 │ 短期趋势分析        │")
    print("│ full        │ 365天 │ ~8760条记录 │ 年度分析、模型训练  │")
    print("└─────────────┴───────┴─────────────┴─────────────────────┘")
    
    print("\n⚠️  注意事项:")
    print("- The Graph API 每天有1000次查询限制")
    print("- 天数越多，获取时间越长")
    print("- 自建数据库模式：365天 ≈ 需要6-8小时")
    print("- 推荐优先使用The Graph模式获取历史数据")

def interactive_config():
    """交互式配置选择"""
    print("\n🎛️  交互式天数配置")
    print("=" * 30)
    
    print("请选择历史数据时间跨度:")
    print("1. 快速测试 (7天) - 适合调试")
    print("2. 短期分析 (90天) - 适合趋势分析")  
    print("3. 完整年度 (365天) - 适合模型训练")
    print("4. 自定义天数")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    config_map = {
        '1': ('QUICK_TEST_DAYS', 7),
        '2': ('MEDIUM_RANGE_DAYS', 90),
        '3': ('FULL_YEAR_DAYS', 365)
    }
    
    if choice in config_map:
        config_name, days = config_map[choice]
        print(f"\n✅ 已选择: {config_name} ({days}天)")
        print(f"📝 在 free_historical_data.py 中修改为:")
        print(f"CURRENT_DAYS_SETTING = {config_name}")
        
    elif choice == '4':
        try:
            custom_days = int(input("请输入自定义天数 (1-365): "))
            if 1 <= custom_days <= 365:
                print(f"\n✅ 已选择自定义: {custom_days}天")
                print(f"📝 在 free_historical_data.py 中修改为:")
                print(f"CURRENT_DAYS_SETTING = {custom_days}")
            else:
                print("❌ 天数必须在 1-365 之间")
        except ValueError:
            print("❌ 请输入有效数字")
    else:
        print("❌ 无效选择")

def show_performance_estimates():
    """显示不同配置的性能预估"""
    print("\n⏱️  性能预估")
    print("=" * 40)
    
    estimates = [
        ("快速测试 (7天)", "7天", "~168条", "1-2分钟", "适合快速验证"),
        ("短期分析 (90天)", "90天", "~2160条", "5-10分钟", "趋势分析推荐"),
        ("完整年度 (365天)", "365天", "~8760条", "30-60分钟", "深度训练必备"),
    ]
    
    print("┌──────────────────┬──────┬──────────┬──────────┬────────────┐")
    print("│ 配置             │ 天数 │ 预计记录 │ 获取时间 │ 推荐用途   │")
    print("├──────────────────┼──────┼──────────┼──────────┼────────────┤")
    
    for config, days, records, time_est, usage in estimates:
        print(f"│ {config:<16} │ {days:<4} │ {records:<8} │ {time_est:<8} │ {usage:<10} │")
    
    print("└──────────────────┴──────┴──────────┴──────────┴────────────┘")
    
    print("\n💡 建议选择策略:")
    print("🚀 首次使用: 选择 'quick' (7天) 快速测试")
    print("📊 正常分析: 选择 'medium' (90天) 平衡效率与数据量")
    print("🎯 模型训练: 选择 'full' (365天) 获取最完整数据")

if __name__ == "__main__":
    show_config_examples()
    show_performance_estimates()
    
    # 可选: 运行交互式配置
    run_interactive = input("\n是否运行交互式配置? (y/n): ").strip().lower()
    if run_interactive == 'y':
        interactive_config()
    
    print("\n🎉 配置示例演示完成!")
    print("💡 现在可以根据需要修改 free_historical_data.py 中的配置了") 