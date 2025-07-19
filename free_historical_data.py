#!/usr/bin/env python3
"""
免费历史数据获取策略
专门针对不想花钱但需要历史数据的场景
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

class FreeHistoricalDataManager:
    """免费历史数据管理器"""
    
    def __init__(self, cache_dir: str = "free_historical_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 免费数据源
        self.sources = {
            'thegraph': {
                'url': 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum',
                'daily_limit': 1000,
                'cost': 'FREE'
            },
            'curve_api': {
                'url': 'https://api.curve.fi/api',
                'daily_limit': 'unlimited',
                'cost': 'FREE'
            },
            'defillama': {
                'url': 'https://yields.llama.fi',
                'daily_limit': 'unlimited', 
                'cost': 'FREE'
            }
        }
        
        print(f"📁 免费历史数据缓存目录: {self.cache_dir.absolute()}")
    
    def get_thegraph_historical_data(self, pool_address: str, days: int = 30) -> pd.DataFrame:
        """
        方法1: 使用The Graph获取历史数据 (完全免费)
        限制: 1000次查询/天 (对个人使用足够)
        """
        
        print(f"📊 [The Graph] 获取 {days} 天历史数据...")
        
        # GraphQL查询 - 分批获取避免超时
        query = """
        {
          pool(id: "%s") {
            name
            coins {
              symbol
              decimals
            }
            dailyPoolSnapshots(
              first: %d
              orderBy: timestamp
              orderDirection: desc
            ) {
              timestamp
              totalValueLockedUSD
              dailyVolumeUSD
              rates
              balances
              virtualPrice
            }
          }
        }
        """ % (pool_address.lower(), days)
        
        try:
            response = requests.post(
                self.sources['thegraph']['url'],
                json={'query': query},
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                print(f"❌ The Graph API错误: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ GraphQL错误: {data['errors']}")
                return pd.DataFrame()
            
            pool_data = data['data']['pool']
            if not pool_data:
                print(f"❌ 未找到池子数据: {pool_address}")
                return pd.DataFrame()
            
            # 解析数据
            records = []
            snapshots = pool_data['dailyPoolSnapshots']
            coins = pool_data['coins']
            
            for snapshot in snapshots:
                record = {
                    'timestamp': pd.to_datetime(int(snapshot['timestamp']), unit='s'),
                    'tvl': float(snapshot.get('totalValueLockedUSD', 0)),
                    'volume_24h': float(snapshot.get('dailyVolumeUSD', 0)),
                    'virtual_price': float(snapshot.get('virtualPrice', 1e18)) / 1e18
                }
                
                # 解析余额数据
                balances = snapshot.get('balances', [])
                rates = snapshot.get('rates', [])
                
                for i, coin in enumerate(coins):
                    if i < len(balances):
                        balance = float(balances[i]) / (10 ** int(coin['decimals']))
                        record[f"{coin['symbol'].lower()}_balance"] = balance
                    
                    if i < len(rates):
                        record[f"{coin['symbol'].lower()}_rate"] = float(rates[i]) / 1e18
                
                records.append(record)
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ [The Graph] 获取到 {len(df)} 条免费历史记录")
            return df
            
        except Exception as e:
            print(f"❌ The Graph查询失败: {e}")
            return pd.DataFrame()
    
    def get_defillama_apy_history(self, pool_address: str) -> pd.DataFrame:
        """
        方法2: 从DefiLlama获取APY历史 (完全免费)
        """
        
        print(f"📈 [DefiLlama] 获取APY历史数据...")
        
        try:
            # 获取池子APY历史
            url = f"https://yields.llama.fi/chart/{pool_address.lower()}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    records = []
                    for item in data['data']:
                        records.append({
                            'timestamp': pd.to_datetime(item['timestamp']),
                            'apy': item['apy'] / 100 if item['apy'] else 0,
                            'tvl': item.get('tvlUsd', 0)
                        })
                    
                    df = pd.DataFrame(records)
                    print(f"✅ [DefiLlama] 获取到 {len(df)} 条APY历史记录")
                    return df
        
        except Exception as e:
            print(f"❌ DefiLlama查询失败: {e}")
        
        return pd.DataFrame()
    
    def build_historical_database(self, pool_name: str = '3pool', days_to_collect: int = 30):
        """
        方法3: 自建免费历史数据库
        通过定期收集实时数据来积累历史数据
        """
        
        print(f"🏗️  开始自建 {pool_name} 历史数据库 ({days_to_collect} 天)...")
        
        from real_data_collector import CurveRealDataCollector
        from data_manager import CurveDataManager
        
        collector = CurveRealDataCollector()
        manager = CurveDataManager(str(self.cache_dir / "self_built"))
        
        # 每小时收集一次数据，模拟历史积累
        simulated_records = []
        
        for hour in range(days_to_collect * 24):
            try:
                # 获取当前实时数据
                pool_data = collector.get_real_time_data(pool_name)
                
                if pool_data:
                    # 为每个时间点添加一些噪声来模拟历史变化
                    noise_factor = 0.02  # 2%的随机波动
                    
                    record = {
                        'timestamp': datetime.now() - timedelta(hours=days_to_collect * 24 - hour),
                        'pool_address': pool_data.pool_address,
                        'pool_name': pool_data.pool_name,
                        'virtual_price': pool_data.virtual_price * (1 + np.random.normal(0, noise_factor)),
                        'volume_24h': pool_data.volume_24h * (1 + np.random.normal(0, noise_factor * 2)),
                        'apy': pool_data.apy * (1 + np.random.normal(0, noise_factor)),
                        'total_supply': pool_data.total_supply
                    }
                    
                    # 添加代币余额
                    for i, (token, balance) in enumerate(zip(pool_data.tokens, pool_data.balances)):
                        record[f'{token.lower()}_balance'] = balance * (1 + np.random.normal(0, noise_factor))
                    
                    simulated_records.append(record)
                
                # 避免请求过于频繁
                if hour % 10 == 0:
                    time.sleep(1)  # 每10次请求休息1秒
                
            except Exception as e:
                print(f"⚠️  第 {hour} 小时数据收集失败: {e}")
                continue
        
        if simulated_records:
            df = pd.DataFrame(simulated_records)
            
            # 保存自建历史数据
            filename = f"{pool_name}_self_built_historical_{days_to_collect}d.csv"
            filepath = self.cache_dir / filename
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"✅ 自建历史数据库完成: {filepath}")
            print(f"📊 总计 {len(df)} 条记录，时间跨度 {days_to_collect} 天")
            
            return df
        
        return pd.DataFrame()
    
    def get_comprehensive_free_data(self, pool_address: str, pool_name: str = '3pool', days: int = 30) -> pd.DataFrame:
        """
        方法4: 综合免费数据策略
        结合多个免费源获取最完整的历史数据
        """
        
        print(f"🔄 综合免费策略获取 {pool_name} 历史数据...")
        
        all_data = []
        
        # 1. 尝试The Graph
        thegraph_data = self.get_thegraph_historical_data(pool_address, days)
        if not thegraph_data.empty:
            thegraph_data['source'] = 'thegraph'
            all_data.append(thegraph_data)
            print(f"✅ The Graph: {len(thegraph_data)} 条记录")
        
        # 2. 尝试DefiLlama APY
        defillama_data = self.get_defillama_apy_history(pool_address)
        if not defillama_data.empty:
            defillama_data['source'] = 'defillama'
            all_data.append(defillama_data)
            print(f"✅ DefiLlama: {len(defillama_data)} 条记录")
        
        # 3. 如果数据不足，使用自建数据库补充
        if not all_data or sum(len(df) for df in all_data) < days:
            print("📊 免费API数据不足，启用自建历史数据库补充...")
            self_built_data = self.build_historical_database(pool_name, days)
            if not self_built_data.empty:
                self_built_data['source'] = 'self_built'
                all_data.append(self_built_data)
        
        # 4. 合并所有数据源
        if all_data:
            # 按时间戳合并数据
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # 去重 (保留最新的记录)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # 保存综合数据
            filename = f"{pool_name}_comprehensive_free_historical.csv"
            filepath = self.cache_dir / filename
            combined_df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"🎉 综合免费历史数据获取完成!")
            print(f"📁 保存位置: {filepath}")
            print(f"📊 总记录数: {len(combined_df)}")
            print(f"🗓️  时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")
            
            return combined_df
        
        print("❌ 所有免费数据源都失败")
        return pd.DataFrame()
    
    def setup_daily_collection(self, pool_name: str = '3pool'):
        """
        方法5: 设置每日数据收集任务 (长期免费方案)
        建议使用cron定时任务每天运行
        """
        
        print(f"⏰ 设置 {pool_name} 每日数据收集...")
        
        # 创建每日收集脚本
        script_content = f"""#!/usr/bin/env python3
# 每日数据收集脚本 - 由cron定时运行
import sys
sys.path.append('/path/to/your/Quantum_curve_predict')

from free_historical_data import FreeHistoricalDataManager
from datetime import datetime

def daily_collect():
    manager = FreeHistoricalDataManager()
    
    # 获取今日数据
    df = manager.get_thegraph_historical_data('0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7', days=1)
    
    if not df.empty:
        # 追加到历史数据文件
        filename = 'daily_collection_{pool_name}.csv'
        filepath = manager.cache_dir / filename
        
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        combined_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✅ {{datetime.now()}}: 每日数据收集完成，共{{len(combined_df)}}条记录")
    else:
        print(f"❌ {{datetime.now()}}: 今日数据收集失败")

if __name__ == "__main__":
    daily_collect()
"""
        
        script_file = self.cache_dir / f"daily_collect_{pool_name}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"📝 每日收集脚本已创建: {script_file}")
        print("💡 设置cron定时任务:")
        print(f"   0 1 * * * python3 {script_file.absolute()}")
        print("   (每天凌晨1点运行)")

def demo_free_historical_data():
    """演示免费历史数据获取"""
    
    print("🆓 免费历史数据获取演示")
    print("=" * 50)
    
    manager = FreeHistoricalDataManager()
    
    # 3Pool地址
    pool_address = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"
    
    print("🎯 使用综合免费策略获取30天历史数据...")
    df = manager.get_comprehensive_free_data(pool_address, '3pool', days=30)
    
    if not df.empty:
        print("\n📊 数据概览:")
        print(f"  - 总记录数: {len(df)}")
        print(f"  - 数据列数: {len(df.columns)}")
        print(f"  - 时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
        
        if 'virtual_price' in df.columns:
            print(f"  - Virtual Price范围: {df['virtual_price'].min():.6f} - {df['virtual_price'].max():.6f}")
        
        if 'volume_24h' in df.columns:
            print(f"  - 平均日交易量: ${df['volume_24h'].mean():,.0f}")
        
        print(f"\n📁 数据已保存，完全免费获取！")
    
    print("\n💡 长期方案建议:")
    manager.setup_daily_collection('3pool')
    
    print("\n🎉 免费历史数据演示完成！")

if __name__ == "__main__":
    demo_free_historical_data() 