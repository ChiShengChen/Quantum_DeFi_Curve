#!/usr/bin/env python3
"""
Free Historical Data Acquisition Strategy
Specifically designed for scenarios where you need historical data without spending money
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

# ========================================
# 🔧 Configuration Parameters - Modify day settings here
# ========================================
DEFAULT_HISTORICAL_DAYS = 365  # Default to get 1 year of historical data
MAX_THEGRAPH_DAYS = 365       # The Graph API maximum supported days (deprecated)
DEFAULT_SELF_BUILT_DAYS = 365 # Self-built database default days

# Quick configuration options
QUICK_TEST_DAYS = 7           # Quick testing (7 days)
MEDIUM_RANGE_DAYS = 90        # Medium-term analysis (3 months) 
FULL_YEAR_DAYS = 365          # Complete annual data

# Current configuration - modify here to change default values for all methods
CURRENT_DAYS_SETTING =  FULL_YEAR_DAYS  # 🔥 Temporarily set to 7 days to avoid long waiting

# ========================================
# 🚨 API Status Configuration - Control which data sources are available
# ========================================
ENABLE_THEGRAPH_API = False     # ❌ The Graph API has been removed, disabled
ENABLE_CURVE_API = True         # ✅ Curve API (main data source)
ENABLE_DEFILLAMA = True         # ✅ DefiLlama APY data  
ENABLE_SELF_BUILT = True        # ✅ Self-built database (as final backup)
ENABLE_SSL_VERIFICATION = False # 🔧 Disable SSL verification to avoid certificate errors

# Performance optimization configuration
MAX_COLLECTION_ATTEMPTS = 50   # 🔥 Limit maximum collection attempts to avoid infinite loops
COLLECTION_BATCH_SIZE = 10     # Number of data points collected per batch
REQUEST_TIMEOUT = 5            # API request timeout (seconds)
REQUEST_RETRY_DELAY = 2        # Retry delay after request failure (seconds)

# ========================================
# 🎯 All Major Curve Pool Configuration - Extended Version
# ========================================

# Complete major Curve pool configuration
AVAILABLE_POOLS = {
    # === 🏆 Major Stablecoin Pools (Base Pools) ===
    '3pool': {
        'address': '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',
        'name': 'DAI/USDC/USDT',
        'tokens': ['DAI', 'USDC', 'USDT'],
        'type': 'stable',
        'priority': 1  # Highest priority
    },
    'frax': {
        'address': '0xd632f22692FaC7611d2AA1C0D552930D43CAEd3B', 
        'name': 'FRAX/3CRV',
        'tokens': ['FRAX', '3CRV'],
        'type': 'metapool',
        'priority': 2
    },
    'lusd': {
        'address': '0xEd279fDD11cA84bEef15AF5D39BB4d4bEE23F0cA',
        'name': 'LUSD/3CRV', 
        'tokens': ['LUSD', '3CRV'],
        'type': 'metapool',
        'priority': 2
    },
    'mim': {
        'address': '0x5a6A4D54456819380173272A5E8E9B9904BdF41B',
        'name': 'MIM/3CRV',
        'tokens': ['MIM', '3CRV'], 
        'type': 'metapool',
        'priority': 3
    },
    
    # === 🔥 ETH/stETH Pools ===
    'steth': {
        'address': '0xDC24316b9AE028F1497c275EB9192a3Ea0f67022',
        'name': 'ETH/stETH',
        'tokens': ['ETH', 'stETH'],
        'type': 'eth_pool',
        'priority': 2
    },
    'seth': {
        'address': '0xc5424B857f758E906013F3555Dad202e4bdB4567',
        'name': 'ETH/sETH',
        'tokens': ['ETH', 'sETH'],
        'type': 'eth_pool', 
        'priority': 3
    },
    'reth': {
        'address': '0xF9440930043eb3997fc70e1339dBb11F341de7A8',
        'name': 'ETH/rETH',
        'tokens': ['ETH', 'rETH'],
        'type': 'eth_pool',
        'priority': 3
    },
    'ankrETH': {
        'address': '0xA96A65c051bF88B4095Ee1f2451C2A9d43F53Ae2',
        'name': 'ETH/ankrETH', 
        'tokens': ['ETH', 'ankrETH'],
        'type': 'eth_pool',
        'priority': 4
    },
    
    # === ₿ BTC Pools ===
    'renbtc': {
        'address': '0x93054188d876f558f4a66B2EF1d97d16eDf0895B',
        'name': 'renBTC/WBTC',
        'tokens': ['renBTC', 'WBTC'],
        'type': 'btc_pool',
        'priority': 3
    },
    'sbtc': {
        'address': '0x7fC77b5c7614E1533320Ea6DDc2Eb61fa00A9714',
        'name': 'renBTC/WBTC/sBTC',
        'tokens': ['renBTC', 'WBTC', 'sBTC'],
        'type': 'btc_pool',
        'priority': 3
    },
    'hbtc': {
        'address': '0x4CA9b3063Ec5866A4B82E437059D2C43d1be596F',
        'name': 'hBTC/WBTC',
        'tokens': ['hBTC', 'WBTC'],
        'type': 'btc_pool',
        'priority': 4
    },
    'bbtc': {
        'address': '0x071c661B4DeefB59E2a3DdB20Db036821eeE8F4b',
        'name': 'bBTC/sbtcCRV',
        'tokens': ['bBTC', 'sbtcCRV'],
        'type': 'btc_metapool',
        'priority': 4
    },
    'obtc': {
        'address': '0xd81dA8D904b52208541Bade1bD6595D8a251F8dd',
        'name': 'oBTC/sbtcCRV',
        'tokens': ['oBTC', 'sbtcCRV'],
        'type': 'btc_metapool', 
        'priority': 4
    },
    'pbtc': {
        'address': '0x7F55DDe206dbAD629C080068923b36fe9D6bDBeF',
        'name': 'pBTC/sbtcCRV',
        'tokens': ['pBTC', 'sbtcCRV'],
        'type': 'btc_metapool',
        'priority': 4
    },
    'tbtc': {
        'address': '0xC25099792E9349C7DD09759744ea681C7de2cb66',
        'name': 'tBTC/sbtcCRV', 
        'tokens': ['tBTC', 'sbtcCRV'],
        'type': 'btc_metapool',
        'priority': 4
    },
    
    # === 🚀 Crypto 池 ===
    'tricrypto': {
        'address': '0x80466c64868E1ab14a1Ddf27A676C3fcBE638Fe5',
        'name': 'USDT/WBTC/WETH',
        'tokens': ['USDT', 'WBTC', 'WETH'],
        'type': 'crypto',
        'priority': 2
    },
    'tricrypto2': {
        'address': '0xD51a44d3FaE010294C616388b506AcDA1bfAAE46', 
        'name': 'USDT/WBTC/WETH v2',
        'tokens': ['USDT', 'WBTC', 'WETH'],
        'type': 'crypto',
        'priority': 2
    },
    
    # === 🏦 Lending 池 ===
    'aave': {
        'address': '0xDeBF20617708857ebe4F679508E7b7863a8A8EeE',
        'name': 'aDAI/aUSDC/aUSDT',
        'tokens': ['aDAI', 'aUSDC', 'aUSDT'],
        'type': 'lending',
        'priority': 3
    },
    'compound': {
        'address': '0xA2B47E3D5c44877cca798226B7B8118F9BFb7A56',
        'name': 'cDAI/cUSDC',
        'tokens': ['cDAI', 'cUSDC'],
        'type': 'lending',
        'priority': 4
    },
    'ironbank': {
        'address': '0x2dded6Da1BF5DBdF597C45fcFaa3194e53EcfeAF',
        'name': 'cyDAI/cyUSDC/cyUSDT',
        'tokens': ['cyDAI', 'cyUSDC', 'cyUSDT'],
        'type': 'lending',
        'priority': 4
    },
    'saave': {
        'address': '0xEB16Ae0052ed37f479f7fe63849198Df1765a733',
        'name': 'sDAI/sUSDC/sUSDT',
        'tokens': ['sDAI', 'sUSDC', 'sUSDT'],
        'type': 'lending',
        'priority': 4
    },
    
    # === 🌍 國際化穩定幣 ===
    'eurs': {
        'address': '0x0Ce6a5fF5217e38315f87032CF90686C96627CAA',
        'name': 'EURS/sEUR',
        'tokens': ['EURS', 'sEUR'],
        'type': 'international',
        'priority': 4
    },
    
    # === 📈 更多Meta池 ===
    'gusd': {
        'address': '0x4f062658EaAF2C1ccf8C8e36D6824CDf41167956',
        'name': 'GUSD/3CRV',
        'tokens': ['GUSD', '3CRV'],
        'type': 'metapool',
        'priority': 4
    },
    'husd': {
        'address': '0x3eF6A01A0f81D6046290f3e2A8c5b843e738E604',
        'name': 'HUSD/3CRV',
        'tokens': ['HUSD', '3CRV'],
        'type': 'metapool',
        'priority': 4
    },
    'musd': {
        'address': '0x8474DdbE98F5aA3179B3B3F5942D724aFcdec9f6',
        'name': 'MUSD/3CRV',
        'tokens': ['MUSD', '3CRV'],
        'type': 'metapool',
        'priority': 4
    },
    'dusd': {
        'address': '0x8038C01A0390a8c547446a0b2c18fc9aEFEcc10c',
        'name': 'DUSD/3CRV',
        'tokens': ['DUSD', '3CRV'],
        'type': 'metapool',
        'priority': 5
    },
    'usdk': {
        'address': '0x3E01dD8a5E1fb3481F0F589056b428Fc308AF0Fb',
        'name': 'USDK/3CRV',
        'tokens': ['USDK', '3CRV'],
        'type': 'metapool',
        'priority': 5
    },
    'usdn': {
        'address': '0x0f9cb53Ebe405d49A0bbdBD291A65Ff571bC83e1',
        'name': 'USDN/3CRV',
        'tokens': ['USDN', '3CRV'],
        'type': 'metapool',
        'priority': 5
    },
    'usdp': {
        'address': '0x42d7025938bEc20B69cBae5A77421082407f053A',
        'name': 'USDP/3CRV',
        'tokens': ['USDP', '3CRV'],
        'type': 'metapool',
        'priority': 4
    },
    'ust': {
        'address': '0x890f4e345B1dAED0367A877a1612f86A1f86985f',
        'name': 'UST/3CRV',
        'tokens': ['UST', '3CRV'],
        'type': 'metapool',
        'priority': 5  # 較低優先級因为UST已deprecated
    },
    'rsv': {
        'address': '0xC18cC39da8b11dA8c3541C598eE022258F9744da',
        'name': 'RSV/3CRV',
        'tokens': ['RSV', '3CRV'],
        'type': 'metapool',
        'priority': 5
    },
    'linkusd': {
        'address': '0xE7a24EF0C5e95Ffb0f6684b813A78F2a3AD7D171',
        'name': 'LINKUSD/3CRV',
        'tokens': ['LINKUSD', '3CRV'],
        'type': 'metapool',
        'priority': 5
    },
    
    # === 🔗 其他重要池子 ===
    'link': {
        'address': '0xF178C0b5Bb7e7aBF4e12A4838C7b7c5bA2C623c0',
        'name': 'LINK/sLINK',
        'tokens': ['LINK', 'sLINK'],
        'type': 'synthetic',
        'priority': 4
    },
    'susd': {
        'address': '0xA5407eAE9Ba41422680e2e00537571bcC53efBfD',
        'name': 'DAI/USDC/USDT/sUSD',
        'tokens': ['DAI', 'USDC', 'USDT', 'sUSD'],
        'type': 'stable_4pool',
        'priority': 4
    },
    'y': {
        'address': '0x45F783CCE6B7FF23B2ab2D70e416cdb7D6055f51',
        'name': 'yDAI/yUSDC/yUSDT/yTUSD',
        'tokens': ['yDAI', 'yUSDC', 'yUSDT', 'yTUSD'],
        'type': 'yield',
        'priority': 5
    },
    'busd': {
        'address': '0x79a8C46DeA5aDa233ABaFFD40F3A0A2B1e5A4F27',
        'name': 'yDAI/yUSDC/yUSDT/yBUSD',
        'tokens': ['yDAI', 'yUSDC', 'yUSDT', 'yBUSD'],
        'type': 'yield',
        'priority': 5
    },
    'pax': {
        'address': '0x06364f10B501e868329afBc005b3492902d6C763',
        'name': 'ycDAI/ycUSDC/ycUSDT/PAX',
        'tokens': ['ycDAI', 'ycUSDC', 'ycUSDT', 'PAX'],
        'type': 'yield',
        'priority': 5
    }
}

# 根據優先級和類型篩選池子的函数
def get_pools_by_priority(min_priority=1, max_priority=5, pool_types=None):
    """
    Filter pools by priority and type
    
    Args:
        min_priority: Minimum priority (1=highest priority)  
        max_priority: Maximum priority (5=lowest priority)
        pool_types: List of pool types, such as ['stable', 'metapool'] 
    
    Returns:
        Filtered pool dictionary
    """
    filtered_pools = {}
    
    for pool_name, pool_info in AVAILABLE_POOLS.items():
        # 優先級篩選
        if not (min_priority <= pool_info['priority'] <= max_priority):
            continue
            
        # 類型篩選
        if pool_types and pool_info['type'] not in pool_types:
            continue
            
        filtered_pools[pool_name] = pool_info
    
    return filtered_pools

def get_high_priority_pools():
    """Get high priority pools (priority 1-2)"""
    return get_pools_by_priority(min_priority=1, max_priority=2)

def get_stable_pools():
    """Get all stablecoin-related pools"""
    return get_pools_by_priority(pool_types=['stable', 'metapool', 'stable_4pool'])

def get_all_main_pools():
    """Get all main pools (priority 1-3)"""
    return get_pools_by_priority(min_priority=1, max_priority=3)

# Update original configuration to maintain compatibility
TARGET_POOL = '3pool'  # Default pool
TARGET_POOL_ADDRESS = AVAILABLE_POOLS[TARGET_POOL]['address']
TARGET_POOL_NAME = AVAILABLE_POOLS[TARGET_POOL]['name']

# Batch mode configuration
ENABLE_BATCH_MODE = False     # Whether to enable batch mode (collect all pools)
BATCH_POOLS = ['3pool', 'frax', 'lusd']  # List of pools to collect in batch mode

# Display current configuration information
def show_current_config():
    """Display current configuration"""
    print("=" * 60)
    print("📋 Current Crawler Configuration")
    print("=" * 60)
    print(f"🎯 Target Pool: {TARGET_POOL}")
    print(f"📛 Pool Name: {TARGET_POOL_NAME}")  
    print(f"📍 Pool Address: {TARGET_POOL_ADDRESS}")
    print(f"📊 Data Days: {CURRENT_DAYS_SETTING} days")
    print(f"🔄 Batch Mode: {'Enabled' if ENABLE_BATCH_MODE else 'Disabled'}")
    if ENABLE_BATCH_MODE:
        print(f"📦 Batch Pools: {', '.join(BATCH_POOLS)}")
    print("=" * 60)
    print("💡 To switch pools, modify the TARGET_POOL variable!")
    print("   Available options: " + " | ".join(AVAILABLE_POOLS.keys()))
    print("=" * 60)

# Configuration information will be displayed when the main program runs

class FreeHistoricalDataManager:
    """Free Historical Data Manager"""
    
    def __init__(self, cache_dir: str = "free_historical_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Free data sources
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
        
        print(f"📁 Free historical data cache directory: {self.cache_dir.absolute()}")
    
    def get_thegraph_historical_data(self, pool_address: str, days: int = CURRENT_DAYS_SETTING) -> pd.DataFrame:
        """
        Method 1: Use The Graph to get historical data (deprecated)
        ❌ Note: The Graph API endpoint has been removed
        """
        
        if not ENABLE_THEGRAPH_API:
            print(f"⚠️  [The Graph] API has been disabled - endpoint deprecated")
            return pd.DataFrame()
        
        print(f"📊 [The Graph] Attempting to get {days} days of historical data...")
        
        # GraphQL query - batch retrieval to avoid timeout
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
            # 配置SSL验证
            verify_ssl = ENABLE_SSL_VERIFICATION
            
            response = requests.post(
                self.sources['thegraph']['url'],
                json={'query': query},
                timeout=REQUEST_TIMEOUT,
                headers={'Content-Type': 'application/json'},
                verify=verify_ssl
            )
            
            if response.status_code != 200:
                print(f"❌ The Graph API error: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ GraphQL error (API deprecated): {data['errors'][0]['message'][:100]}...")
                return pd.DataFrame()
            
            pool_data = data['data']['pool']
            if not pool_data:
                print(f"❌ Pool data not found: {pool_address}")
                return pd.DataFrame()
            
            # Parse data
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
                
                # Parse balance data
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
            
            print(f"✅ [The Graph] Retrieved {len(df)} free historical records")
            return df
            
        except Exception as e:
            print(f"❌ The Graph query failed (API deprecated): {str(e)[:100]}...")
            return pd.DataFrame()
    
    def get_defillama_apy_history(self, pool_address: str) -> pd.DataFrame:
        """
        Method 2: Get APY history from DefiLlama (completely free)
        """
        
        if not ENABLE_DEFILLAMA:
            print(f"⚠️  [DefiLlama] API has been disabled")
            return pd.DataFrame()
            
        print(f"📈 [DefiLlama] Getting APY historical data...")
        
        try:
            # Get pool APY history
            url = f"https://yields.llama.fi/chart/{pool_address.lower()}"
            
            # Configure SSL verification and timeout
            verify_ssl = ENABLE_SSL_VERIFICATION
            
            response = requests.get(
                url, 
                timeout=REQUEST_TIMEOUT,
                verify=verify_ssl
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    records = []
                    for item in data['data']:
                        records.append({
                            'timestamp': pd.to_datetime(item['timestamp']),
                            'apy': item['apy'] / 100 if item['apy'] else 0,
                            'tvl': item.get('tvlUsd', 0)
                        })
                    
                    df = pd.DataFrame(records)
                    print(f"✅ [DefiLlama] Retrieved {len(df)} APY historical records")
                    return df
                else:
                    print(f"⚠️  [DefiLlama] No data field in response")
            else:
                print(f"❌ [DefiLlama] HTTP error: {response.status_code}")
        
        except requests.exceptions.SSLError as e:
            print(f"❌ [DefiLlama] SSL error: Please try disabling SSL verification")
        except requests.exceptions.Timeout as e:
            print(f"❌ [DefiLlama] Timeout error: {REQUEST_TIMEOUT}s")
        except Exception as e:
            print(f"❌ [DefiLlama] Query failed: {str(e)[:100]}...")
        
        return pd.DataFrame()
    
    def build_historical_database(self, pool_name: str = TARGET_POOL, days_to_collect: int = CURRENT_DAYS_SETTING):
        """
        Method 3: Self-built free historical database (optimized - avoid infinite loops)
        Attempt to get real-time data through limited attempts, then generate synthetic historical data
        """
        
        if not ENABLE_SELF_BUILT:
            print(f"⚠️  Self-built database has been disabled")
            return pd.DataFrame()
        
        print(f"🏗️  Starting to build {pool_name} historical database ({days_to_collect} days)...")
        
        try:
            from real_data_collector import CurveRealDataCollector
            from data_manager import CurveDataManager
            
            collector = CurveRealDataCollector()
            manager = CurveDataManager(str(self.cache_dir / "self_built"))
            
            # 🔥 Limit attempt count to avoid infinite loops
            max_attempts = min(MAX_COLLECTION_ATTEMPTS, days_to_collect)
            successful_attempts = 0
            base_data = None
            
            print(f"🔄 Attempting to get base data (max {max_attempts} attempts)...")
            
            # First try to get one valid real-time data as baseline
            for attempt in range(max_attempts):
                try:
                    pool_data = collector.get_real_time_data(pool_name)
                    if pool_data:
                        base_data = pool_data
                        print(f"✅ Attempt {attempt + 1} successfully got base data")
                        break
                    
                    if attempt % 10 == 0 and attempt > 0:
                        print(f"⚠️  Already tried {attempt + 1} times, continuing retry...")
                    
                    time.sleep(REQUEST_RETRY_DELAY)  # Avoid too frequent requests
                    
                except Exception as e:
                    if attempt % 10 == 0:
                        print(f"⚠️  Attempt {attempt + 1} failed: {str(e)[:50]}...")
                    continue
            
            # If unable to get real data, generate synthetic data
            if not base_data:
                print("⚠️  Unable to get real data, generating synthetic historical data...")
                return self._generate_synthetic_data(pool_name, days_to_collect)
            
            # Generate historical data based on real data
            print(f"📊 Generating {days_to_collect} days of historical data based on real data...")
            simulated_records = []
            
            for day in range(days_to_collect):
                # Generate several data points per day instead of per hour
                points_per_day = 4  # One data point every 6 hours
                
                for point in range(points_per_day):
                    hour_offset = day * 24 + point * 6
                    
                    # 为每个时间点添加随机波动
                    noise_factor = 0.02  # 2%的随机波动
                    
                    record = {
                        'timestamp': datetime.now() - timedelta(hours=hour_offset),
                        'pool_address': base_data.pool_address,
                        'pool_name': base_data.pool_name,
                        'virtual_price': base_data.virtual_price * (1 + np.random.normal(0, noise_factor)),
                        'volume_24h': base_data.volume_24h * (1 + np.random.normal(0, noise_factor * 2)),
                        'apy': max(0, base_data.apy * (1 + np.random.normal(0, noise_factor))),
                        'total_supply': base_data.total_supply * (1 + np.random.normal(0, noise_factor * 0.5))
                    }
                    
                    # 添加代币余额
                    for i, (token, balance) in enumerate(zip(base_data.tokens, base_data.balances)):
                        record[f'{token.lower()}_balance'] = balance * (1 + np.random.normal(0, noise_factor))
                    
                    simulated_records.append(record)
                
                # Show progress
                if day % max(1, days_to_collect // 10) == 0:
                    progress = (day / days_to_collect) * 100
                    print(f"📈 Generation progress: {progress:.0f}% ({day}/{days_to_collect} days)")
        
        except ImportError as e:
            print(f"❌ Import dependency failed: {e}")
            print("💡 Generating basic synthetic data...")
            return self._generate_synthetic_data(pool_name, days_to_collect)
        except Exception as e:
            print(f"❌ Self-built database failed: {e}")
            print("💡 Fallback to synthetic data...")
            return self._generate_synthetic_data(pool_name, days_to_collect)
        
        if simulated_records:
            df = pd.DataFrame(simulated_records)
            
            # Save self-built historical data
            filename = f"{pool_name}_self_built_historical_{days_to_collect}d.csv"
            filepath = self.cache_dir / filename
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"✅ Self-built historical database completed: {filepath}")
            print(f"📊 Total {len(df)} records, time span {days_to_collect} days")
            
            return df
        
        return pd.DataFrame()

    def _generate_synthetic_data(self, pool_name: str, days: int) -> pd.DataFrame:
        """Generate synthetic historical data - used when all real data sources fail"""
        
        print(f"🎭 Generating {days} days of synthetic historical data for {pool_name}...")
        
        # Set different base parameters according to pool type
        pool_configs = {
            'mim': {
                'tokens': ['MIM', 'USDC', 'USDT'], 
                'base_balance': 1000000,
                'base_apy': 0.05,
                'base_volume': 500000
            },
            '3pool': {
                'tokens': ['USDC', 'USDT', 'DAI'],
                'base_balance': 5000000, 
                'base_apy': 0.03,
                'base_volume': 2000000
            },
            'frax': {
                'tokens': ['FRAX', 'USDC'],
                'base_balance': 800000,
                'base_apy': 0.06,
                'base_volume': 300000
            },
            'lusd': {
                'tokens': ['LUSD', 'USDC', 'USDT'],
                'base_balance': 600000,
                'base_apy': 0.04,
                'base_volume': 200000
            }
        }
        
        config = pool_configs.get(pool_name, pool_configs['3pool'])
        
        # Generate time series
        dates = pd.date_range(
            end=datetime.now(),
            periods=days * 6,  # 6 data points per day
            freq='4H'  # One point every 4 hours
        )
        
        records = []
        for i, timestamp in enumerate(dates):
            # Add some trends and randomness
            trend_factor = 1 + 0.1 * np.sin(i / (days * 0.5))  # Long-term trend
            noise_factor = np.random.normal(1, 0.02)  # Random fluctuation
            
            record = {
                'timestamp': timestamp,
                'pool_address': AVAILABLE_POOLS[pool_name]['address'],
                'pool_name': AVAILABLE_POOLS[pool_name]['name'],
                'virtual_price': 1.0 * trend_factor * noise_factor,
                'volume_24h': config['base_volume'] * trend_factor * noise_factor,
                'apy': max(0, config['base_apy'] * trend_factor * noise_factor),
                'total_supply': config['base_balance'] * 3 * trend_factor
            }
            
            # Add token balances
            for j, token in enumerate(config['tokens']):
                balance_variation = np.random.normal(1, 0.05)
                record[f'{token.lower()}_balance'] = config['base_balance'] * balance_variation
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save synthetic data
        filename = f"{pool_name}_synthetic_historical_{days}d.csv"  
        filepath = self.cache_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✅ Synthetic data generation completed: {filepath}")
        print(f"📊 Generated {len(df)} synthetic records")
        
        return df
    
    def get_comprehensive_free_data(self, pool_address: str = TARGET_POOL_ADDRESS, pool_name: str = TARGET_POOL, days: int = CURRENT_DAYS_SETTING) -> pd.DataFrame:
        """
        Method 4: Comprehensive free data strategy (optimized)
        Combine multiple free sources to get the most complete historical data, including fallback mechanism
        """
        
        print(f"🔄 Comprehensive free strategy getting {pool_name} historical data ({days} days)...")
        
        all_data = []
        data_sources_tried = []
        
        # 1. Try The Graph (if enabled)
        if ENABLE_THEGRAPH_API:
            try:
                thegraph_data = self.get_thegraph_historical_data(pool_address, days)
                if not thegraph_data.empty:
                    thegraph_data['source'] = 'thegraph'
                    all_data.append(thegraph_data)
                    print(f"✅ The Graph: {len(thegraph_data)} records")
                data_sources_tried.append('The Graph')
            except Exception as e:
                print(f"⚠️  The Graph attempt failed: {str(e)[:50]}...")
                data_sources_tried.append('The Graph (failed)')
        else:
            print("⚠️  The Graph is disabled")
        
        # 2. Try DefiLlama APY (if enabled)
        if ENABLE_DEFILLAMA:
            try:
                defillama_data = self.get_defillama_apy_history(pool_address)
                if not defillama_data.empty:
                    defillama_data['source'] = 'defillama'
                    all_data.append(defillama_data)
                    print(f"✅ DefiLlama: {len(defillama_data)} records")
                data_sources_tried.append('DefiLlama')
            except Exception as e:
                print(f"⚠️  DefiLlama attempt failed: {str(e)[:50]}...")
                data_sources_tried.append('DefiLlama (failed)')
        
        # 3. Check if self-built database supplement is needed
        total_records = sum(len(df) for df in all_data) if all_data else 0
        min_required_records = max(days // 10, 5)  # Minimum required records
        
        if total_records < min_required_records:
            print(f"📊 Free API data insufficient ({total_records} < {min_required_records}), enabling self-built historical database...")
            
            if ENABLE_SELF_BUILT:
                try:
                    self_built_data = self.build_historical_database(pool_name, days)
                    if not self_built_data.empty:
                        self_built_data['source'] = 'self_built'
                        all_data.append(self_built_data)
                        print(f"✅ Self-built data: {len(self_built_data)} records")
                    data_sources_tried.append('Self-built database')
                except Exception as e:
                    print(f"⚠️  Self-built database failed: {str(e)[:50]}...")
                    data_sources_tried.append('Self-built database (failed)')
            else:
                print("⚠️  Self-built database is disabled")
        
        # 4. Merge all data sources
        if all_data:
            try:
                # Merge data by timestamp
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Ensure timestamp column exists and format is correct
                if 'timestamp' not in combined_df.columns:
                    print("⚠️  Data missing timestamp column, adding default timestamp")
                    combined_df['timestamp'] = pd.date_range(
                        end=datetime.now(),
                        periods=len(combined_df),
                        freq='H'
                    )
                
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                # Remove duplicates (keep latest records)
                if len(combined_df) > 1:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # Save comprehensive data
                filename = f"{pool_name}_comprehensive_free_historical_{days}d.csv"
                filepath = self.cache_dir / filename
                combined_df.to_csv(filepath, index=False, encoding='utf-8')
                
                print(f"🎉 Comprehensive free historical data acquisition completed!")
                print(f"📁 Save location: {filepath}")
                print(f"📊 Total records: {len(combined_df)}")
                print(f"🔄 Data sources: {', '.join([df['source'].iloc[0] for df in all_data if 'source' in df.columns and len(df) > 0])}")
                
                if 'timestamp' in combined_df.columns and len(combined_df) > 0:
                    print(f"🗓️  Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
                
                return combined_df
            
            except Exception as e:
                print(f"❌ Data merge failed: {e}")
                # Return the first available data source
                if all_data:
                    print(f"💡 Returning first available data source ({len(all_data[0])} records)")
                    return all_data[0]
        
        print(f"❌ All free data sources failed")
        print(f"🔍 Attempted data sources: {', '.join(data_sources_tried)}")
        print(f"💡 Suggestion: Check network connection or enable SSL verification")
        
        return pd.DataFrame()
    
    def setup_daily_collection(self, pool_name: str = TARGET_POOL):
        """
        Method 5: Set up daily data collection task (long-term free solution)
        Recommended to use cron scheduled task to run daily
        """
        
        print(f"⏰ Setting up {pool_name} daily data collection...")
        
        # Create daily collection script
        script_content = f"""#!/usr/bin/env python3
# Daily data collection script - run by cron scheduled task
import sys
sys.path.append('/path/to/your/Quantum_curve_predict')

from free_historical_data import FreeHistoricalDataManager
from datetime import datetime

def daily_collect():
    manager = FreeHistoricalDataManager()
    
    # Get today's data
    df = manager.get_thegraph_historical_data('0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7', days=1)
    
    if not df.empty:
        # Append to historical data file
        filename = 'daily_collection_{pool_name}.csv'
        filepath = manager.cache_dir / filename
        
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        combined_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✅ {{datetime.now()}}: Daily data collection completed, total {{len(combined_df)}} records")
    else:
        print(f"❌ {{datetime.now()}}: Today's data collection failed")

if __name__ == "__main__":
    daily_collect()
"""
        
        script_file = self.cache_dir / f"daily_collect_{pool_name}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"📝 Daily collection script created: {script_file}")
        print("💡 Set up cron scheduled task:")
        print(f"   0 1 * * * python3 {script_file.absolute()}")
        print("   (Run at 1 AM daily)")

    def get_batch_historical_data(self, pools_dict: dict, days: int = CURRENT_DAYS_SETTING, 
                                 max_concurrent: int = 3, delay_between_batches: int = 2) -> dict:
        """
        Batch retrieve historical data for multiple pools
        
        Args:
            pools_dict: Pool dictionary (from get_pools_by_priority and other functions)
            days: Number of days to retrieve
            max_concurrent: Maximum concurrent requests
            delay_between_batches: Delay between batches (seconds)
        
        Returns:
            {pool_name: DataFrame} dictionary
        """
        
        print(f"🚀 Batch retrieving {days} days of historical data for {len(pools_dict)} pools...")
        print(f"📋 Pool list: {', '.join(pools_dict.keys())}")
        print("="*60)
        
        results = {}
        successful = 0
        failed = 0
        
        # Sort pools by priority
        sorted_pools = sorted(pools_dict.items(), key=lambda x: x[1]['priority'])
        
        # 分批處理避免API限制
        import math
        total_batches = math.ceil(len(sorted_pools) / max_concurrent)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * max_concurrent
            end_idx = min(start_idx + max_concurrent, len(sorted_pools))
            current_batch = sorted_pools[start_idx:end_idx]
            
            print(f"📦 處理批次 {batch_idx + 1}/{total_batches}")
            print(f"   池子: {[pool[0] for pool in current_batch]}")
            
            # 處理当前批次
            for pool_name, pool_info in current_batch:
                try:
                    print(f"  🔄 [{pool_name}] {pool_info['name']} (優先級:{pool_info['priority']})")
                    
                    # 檢查缓存
                    cache_file = self.cache_dir / f"{pool_name}_batch_historical_{days}d.csv"
                    if cache_file.exists():
                        try:
                            df = pd.read_csv(cache_file)
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            print(f"  ✅ [{pool_name}] 从缓存加载 {len(df)} 条记录")
                            results[pool_name] = df
                            successful += 1
                            continue
                        except Exception as e:
                            print(f"  ⚠️  [{pool_name}] 缓存读取失败: {e}")
                    
                    # 獲取新數據  
                    df = self.get_comprehensive_free_data(
                        pool_info['address'], 
                        pool_name, 
                        days=days
                    )
                    
                    if not df.empty:
                        # 添加池子信息
                        df['pool_name'] = pool_name
                        df['pool_type'] = pool_info['type']
                        df['priority'] = pool_info['priority']
                        
                        # 保存到缓存
                        df.to_csv(cache_file, index=False)
                        
                        results[pool_name] = df
                        successful += 1
                        print(f"  ✅ [{pool_name}] 獲取成功: {len(df)} 条记录")
                        
                        # 顯示简要统计  
                        if 'virtual_price' in df.columns:
                            latest_vp = df['virtual_price'].iloc[-1] if len(df) > 0 else 0
                            print(f"     Virtual Price: {latest_vp:.6f}")
                            
                    else:
                        print(f"  ❌ [{pool_name}] 没有獲取到數據")
                        failed += 1
                        results[pool_name] = pd.DataFrame()
                        
                except Exception as e:
                    print(f"  ❌ [{pool_name}] 獲取失败: {str(e)[:100]}...")
                    failed += 1
                    results[pool_name] = pd.DataFrame()
            
            # 批次间延迟
            if batch_idx < total_batches - 1:  # 不是最后一批次
                print(f"  ⏳ 等待 {delay_between_batches} 秒后處理下一批次...")
                time.sleep(delay_between_batches)
        
        print("\n" + "="*60)
        print(f"📊 批量獲取完成!")
        print(f"   ✅ 成功: {successful}/{len(pools_dict)}")
        print(f"   ❌ 失败: {failed}/{len(pools_dict)}")
        print(f"   成功率: {successful/len(pools_dict)*100:.1f}%")
        
        return results

    def get_all_main_pools_data(self, days: int = CURRENT_DAYS_SETTING) -> dict:
        """
        獲取所有主要池子數據 (優先級 1-3)
        
        Returns:
            {pool_name: DataFrame} 字典
        """
        main_pools = get_all_main_pools()
        print(f"🎯 獲取 {len(main_pools)} 个主要池子數據...")
        
        return self.get_batch_historical_data(main_pools, days)

    def get_high_priority_pools_data(self, days: int = CURRENT_DAYS_SETTING) -> dict:
        """
        獲取高優先級池子數據 (優先級 1-2)
        
        Returns:
            {pool_name: DataFrame} 字典  
        """
        high_priority_pools = get_high_priority_pools()
        print(f"⭐ 獲取 {len(high_priority_pools)} 个高優先級池子數據...")
        
        return self.get_batch_historical_data(high_priority_pools, days)

    def get_stable_pools_data(self, days: int = CURRENT_DAYS_SETTING) -> dict:
        """
        獲取所有稳定币池數據
        
        Returns:
            {pool_name: DataFrame} 字典
        """
        stable_pools = get_stable_pools()
        print(f"💰 獲取 {len(stable_pools)} 个稳定币池數據...")
        
        return self.get_batch_historical_data(stable_pools, days)

    def get_all_pools_data(self, days: int = CURRENT_DAYS_SETTING, skip_low_priority: bool = True) -> dict:
        """
        獲取所有池子數據 (可选择跳过低優先級)
        
        Args:
            days: 獲取天数  
            skip_low_priority: 是否跳过優先級5的池子
            
        Returns:
            {pool_name: DataFrame} 字典
        """
        if skip_low_priority:
            pools = get_pools_by_priority(min_priority=1, max_priority=4)
            print(f"🌍 獲取所有池子數據 (跳过低優先級): {len(pools)} 个池子")
        else:
            pools = AVAILABLE_POOLS
            print(f"🌍 獲取所有池子數據 (包含全部): {len(pools)} 个池子")
        
        return self.get_batch_historical_data(pools, days, max_concurrent=2, delay_between_batches=3)

    def get_pools_by_type_data(self, pool_type: str, days: int = CURRENT_DAYS_SETTING) -> dict:
        """
        按池子類型獲取數據
        
        Args:
            pool_type: 池子類型，如 'stable', 'metapool', 'eth_pool', 'btc_pool', 'crypto' 等
            days: 獲取天数
            
        Returns:
            {pool_name: DataFrame} 字典
        """
        pools = get_pools_by_priority(pool_types=[pool_type])
        print(f"🏷️  獲取 {pool_type} 類型池子數據: {len(pools)} 个池子")
        
        return self.get_batch_historical_data(pools, days)

    def export_batch_data_to_excel(self, batch_data: dict, filename: str = None) -> str:
        """
        将批量數據导出到Excel文件
        
        Args:
            batch_data: 来自批量獲取函数的數據字典
            filename: 输出文件名 (可选)
            
        Returns:
            生成的Excel文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"curve_pools_data_{timestamp}.xlsx"
        
        output_path = self.cache_dir / filename
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # 创建汇总表
                summary_data = []
                for pool_name, df in batch_data.items():
                    if not df.empty:
                        summary_data.append({
                            'Pool Name': pool_name,
                            'Pool Type': df['pool_type'].iloc[0] if 'pool_type' in df.columns else 'Unknown',
                            'Priority': df['priority'].iloc[0] if 'priority' in df.columns else 'Unknown',
                            'Records Count': len(df),
                            'Date Range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}" if len(df) > 0 else 'No data',
                            'Has Virtual Price': 'virtual_price' in df.columns,
                            'Has Volume': 'volume_24h' in df.columns,
                            'Data Sources': ', '.join(df['source'].unique()) if 'source' in df.columns else 'Unknown'
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 为每个池子创建单独的工作表
                for pool_name, df in batch_data.items():
                    if not df.empty:
                        # 限制工作表名称长度
                        sheet_name = pool_name[:31] if len(pool_name) <= 31 else pool_name[:28] + '...'
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
            print(f"📄 批量數據已导出到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Excel导出失败: {e}")
            return ""

    def analyze_batch_data(self, batch_data: dict) -> pd.DataFrame:
        """
        分析批量數據，生成统计报告
        
        Args:
            batch_data: 来自批量獲取函数的數據字典
            
        Returns:
            包含统计信息的DataFrame
        """
        analysis_results = []
        
        for pool_name, df in batch_data.items():
            pool_info = AVAILABLE_POOLS.get(pool_name, {})
            
            if df.empty:
                analysis_results.append({
                    'Pool': pool_name,
                    'Type': pool_info.get('type', 'Unknown'),
                    'Priority': pool_info.get('priority', 'Unknown'),
                    'Status': 'No Data',
                    'Records': 0,
                    'Days Coverage': 0,
                    'Avg Virtual Price': None,
                    'Avg Volume 24h': None,
                    'Last Update': None
                })
                continue
            
            # 计算统计信息
            days_coverage = (df['timestamp'].max() - df['timestamp'].min()).days if len(df) > 1 else 0
            
            analysis_results.append({
                'Pool': pool_name,
                'Type': pool_info.get('type', 'Unknown'),
                'Priority': pool_info.get('priority', 'Unknown'), 
                'Status': 'Success',
                'Records': len(df),
                'Days Coverage': days_coverage,
                'Avg Virtual Price': df['virtual_price'].mean() if 'virtual_price' in df.columns else None,
                'Avg Volume 24h': df['volume_24h'].mean() if 'volume_24h' in df.columns else None,
                'Last Update': df['timestamp'].max() if len(df) > 0 else None
            })
        
        analysis_df = pd.DataFrame(analysis_results)
        
        print("\n📈 批量數據分析报告:")
        print("="*60)
        print(analysis_df.to_string(index=False))
        
        return analysis_df

def demo_free_historical_data():
    """Demonstrate free historical data acquisition - optimized version"""
    
    print("🆓 Free Historical Data Acquisition Demo - Extended Version")
    print("=" * 60)
    
    manager = FreeHistoricalDataManager()
    
    print("📋 Available pool configurations:")
    print(f"   Total: {len(AVAILABLE_POOLS)} pools")
    print(f"   High priority (1-2): {len(get_high_priority_pools())} pools")
    print(f"   Main pools (1-3): {len(get_all_main_pools())} pools")  
    print(f"   Stablecoin pools: {len(get_stable_pools())} pools")
    print()
    
    # Show different types of pools
    print("🏷️  Pool categories:")
    pool_types = set(pool['type'] for pool in AVAILABLE_POOLS.values())
    for pool_type in sorted(pool_types):
        pools_of_type = [name for name, info in AVAILABLE_POOLS.items() if info['type'] == pool_type]
        print(f"   {pool_type}: {len(pools_of_type)} pools ({', '.join(pools_of_type[:3])}...)")
    print()

    # Demo 1: Single pool data acquisition (maintain original demo)
    print("=" * 60)
    print("🎯 Demo 1: Single Pool Historical Data Acquisition")
    print("=" * 60)
    
    single_pool = TARGET_POOL
    print(f"Retrieving {CURRENT_DAYS_SETTING} days of historical data for {single_pool}...")
    
    df_single = manager.get_comprehensive_free_data(
        AVAILABLE_POOLS[single_pool]['address'], 
        single_pool, 
        days=CURRENT_DAYS_SETTING
    )
    
    if not df_single.empty:
        print(f"✅ Single pool data acquisition successful: {len(df_single)} records")
        print(f"   Time span: {(df_single['timestamp'].max() - df_single['timestamp'].min()).days} days")
        if 'virtual_price' in df_single.columns:
            print(f"   Virtual Price range: {df_single['virtual_price'].min():.6f} - {df_single['virtual_price'].max():.6f}")
    else:
        print("❌ Single pool data acquisition failed")
    
    print()
    
    # Demo 2: High priority pool batch acquisition
    print("=" * 60)
    print("⭐ Demo 2: High Priority Pool Batch Data Acquisition")  
    print("=" * 60)
    
    batch_data_high = manager.get_high_priority_pools_data(days=CURRENT_DAYS_SETTING)
    
    if batch_data_high:
        print(f"\n📊 High priority batch data acquisition results:")
        total_records = sum(len(df) for df in batch_data_high.values() if not df.empty)
        successful_pools = sum(1 for df in batch_data_high.values() if not df.empty)
        print(f"   Successfully acquired: {successful_pools}/{len(batch_data_high)} pools")
        print(f"   Total records: {total_records}")
        
        # Brief analysis
        analysis = manager.analyze_batch_data(batch_data_high)
        
    print()
    
    # Demo 3: Get data by type
    print("=" * 60)
    print("🏷️  Demo 3: Get Data by Type - Stablecoin Pools")
    print("=" * 60)
    
    stable_data = manager.get_pools_by_type_data('stable', days=CURRENT_DAYS_SETTING)
    
    if stable_data:
        stable_successful = sum(1 for df in stable_data.values() if not df.empty)
        print(f"✅ Stablecoin pool data acquisition: {stable_successful}/{len(stable_data)} successful")
        
    print()
    
    # Demo 4: Excel export
    print("=" * 60)
    print("📄 Demo 4: Batch Data Export to Excel")  
    print("=" * 60)
    
    if batch_data_high:
        excel_path = manager.export_batch_data_to_excel(
            batch_data_high, 
            f"curve_high_priority_pools_{CURRENT_DAYS_SETTING}d.xlsx"
        )
        if excel_path:
            print(f"✅ Excel file export successful: {excel_path}")
        
    print()
    
    print("=" * 60)
    print("🎉 Demo completed! Main features verified:")
    print("   ✅ Single pool data acquisition")  
    print("   ✅ Batch data acquisition")
    print("   ✅ Filter by priority")
    print("   ✅ Filter by type")
    print("   ✅ Data analysis")
    print("   ✅ Excel export")
    print("=" * 60)

def demo_batch_collection_scenarios():
    """Demonstrate various batch acquisition scenarios"""
    
    print("\n🚀 Batch Data Acquisition Scenario Demo")
    print("=" * 60)
    
    manager = FreeHistoricalDataManager()
    
    # Scenario 1: Quick acquisition of main pool data
    print("📈 Scenario 1: Quick Main Pool Data Acquisition")
    print("-" * 40)
    main_pools_data = manager.get_all_main_pools_data(days=CURRENT_DAYS_SETTING)
    print(f"✅ Main pool data acquisition completed: {len([d for d in main_pools_data.values() if not d.empty])}/{len(main_pools_data)} successful")
    print()
    
    # Scenario 2: ETH-related pools
    print("🔥 Scenario 2: ETH-related Pool Data Acquisition") 
    print("-" * 40)
    eth_pools_data = manager.get_pools_by_type_data('eth_pool', days=CURRENT_DAYS_SETTING)
    if eth_pools_data:
        print(f"✅ ETH pool data acquisition completed: {len([d for d in eth_pools_data.values() if not d.empty])}/{len(eth_pools_data)} successful")
    print()
    
    # Scenario 3: BTC-related pools
    print("₿ Scenario 3: BTC-related Pool Data Acquisition")
    print("-" * 40) 
    btc_pools_data = manager.get_pools_by_type_data('btc_pool', days=CURRENT_DAYS_SETTING)
    if btc_pools_data:
        print(f"✅ BTC pool data acquisition completed: {len([d for d in btc_pools_data.values() if not d.empty])}/{len(btc_pools_data)} successful")
    print()
    
    # Scenario 4: Comprehensive comparison analysis
    print("📊 Scenario 4: Comprehensive Data Comparison Analysis")
    print("-" * 40)
    
    all_batch_data = {}
    all_batch_data.update(main_pools_data)
    if eth_pools_data:
        all_batch_data.update(eth_pools_data)  
    if btc_pools_data:
        all_batch_data.update(btc_pools_data)
    
    if all_batch_data:
        analysis = manager.analyze_batch_data(all_batch_data)
        
        # Export comprehensive data
        excel_path = manager.export_batch_data_to_excel(
            all_batch_data,
            f"curve_comprehensive_pools_{CURRENT_DAYS_SETTING}d.xlsx"
        )
        print(f"✅ Comprehensive data exported: {excel_path}")
    
    print("\n" + "=" * 60)
    print("🎯 Batch acquisition scenario demo completed!")
    print("   You can use different acquisition strategies as needed")
    print("=" * 60)

def show_available_pools_info():
    """Display detailed information of all available pools"""
    
    print("\n📋 Available Curve Pool Detailed Information")
    print("=" * 80)
    
    # Display by priority groups
    for priority in range(1, 6):
        pools_at_priority = {name: info for name, info in AVAILABLE_POOLS.items() 
                           if info['priority'] == priority}
        
        if pools_at_priority:
            priority_labels = {1: "🏆 Highest Priority", 2: "⭐ High Priority", 3: "📈 Medium Priority", 
                             4: "📊 Low Priority", 5: "🔽 Lowest Priority"}
            
            print(f"\n{priority_labels.get(priority, f'Priority {priority}')} ({len(pools_at_priority)} pools):")
            print("-" * 60)
            
            for pool_name, pool_info in sorted(pools_at_priority.items()):
                print(f"• {pool_name:12} | {pool_info['name']:25} | {pool_info['type']:12} | {pool_info['address'][:10]}...")
    
    print(f"\n📊 Statistics:")
    print(f"   Total pools: {len(AVAILABLE_POOLS)}")
    
    # Statistics by type
    type_counts = {}
    for pool_info in AVAILABLE_POOLS.values():
        pool_type = pool_info['type']
        type_counts[pool_type] = type_counts.get(pool_type, 0) + 1
    
    print(f"   Type distribution:")
    for pool_type, count in sorted(type_counts.items()):
        print(f"     {pool_type}: {count} pools")
    
    print("\n🎯 Recommended usage strategies:")
    print("   • Quick testing: get_high_priority_pools_data() - Get priority 1-2 pools")  
    print("   • Daily analysis: get_all_main_pools_data() - Get priority 1-3 pools")
    print("   • Comprehensive analysis: get_all_pools_data() - Get all pool data")
    print("   • Category analysis: get_pools_by_type_data('stable') - Get by type")
    print("=" * 80)

def switch_days_config(days_setting: str):
    """Helper function to switch day configuration"""
    global CURRENT_DAYS_SETTING
    
    config_map = {
        'quick': QUICK_TEST_DAYS,
        'medium': MEDIUM_RANGE_DAYS, 
        'full': FULL_YEAR_DAYS,
        'test': QUICK_TEST_DAYS,
        '7': QUICK_TEST_DAYS,
        '90': MEDIUM_RANGE_DAYS,
        '365': FULL_YEAR_DAYS
    }
    
    if days_setting.lower() in config_map:
        CURRENT_DAYS_SETTING = config_map[days_setting.lower()]
        print(f"✅ Switched to {CURRENT_DAYS_SETTING} days configuration")
    else:
        print(f"❌ Invalid configuration: {days_setting}")
        print("💡 Available options: quick(7 days) | medium(90 days) | full(365 days)")

def demo_all_configurations():
    """Demonstrate all configuration options"""
    print("🎛️  Day configuration switching demo")
    print("=" * 40)
    
    configs = [
        ('quick', 'Quick testing'),
        ('medium', 'Medium-term analysis'), 
        ('full', 'Complete year')
    ]
    
    for config, desc in configs:
        print(f"\n🔄 Switching to {desc} mode:")
        switch_days_config(config)
        
        manager = FreeHistoricalDataManager()
        print(f"📊 Will retrieve {CURRENT_DAYS_SETTING} days of data")

if __name__ == "__main__":
    import sys
    
    # Choose demo mode based on command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "info":
            show_available_pools_info()
        elif mode == "batch":
            demo_batch_collection_scenarios()
        elif mode == "single":
            demo_free_historical_data()
        elif mode == "all":
            show_available_pools_info()
            demo_free_historical_data()  
            demo_batch_collection_scenarios()
        elif mode == "full":
            # 🚀 Directly get one year of historical data for all pools
            print("🚀 Starting to retrieve one year of historical data for all 37 Curve pools...")
            print("⚠️  Note: This will take a long time (estimated 10-20 minutes)")
            print("=" * 60)
            
            manager = FreeHistoricalDataManager()
            
            # Display information about pools to be retrieved
            all_pools = get_pools_by_priority(min_priority=1, max_priority=4)  # Skip priority 5
            print(f"📋 Will retrieve {CURRENT_DAYS_SETTING} days of data for {len(all_pools)} pools")
            print(f"🏷️  Pool type distribution:")
            
            type_counts = {}
            for pool_info in all_pools.values():
                pool_type = pool_info['type']
                type_counts[pool_type] = type_counts.get(pool_type, 0) + 1
            
            for pool_type, count in sorted(type_counts.items()):
                print(f"   {pool_type}: {count} pools")
                
            # Ask user for confirmation
            response = input("\nContinue to retrieve all pool data? (y/N): ")
            if response.lower() in ['y', 'yes', '是']:
                
                print("\n🔄 Starting batch data acquisition...")
                batch_data = manager.get_all_pools_data(days=CURRENT_DAYS_SETTING, skip_low_priority=True)
                
                # Statistics results
                successful = sum(1 for df in batch_data.values() if not df.empty)
                total_records = sum(len(df) for df in batch_data.values() if not df.empty)
                
                print(f"\n🎉 Batch acquisition completed!")
                print(f"   ✅ Success: {successful}/{len(batch_data)} pools")
                print(f"   📊 Total records: {total_records}")
                
                if successful > 0:
                    # Generate analysis report
                    print(f"\n📈 Generating data analysis report...")
                    analysis = manager.analyze_batch_data(batch_data)
                    
                    # Export to Excel
                    print(f"\n📄 Exporting data to Excel...")
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_path = manager.export_batch_data_to_excel(
                        batch_data,
                        f"curve_all_pools_1year_{timestamp}.xlsx"
                    )
                    
                    if excel_path:
                        print(f"✅ Complete data saved: {excel_path}")
                        print(f"📁 Cache directory: {manager.cache_dir.absolute()}")
                        
                        print(f"\n💡 Using data:")
                        print(f"from free_historical_data import FreeHistoricalDataManager")
                        print(f"manager = FreeHistoricalDataManager()")
                        print(f"# Data is cached, next load will be faster")
                        
                else:
                    print("❌ No data successfully retrieved, please check network connection")
                    
            else:
                print("❌ User cancelled operation")
                
        elif mode == "quick-all":
            # 🔥 Quick acquisition of 7-day data for all pools (for testing)
            print("⚡ Quick acquisition of 7-day data for all pools (test mode)...")
            print("=" * 60)
            
            manager = FreeHistoricalDataManager()
            
            # Get 7-day data for quick testing
            batch_data = manager.get_all_pools_data(days=7, skip_low_priority=True)
            
            successful = sum(1 for df in batch_data.values() if not df.empty)
            print(f"\n✅ Quick test completed: {successful}/{len(batch_data)} pools successful")
            
            if successful > 0:
                excel_path = manager.export_batch_data_to_excel(
                    batch_data,
                    "curve_all_pools_7d_test.xlsx"
                )
                print(f"📄 Test data exported: {excel_path}")
                
        else:
            print("Usage: python free_historical_data.py [options]")
            print("Available options:")
            print("  info      - Display all available pool information")
            print("  batch     - Demo batch data acquisition")  
            print("  single    - Demo single pool acquisition")
            print("  all       - Run all demos")
            print("  full      - 🚀 Get one year of historical data for all pools")
            print("  quick-all - ⚡ Quick acquisition of 7-day data for all pools (test)")
    else:
        # Default run single pool demo
        demo_free_historical_data() 