# check_features.py
import pandas as pd
import numpy as np

# ---------- 參數設定 ----------
CSV_PATH = 'AA Lens 2.csv'
FEATURES = [
    'D6S8S','D7S9S','D2S3S','D4S5S','D2S5S','D3S4S',
    'D4S8S','D5S9S','D6S9S','D7S8S','D8S9S',
    'D2S8S','D2S9S','D3S9S','D4S9S','D5S8S','D6S7S','D7S7S','D8S8S','D9S9S',
    'D2S2S','D3S3S','D4S4S','D5S5S','D6S6S','D7S7S','D8S8S','D9S9S',
    'D2S2T','D3S3T','D4S4T','D5S5T','D6S6T','D7S7T','D8S8T','D9S9T',
    'FS_1S','FS_1T','FS_2S','FS_2T','FS_3S','FS_3T','FS_4S','FS_4T',
    'FS_5S','FS_5T','FS_6S','FS_6T','FS_7S','FS_7T','FS_8S','FS_8T','FS_9S','FS_9T',
    'FS_Ave_S_G2','FS_Ave_T_G2','FS_Ave_S_G3','FS_Ave_T_G3',
    'FS_Delta_S_G2','FS_Delta_T_G2','FS_Delta_S_G3','FS_Delta_T_G3',
    'PeakMTF_1S','PeakMTF_1T','PeakMTF_2S','PeakMTF_2T','PeakMTF_3S','PeakMTF_3T',
    'PeakMTF_4S','PeakMTF_4T','PeakMTF_5S','PeakMTF_5T','PeakMTF_6S','PeakMTF_6T',
    'PeakMTF_7S','PeakMTF_7T','PeakMTF_8S','PeakMTF_8T','PeakMTF_9S','PeakMTF_9T'
]
TARGETS = ['Gx','Gy','Tx','Ty']

# 門檻
STD_THRESHOLD  = 1e-6   # 小於此 std 的欄位視作「常數」
CORR_THRESHOLD = 0.1    # 只顯示絕對相關大於此值的特徵

def main():
    print(f"載入資料：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    # 僅保留實際存在的 FEATURES 和 TARGETS
    available_features = [f for f in FEATURES if f in df.columns]
    available_targets  = [t for t in TARGETS  if t in df.columns]
    df_num = df[available_features + available_targets].replace('--', np.nan).astype(float)
    
    # 1. 變異數檢查
    stats = df_num[available_features].describe().T
    stats = stats[['mean','std','min','max']].sort_values('std')
    print("\n=== 標準差最低的 10 個特徵（可視為常數欄位） ===")
    print(stats.head(10).to_string())
    
    low_var = stats[stats['std'] < STD_THRESHOLD].index.tolist()
    print(f"\n變異度（std）< {STD_THRESHOLD} 的欄位，共 {len(low_var)} 個：")
    print(low_var)
    
    # 2. 相關性檢查
    print("\n=== 各特徵與目標的皮爾森相關性（絕對值） ===")
    for target in available_targets:
        corr = df_num[available_features + [target]].corr()[target].abs().sort_values(ascending=False)
        topk = corr.head(10)
        print(f"\n>>> 與 {target} 相關性最高的前 10：")
        print(topk.to_string())
        useful = corr[corr > CORR_THRESHOLD].index.tolist()
        print(f"相關度 > {CORR_THRESHOLD}（共 {len(useful)} 個）：\n{useful}")
    
    # （可選）把變異數結果存成 CSV
    stats.to_csv('lens2_feature_stats.csv')
    print("\n已將變異數結果輸出到 lens2_feature_stats.csv")

if __name__ == '__main__':
    main()
