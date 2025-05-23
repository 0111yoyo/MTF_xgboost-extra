import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------- 參數設定 ----------
OUTPUT_DIR = '.'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 測試集比例，可自行調整
TEST_SIZE = 0.35

# ---------- 載入資料函式 ----------
def load_and_clean(path):
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if (df[c] == '--').all()]
    return df.drop(columns=drop_cols)

# 特徵與目標欄位定義
FEATURES = [
    'D6S8S','D7S9S','D2S3S','D4S5S','D2S5S','D3S4S',
    'D6T8T','D7T9T','D2T3T','D4T5T','D2T5T','D3T4T',
    'FS_1S','FS_1T','FS_2S','FS_2T','FS_3S','FS_3T','FS_4S','FS_4T',
    'FS_5S','FS_5T','FS_6S','FS_6T','FS_7S','FS_7T','FS_8S','FS_8T',
    'FS_9S','FS_9T',
    'FS_Ave_S_G2','FS_Ave_T_G2','FS_Ave_S_G3','FS_Ave_T_G3',
    'FS_Delta_S_G2','FS_Delta_T_G2','FS_Delta_S_G3','FS_Delta_T_G3',
    'PeakMTF_1S','PeakMTF_1T','PeakMTF_2S','PeakMTF_2T','PeakMTF_3S','PeakMTF_3T',
    'PeakMTF_4S','PeakMTF_4T','PeakMTF_5S','PeakMTF_5T','PeakMTF_6S','PeakMTF_6T',
    'PeakMTF_7S','PeakMTF_7T','PeakMTF_8S','PeakMTF_8T','PeakMTF_9S','PeakMTF_9T'
]
TARGETS = ['Gx','Gy','Tx','Ty']

# ---------- 主流程 ----------
def main():
    print("請選擇訓練模式：")
    print("1. AA Lens 1 內部切割")
    print("2. AA Lens 2 內部切割")
    print("3. 訓練: AA Lens 1 -> 測試: AA Lens 2")
    print("4. 訓練: AA Lens 2 -> 測試: AA Lens 1")
    print("5. 結合兩組資料，打散後切割")
    choice = input("輸入選項(1-5): ").strip()

    # 載入資料
    df1 = load_and_clean('AA Lens 1.csv')
    df2 = load_and_clean('AA Lens 2.csv')
    features = [f for f in FEATURES if f in df1.columns and f in df2.columns]
    targets  = [t for t in TARGETS if t in df1.columns and t in df2.columns]

    # 建立模式資料夾
    mode_dir = os.path.join(OUTPUT_DIR, f"mode_{choice}")
    os.makedirs(mode_dir, exist_ok=True)

    # 隨機種子，用於 shuffle 和 train_test_split
    seed = random.randint(0, 2**32 - 1)

    # 根據模式切割資料
    if choice == '1':
        df = df1.sample(frac=1, random_state=seed)
        X = df[features].replace('--',0).astype(float).values
        Y = df[targets].replace('--',0).astype(float).values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=TEST_SIZE, shuffle=True, random_state=seed
        )
        print(f"Lens1: shuffle+split with random_state={seed}")
    elif choice == '2':
        df = df2.sample(frac=1, random_state=seed)
        X = df[features].replace('--',0).astype(float).values
        Y = df[targets].replace('--',0).astype(float).values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=TEST_SIZE, shuffle=True, random_state=seed
        )
        print(f"Lens2: shuffle+split with random_state={seed}")
    elif choice == '3':
        # Lens1 作為訓練，Lens2 作為測試，皆先隨機洗牌
        tr = df1.sample(frac=1, random_state=seed)
        te = df2.sample(frac=1, random_state=seed+1)
        X_train = tr[features].replace('--',0).astype(float).values
        Y_train = tr[targets].replace('--',0).astype(float).values
        X_test  = te[features].replace('--',0).astype(float).values
        Y_test  = te[targets].replace('--',0).astype(float).values
        print(f"Lens1 (train) shuffle with seed={seed}; Lens2 (test) shuffle with seed={seed+1}")
    elif choice == '4':
        # Lens2 作為訓練，Lens1 作為測試，皆先隨機洗牌
        tr = df2.sample(frac=1, random_state=seed)
        te = df1.sample(frac=1, random_state=seed+1)
        X_train = tr[features].replace('--',0).astype(float).values
        Y_train = tr[targets].replace('--',0).astype(float).values
        X_test  = te[features].replace('--',0).astype(float).values
        Y_test  = te[targets].replace('--',0).astype(float).values
        print(f"Lens2 (train) shuffle with seed={seed}; Lens1 (test) shuffle with seed={seed+1}")
    elif choice == '5':
        df = pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=seed)
        X = df[features].replace('--',0).astype(float).values
        Y = df[targets].replace('--',0).astype(float).values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=TEST_SIZE, shuffle=True, random_state=seed
        )
        print(f"Combined: shuffle+split with random_state={seed}")
    else:
        print("無效選項，請重新執行並輸入 1-5。")
        return

    # 資料標準化
    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    # 模型與超參數設定
    models = {
        'XGB': XGBRegressor(objective='reg:squarederror', verbosity=0),
        'RF': RandomForestRegressor()
    }
    param_dist = {
        'XGB': {
            'n_estimators': [50,100,200], 'max_depth': [3,5,7],
            'learning_rate': [0.01,0.05,0.1], 'subsample': [0.6,0.8,1.0],
            'colsample_bytree': [0.6,0.8,1.0]
        },
        'RF': {
            'n_estimators': [50,100,200], 'max_depth': [None,5,10,20],
            'max_features': ['sqrt','log2', None]
        }
    }

    mse_results = []
    for idx, target in enumerate(targets):
        y_tr = Y_train[:, idx]
        y_te = Y_test[:, idx]
        scaler_y = StandardScaler().fit(y_tr.reshape(-1,1))
        y_tr_s = scaler_y.transform(y_tr.reshape(-1,1)).ravel()
        for name, model in models.items():
            n_iter = random.randint(5, 15)
            print(f"Training {name} for {target} ({n_iter} random searches)")
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist[name],
                n_iter=n_iter,
                cv=ShuffleSplit(n_splits=5, test_size=TEST_SIZE),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            rs.fit(X_train_s, y_tr_s)
            best = rs.best_estimator_
            y_pred_s = best.predict(X_test_s)
            y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
            mse = mean_squared_error(y_te, y_pred)
            mse_results.append({'mode': choice, 'target': target, 'model': name, 'MSE': mse})
            model_dir = os.path.join(mode_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            # 繪圖與儲存
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
            ax1.scatter(y_te, y_pred, alpha=0.6)
            mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
            ax1.plot([mn,mx], [mn,mx], 'r--')
            ax1.set_title(f'{name} {target}\nMSE={mse:.3e}')
            ax1.set_xlabel('True'); ax1.set_ylabel('Pred'); ax1.grid(True)
            fi = sorted(zip(features, best.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
            fnames, fvals = zip(*fi)
            ax2.barh(fnames[::-1], fvals[::-1]); ax2.set_xlabel('Importance'); ax2.set_title('Top5 Features')
            fig.tight_layout()
            fig.savefig(os.path.join(model_dir, f'{target}_{name}.png'), dpi=300)
            plt.close(fig)
            # 儲存預測結果
            pd.DataFrame({'True': y_te, 'Predicted': y_pred}).to_csv(
                os.path.join(model_dir, f'{target}_{name}_predictions.csv'), index=False
            )

    # 儲存 MSE 表
    pd.DataFrame(mse_results).to_csv(os.path.join(mode_dir, 'model_mse.csv'), index=False)
    print("完成！請查看對應資料夾輸出檔案。")

if __name__ == '__main__':
    main()