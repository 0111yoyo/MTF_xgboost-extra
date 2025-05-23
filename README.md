# MTF\_xgboost\_加強版

本專案利用 AA Lens 1/2 的 MTF 資料，透過 XGBoost 與 RandomForest 進行迴歸預測，並可依使用者選擇的模式（Lens1 內部切割、Lens2 內部切割、跨鏡頭測試等）自動切割資料、訓練模型、輸出結果圖表與預測 CSV。

## 目錄結構

```plain
MTF_xgboost_加強版/
├─ AA Lens 1.csv
├─ AA Lens 2.csv
├─ test.py             # 主程式
├─ requirements.txt    # 相依套件
└─ mode_1/             # 模式1的輸出資料夾
   ├─ model_mse.csv
   ├─ XGB/
   │  ├─ Gx_XGB.png
   │  └─ Gx_XGB_predictions.csv
   └─ RF/
      └─ …
```

## 安裝需求

1. **Python 3.7+**
2. 建議使用虛擬環境（venv 或 conda）
3. 安裝套件：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 將 `AA Lens 1.csv` 與 `AA Lens 2.csv` 放到專案根目錄。
2. 在終端機執行：

   ```bash
   python test.py
   ```
3. 依提示輸入模式 (1–5)，程式會自動完成：

   * 資料切割
   * 模型訓練
   * 結果繪圖
   * 預測資料存檔
4. 完成後，可在 `mode_<n>` 資料夾底下查看 `model_mse.csv`、預測結果與圖表。

## 注意事項

* 若資料欄位有變動，請在程式開頭的 `FEATURES` 與 `TARGETS` 區段更新對應欄位名稱。
* 如需改變測試集比例，可修改 `TEST_SIZE` 參數。

---

歡迎提交 Issue 或 Pull Request！
