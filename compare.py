<<<<<<< HEAD
# compare_format.py
import pandas as pd

# 如果懷疑有 BOM，可加上 encoding='utf-8-sig'
df1 = pd.read_csv('AA Lens 1.csv', encoding='utf-8')
df2 = pd.read_csv('AA Lens 2.csv', encoding='utf-8')

print("▶ Lens 1: shape =", df1.shape)
print("▶ Lens 2: shape =", df2.shape)
print()

cols1 = set(df1.columns)
cols2 = set(df2.columns)

print("▶ 只在 Lens 1 出現，Lens 2 缺少的欄位：")
for c in sorted(cols1 - cols2):
    print("  ", c)
print()

print("▶ 只在 Lens 2 出現，Lens 1 缺少的欄位：")
for c in sorted(cols2 - cols1):
    print("  ", c)
print()

print("▶ Lens 1 前 5 列：")
print(df1.head().to_string(index=False))
print()

print("▶ Lens 2 前 5 列：")
print(df2.head().to_string(index=False))
=======
# compare_format.py
import pandas as pd

# 如果懷疑有 BOM，可加上 encoding='utf-8-sig'
df1 = pd.read_csv('AA Lens 1.csv', encoding='utf-8')
df2 = pd.read_csv('AA Lens 2.csv', encoding='utf-8')

print("▶ Lens 1: shape =", df1.shape)
print("▶ Lens 2: shape =", df2.shape)
print()

cols1 = set(df1.columns)
cols2 = set(df2.columns)

print("▶ 只在 Lens 1 出現，Lens 2 缺少的欄位：")
for c in sorted(cols1 - cols2):
    print("  ", c)
print()

print("▶ 只在 Lens 2 出現，Lens 1 缺少的欄位：")
for c in sorted(cols2 - cols1):
    print("  ", c)
print()

print("▶ Lens 1 前 5 列：")
print(df1.head().to_string(index=False))
print()

print("▶ Lens 2 前 5 列：")
print(df2.head().to_string(index=False))
>>>>>>> origin/master
