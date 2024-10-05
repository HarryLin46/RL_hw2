import numpy as np

# 讀取 Ground Truth (GT) 檔案
gt_file_path = 'sample_solutions/prediction_GT.npy'  # 替換為你的 Ground Truth 檔案路徑
gt_data = np.load(gt_file_path)

# 讀取含有多個樣本的 .npy 檔案
sample_file_path = 'my_result/MC.npy'  # 替換為你的樣本檔案路徑
samples_data = np.load(sample_file_path)

# 假設 samples_data 的形狀為 (num_samples, ...) 
# 其中 num_samples 是樣本數，其後是每個樣本的多維數據

# 計算樣本平均值
samples_mean = np.mean(samples_data, axis=0)

# 計算偏差 (Bias) - 樣本平均值與 Ground Truth 的差異
bias = samples_mean - gt_data

# 計算變異數 (Variance) - 每個樣本與樣本平均值的變異數
variance = np.var(samples_data, axis=0)

# 輸出偏差與變異數
print("Bias between samples mean and ground truth:\n", bias)
print("\nVariance of the samples:\n", variance)



# 計算每個狀態的平均偏差 (Mean Bias)
mean_bias = np.mean(bias)

# 計算每個狀態的平均變異數 (Mean Variance)
mean_variance = np.mean(variance)

# 顯示結果
print(f"Mean Bias: {mean_bias}")
print(f"Mean Variance: {mean_variance}")

# 如果想進一步視覺化每個狀態的偏差和變異數分佈，可以畫圖
import matplotlib.pyplot as plt

# 繪製偏差的直方圖
plt.hist(bias.flatten(), bins=20, alpha=0.7, label='Bias')
plt.title("Bias Distribution Across States")
plt.xlabel("Bias")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 繪製變異數的直方圖
plt.hist(variance.flatten(), bins=20, alpha=0.7, label='Variance', color='orange')
plt.title("Variance Distribution Across States")
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.legend()
plt.show()
