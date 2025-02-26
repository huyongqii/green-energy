import pandas as pd

# 读取CSV文件
df = pd.read_csv('sk1/job_timeline_2023.csv')

# 添加 utilization_rate 列
df['utilization_rate'] = round(df['nb_computing'] / 150, 4)

# 保存处理后的文件
df.to_csv('sk1/job_timeline_2023_with_utilization.csv', index=False)

print("处理完成，已生成新文件：job_timeline_2023_with_utilization.csv")
