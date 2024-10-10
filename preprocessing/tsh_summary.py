import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./tsh_distribution.csv')

df = df[(df['TSH'] > 14) & (df['TSH'] < 50)]

# Plot a displot of the TSH levels by 'definitive_diagnosis' with binning
plt.figure(figsize=(10, 6), dpi=500)
sns.displot(df, x='TSH', hue='definitive_diagnosis', multiple='stack', legend=False ) # type: ignore
# plt.title('TSH Level Distribution')
plt.xlabel('TSH Level')
plt.ylabel('Frequency')
plt.legend(title='Definitive Diagnosis', loc='upper right', labels=['Positive', 'Negative'])
plt.tight_layout()

plt.savefig('./tsh_distribution.eps', format='eps')

