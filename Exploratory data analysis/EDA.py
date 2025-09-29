import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')

print("--- Summary Statistics ---")

print(df.describe())
print("\n")

print("--- Generating Visualizations ---")

sns.set_style('whitegrid')

plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png')
print("Saved age distribution plot as 'age_distribution.png'")

plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df)
plt.title('Passenger Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('gender_count.png')
print("Saved gender count plot as 'gender_count.png'")

plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=df)
plt.title('Age Distribution across Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.savefig('age_by_class.png')
print("Saved age by class plot as 'age_by_class.png'")

print("\n--- Generating Correlation Matrix ---")

numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_matrix.png')
print("Saved correlation matrix heatmap as 'correlation_matrix.png'")


