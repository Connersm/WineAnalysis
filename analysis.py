import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("winequality-red.csv", delimiter=';')

num_samples = df.shape[0]

print(f"Number of samples: {num_samples}")


plt.figure(figsize=(8, 6))
plt.hist(df['density'], bins=30, edgecolor='black')
plt.title('Distribution of Density')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.grid (True)
plt.show()

density_stats = df ['density'].describe()
print("\nDescriptive statistics for 'density':")
print(density_stats)

median_density = df[ 'density'].median()
print(f"\nMedian of density: {median_density}")

features = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide'
]

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features [:3], 1):
    plt.subplot (1, 3, i)
    plt.hist(df[feature], bins=30, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid (True)
plt.tight_layout ()
plt.show()


plt. figure(figsize=(15, 5))
for i, feature in enumerate(features [3:], 1):
    plt.subplot (1, 3, i)
    plt.hist(df [feature], bins=30, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel (feature)
    plt.ylabel ('Frequency')
    plt.grid (True)
plt.tight_layout ()
plt.show()

features.append('density')
correlations_with_density = df[features].corr()['density'].drop('density')

print("\nCorrelation of each feature with 'density':")
print (correlations_with_density)

correlation_matrix = df[features].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)