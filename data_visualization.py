import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df):
    """ Plot distribution of target variable """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Benign', 'Malignant'])
    plt.show()


def plot_age_distribution(df):
    """ Plot distribution of age """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age_approx'], bins=30, kde=True)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()


def plot_feature_correlation(df):
    """ Plot correlation heatmap for numeric features """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


