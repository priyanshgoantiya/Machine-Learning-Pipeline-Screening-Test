import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def generate_sample_data(output_file='output.csv', missing_rate=0.1, target_col='petal_width', random_seed=42):
    np.random.seed(random_seed)

    iris = load_iris()
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data = pd.DataFrame(data=iris.data, columns=columns)

    feature_cols = [col for col in columns if col != target_col]
    mask = np.random.rand(*data[feature_cols].shape) < missing_rate
    data.loc[:, feature_cols] = data.loc[:, feature_cols].mask(mask)

    data.to_csv('output.csv', index=False)
    print(f"✅ Dataset with missing values saved to '{'output.csv'}'")

if __name__ == "__main__":
    generate_sample_data()
