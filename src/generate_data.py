import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def generate_sample_data(output_file='data.csv', missing_rate=0.1, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load iris dataset
    iris = load_iris()
    data = pd.DataFrame(
        data=iris.data,
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    )
    
    # Introduce missing values
    mask = np.random.random(data.shape) < missing_rate
    data[mask] = np.nan
    
    # Save to CSV
    data.to_csv('output.csv', index=False)
    print(f"Generated dataset saved to {'output.csv'}")

if __name__ == "__main__":
    generate_sample_data()
