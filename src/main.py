import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, num_features=None):
        self.threshold = threshold
        self.num_features = num_features
        self.selected_indices_ = None

    def fit(self, X, y):
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        correlations = np.array([np.abs(np.corrcoef(X_np[:, i], y)[0, 1]) for i in range(X_np.shape[1])])
        if self.num_features:
            self.selected_indices_ = np.argsort(correlations)[-self.num_features:]
        else:
            self.selected_indices_ = np.where(correlations > self.threshold)[0]
        if len(self.selected_indices_) == 0:
            self.selected_indices_ = np.arange(X_np.shape[1])
        return self

    def transform(self, X):
        return X.iloc[:, self.selected_indices_] if isinstance(X, pd.DataFrame) else X[:, self.selected_indices_]

class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

def load_config(file_path='config/algoparams_from_ui.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data(config, file_path='output.csv'):
    df = pd.read_csv(file_path)
    target = config['target']['target']
    features = [f for f, meta in config['feature_handling'].items() if meta['is_selected']]
    df = df.dropna(subset=[target])
    return df[features], df[target]

def create_imputer(config):
    transformers = []
    for feature, meta in config['feature_handling'].items():
        if not meta['is_selected']: continue
        method = meta['feature_details']['impute_with']
        value = meta['feature_details']['impute_value']
        if method == 'Average of values':
            imp = SimpleImputer(strategy='mean')
        elif method == 'Median of values':
            imp = SimpleImputer(strategy='median')
        elif method == 'Constant':
            imp = SimpleImputer(strategy='constant', fill_value=value)
        else:
            raise ValueError(f"Invalid impute method: {method}")
        transformers.append((f'imp_{feature}', imp, [feature]))
    return ColumnTransformer(transformers, remainder='passthrough')

def create_feature_reducer(config):
    method = config['feature_reduction']['feature_reduction_method']
    opts = config['feature_reduction'].get(method, {})
    if method == 'No Reduction':
        return PassthroughTransformer()
    elif method == 'Principal Component Analysis':
        return PCA(n_components=opts.get('num_of_features_to_keep', 2))
    elif method == 'Correlation with target':
        return CorrelationThresholdSelector(
            threshold=opts.get('threshold', 0.5),
            num_features=opts.get('num_of_features_to_keep')
        )
    elif method == 'Tree-based':
        return SelectFromModel(
            RandomForestRegressor(
                n_estimators=opts.get('num_of_trees', 100),
                max_depth=opts.get('depth_of_trees', 10),
                random_state=0
            ),
            max_features=opts.get('num_of_features_to_keep', 2)
        )
    else:
        raise ValueError(f"Unknown feature reduction method: {method}")

def get_model_class(name, task):
    if task == 'Regression':
        return {
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor
        }[name]
    raise ValueError(f"Unsupported prediction_type: {task}")

def main():
    config = load_config()
    X, y = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = create_imputer(config)
    reducer = create_feature_reducer(config)
    prediction_type = config['target']['prediction_type']

    for model_conf in config['models']:
        if not model_conf['is_selected']: continue
        name = model_conf['model_name']
        hyperparams = model_conf.get('hyperparameters', {})

        if name == "RandomForestRegressor" and 'bootstrap' in hyperparams and 'max_samples' in hyperparams:
            if False in hyperparams['bootstrap']:
                print(f"⚠️ Removing 'max_samples' because bootstrap=False is included for {name}")
                del hyperparams['max_samples']

        model_class = get_model_class(name, prediction_type)
        pipeline = Pipeline([
            ('imputer', imputer),
            ('reducer', reducer),
            ('model', model_class())
        ])

        grid_params = {f"model__{k}": v for k, v in hyperparams.items()}
        folds = config['hyperparameters']['Grid Search']['Time-based K-fold(with overlap)']['num_of_folds']

        search = GridSearchCV(pipeline, grid_params, scoring='neg_mean_squared_error', cv=folds, n_jobs=-1)

        try:
            search.fit(X_train, y_train)
            preds = search.predict(X_test)

            print(f"\n✅ Model: {name}")
            print(f"Best Params: {search.best_params_}")
            print(f"MSE: {mean_squared_error(y_test, preds):.4f}")
            print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
            print(f"R2 Score: {r2_score(y_test, preds):.4f}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Error with {name}: {e}")

if __name__ == "__main__":
    main()
