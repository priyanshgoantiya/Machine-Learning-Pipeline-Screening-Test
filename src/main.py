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

# Custom transformer for correlation-based feature selection
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, num_features=None):
        self.threshold = threshold
        self.num_features = num_features
        self.selected_indices_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # Handle potential NaNs in correlations
        correlations = []
        for i in range(X.shape[1]):
            valid_idx = ~np.isnan(X[:, i]) & ~np.isnan(y)
            if valid_idx.sum() > 1:  # Ensure enough valid data
                corr = np.abs(np.corrcoef(X[valid_idx, i], y[valid_idx])[0, 1])
            else:
                corr = 0
            correlations.append(corr)
        correlations = np.array(correlations)
        if self.num_features:
            self.selected_indices_ = np.argsort(correlations)[-self.num_features:]
        else:
            self.selected_indices_ = np.where(correlations > self.threshold)[0]
        if len(self.selected_indices_) == 0:
            self.selected_indices_ = np.arange(X.shape[1])  # Fallback to all features
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

# Passthrough transformer for no reduction
class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

def load_config(json_file='config/algoparams_from_ui.json'):
    """Load and parse the JSON configuration."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {json_file} not found")

def load_data(config, data_file='data.csv'):
    """Load dataset and split into features and target."""
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {data_file} not found")
    
    target = config['target']['target']
    features = [f for f, details in config['feature_handling'].items() if details.get('is_selected', False)]
    
    if not features:
        raise ValueError("No features selected in feature_handling")
    
    # Drop rows with NaN in the target column
    data = data.dropna(subset=[target])
    
    X = data[features]
    y = data[target]
    return X, y

def create_imputation_transformer(config):
    """Create ColumnTransformer for imputation based on feature_handling."""
    transformers = []
    for feature, details in config['feature_handling'].items():
        if not details.get('is_selected', False):
            continue
        feature_details = details.get('feature_details', {})
        impute_with = feature_details.get('impute_with', 'Average of values')
        impute_value = feature_details.get('impute_value', 0)
        
        if impute_with == 'Average of values':
            imputer = SimpleImputer(strategy='mean')
        elif impute_with == 'Median of values':
            imputer = SimpleImputer(strategy='median')
        elif impute_with == 'Constant':
            imputer = SimpleImputer(strategy='constant', fill_value=impute_value)
        else:
            raise ValueError(f"Unknown imputation method for {feature}: {impute_with}")
        transformers.append((f'imputer_{feature}', imputer, [feature]))
    
    if not transformers:
        return PassthroughTransformer()  # No imputation needed
    return ColumnTransformer(transformers, remainder='passthrough')

def create_feature_reduction_transformer(config, prediction_type):
    """Create feature reduction transformer based on feature_reduction."""
    feature_reduction = config.get('feature_reduction', {})
    method = feature_reduction.get('feature_reduction_method', 'No Reduction')
    reduction_config = feature_reduction.get(method, {})

    if method == 'No Reduction':
        return PassthroughTransformer()
    elif method == 'Principal Component Analysis':
        n_components = reduction_config.get('num_of_features_to_keep', 2)
        return PCA(n_components=n_components)
    elif method == 'Correlation with target':
        threshold = reduction_config.get('threshold', 0.5)
        num_features = reduction_config.get('num_of_features_to_keep')
        return CorrelationThresholdSelector(threshold=threshold, num_features=num_features)
    elif method == 'Tree-based':
        num_features = reduction_config.get('num_of_features_to_keep', 2)
        max_depth = reduction_config.get('depth_of_trees', 10)
        n_estimators = reduction_config.get('num_of_trees', 100)
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0
        )
        return SelectFromModel(estimator, max_features=num_features)
    else:
        raise ValueError(f"Unknown feature reduction method: {method}")

def get_model_class(model_name, prediction_type):
    """Return model class based on name and prediction type."""
    if prediction_type.lower() == 'regression':
        models = {
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor
        }
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported for {prediction_type}")
    return models[model_name]

def main():
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    prediction_type = config['target'].get('prediction_type', 'Regression')

    # Load data
    try:
        X, y = load_data(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    # Create imputation transformer
    try:
        imputation_transformer = create_imputation_transformer(config)
    except Exception as e:
        print(f"Error creating imputation transformer: {e}")
        return

    # Create feature reduction transformer
    try:
        feature_reduction_transformer = create_feature_reduction_transformer(config, prediction_type)
    except Exception as e:
        print(f"Error creating feature reduction transformer: {e}")
        return

    # Process models
    models = config.get('models', [])
    if not models:
        print("No models specified in configuration")
        return

    for model_config in models:
        if not model_config.get('is_selected', False):
            continue

        model_name = model_config.get('model_name')
        hyperparams = model_config.get('hyperparameters', {})

        # Create model
        try:
            model_class = get_model_class(model_name, prediction_type)
            model = model_class()
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            continue

        # Create pipeline
        pipeline = Pipeline([
            ('imputation', imputation_transformer),
            ('feature_reduction', feature_reduction_transformer),
            ('model', model)
        ])

        # Prepare GridSearchCV
        param_grid = {f"model__{k}": v for k, v in hyperparams.items()}
        cv_strategy = config.get('hyperparameters', {}).get('Grid Search', {}).get(
            'Time-based K-fold(with overlap)', {}
        ).get('num_of_folds', 5)

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            error_score='raise'
        )

        # Fit and predict
        try:
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)

            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log results
            print(f"Model: {model_name}")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()
