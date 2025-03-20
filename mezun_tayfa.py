import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import numpy as np

train_df = pd.read_csv('train.csv') 

X = train_df.drop(columns=['id', 'Status'])
y = train_df['Status']

numerical_cols = X.select_dtypes(include=['float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)
y_mapped = y.map({'C': 0, 'CL': 1, 'D': 2}) 

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_loss_scores = []

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'learning_rate': 0.03,
    'max_depth': 8,
    'n_estimators': 4687,
    'eval_metric': 'mlogloss',
    'subsample': 0.7,
    'colsample_bytree': 0.14,
    'gamma': 1
}

for train_index, val_index in kf.split(X, y_mapped):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y_mapped.iloc[train_index], y_mapped.iloc[val_index]
    
    # Fit the preprocessor only on the training fold
    preprocessor.fit(X_train_fold)
    X_train_transformed = preprocessor.transform(X_train_fold)
    X_val_transformed = preprocessor.transform(X_val_fold)

    model = XGBClassifier(**xgb_params)
    model.fit(X_train_transformed, y_train_fold)

    y_val_proba = model.predict_proba(X_val_transformed)
    log_loss_score = log_loss(y_val_fold, y_val_proba)
    log_loss_scores.append(log_loss_score)



average_log_loss = np.mean(log_loss_scores)
print(f"Average Log Loss: {average_log_loss}")

test_df = pd.read_csv('test.csv') 

test_preprocessed = preprocessor.transform(test_df.drop(columns=['id']))

test_predictions = model.predict_proba(test_preprocessed)

submission_df = pd.DataFrame(test_predictions, columns=['Status_C', 'Status_CL', 'Status_D'])
submission_df.insert(0, 'id', test_df['id']) 
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")