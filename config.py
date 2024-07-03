# Paths
TRAIN_PATH = 'mnt/data/train.csv'
TEST_PATH = 'mnt/data/test.csv'
MODEL_SAVE_PATH = 'models/melanoma_predict_model.pkl'
LOG_FILE_PATH = 'logs/melanoma_classification.log'
TEST_PREDICTIONS_PATH = 'mnt/data/test_predictions.csv'

# Optuna settings
STUDY_NAME = "melanoma_classification"
DIRECTION = "maximize"
N_TRIALS = 100

# Imbalance handling
SMOTE_RANDOM_STATE = 42

# Model parameters range
XGB_PARAM_RANGE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.7, 1.0)
}

RF_PARAM_RANGE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
    'bootstrap': [True, False]
}

# Preprocessing settings
NUMERIC_FEATURES = ['age_approx']
CATEGORICAL_FEATURES = ['sex', 'anatom_site_general_challenge']
DROP_COLUMNS_TRAIN = ['image_name', 'patient_id', 'lesion_id', 'diagnosis', 'benign_malignant']
DROP_COLUMNS_TEST = ['image_name', 'patient_id']
AGE_GROUP_BINS = [0, 20, 40, 60, 80, float('inf')]
AGE_GROUP_LABELS = ['0-20', '20-40', '40-60', '60-80', '80+']

# Logging settings
LOG_ROTATION = "10 MB"
