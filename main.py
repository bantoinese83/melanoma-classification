import joblib
import optuna
import config
import numpy as np
import pandas as pd
from halo import Halo
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from data_visualization import plot_target_distribution, plot_age_distribution, plot_feature_correlation

# Create a study with a custom name
study = optuna.create_study(direction=config.DIRECTION, study_name=config.STUDY_NAME)

# Set up logging
logger.add(config.LOG_FILE_PATH, rotation=config.LOG_ROTATION)


def load_data(train_path, test_path):
    spinner = Halo(text='Loading data...', spinner='dots')
    spinner.start()
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        spinner.succeed('Data loaded successfully')
        return train_df, test_df
    except Exception as e:
        spinner.fail('Failed to load data')
        logger.error(e)
        raise


def verify_data_shapes(train_path, test_path):
    train_df, test_df = load_data(train_path, test_path)
    X_train, y_train = preprocess_data(train_df, is_train=True)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    processed_test_data = preprocess_data(test_df, is_train=False)
    if isinstance(processed_test_data, tuple):
        X_test = processed_test_data[0]
    else:
        X_test = processed_test_data
    print(f"X_test shape: {X_test.shape}")


def preprocess_data(df, is_train=True):
    spinner = Halo(text='Preprocessing data...', spinner='dots')
    spinner.start()
    df['age_group'] = pd.cut(df['age_approx'], bins=config.AGE_GROUP_BINS, labels=config.AGE_GROUP_LABELS)
    df['anatom_site_category'] = df['anatom_site_general_challenge'].astype('category').cat.codes
    df['sex_site_interaction'] = df['sex'].astype(str) + '_' + df['anatom_site_general_challenge'].astype(str)

    for feature in config.NUMERIC_FEATURES:
        if feature in df.columns:
            imputer = SimpleImputer(strategy='median')
            df[feature] = imputer.fit_transform(df[[feature]]).ravel()
        else:
            logger.warning(f"Warning: {feature} is not present in the DataFrame")

    for feature in config.CATEGORICAL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].astype(str)
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])

    for feature in config.CATEGORICAL_FEATURES:
        if feature in df.columns:
            imputer = SimpleImputer(strategy='most_frequent')
            df[feature] = imputer.fit_transform(df[[feature]]).ravel()
        else:
            logger.warning(f"Warning: {feature} is not present in the DataFrame")

    for feature in config.CATEGORICAL_FEATURES + ['age_group', 'anatom_site_category', 'sex_site_interaction']:
        if feature in df.columns:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
        else:
            logger.warning(f"Warning: '{feature}' is not present in the DataFrame")

    if is_train:
        df = df.drop(columns=[col for col in config.DROP_COLUMNS_TRAIN if col in df.columns])
        X = df.drop(columns=['target'])
        y = df['target']
        spinner.succeed('Training data preprocessed')
        return X, y
    else:
        df = df.drop(columns=[col for col in config.DROP_COLUMNS_TEST if col in df.columns])
        spinner.succeed('Test data preprocessed')
        return df


def analyze_feature_importance(fitted_model, training_data):
    spinner = Halo(text='Analyzing feature importance...', spinner='dots')
    spinner.start()
    importances = fitted_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    logger.info(f"Number of features in training data: {training_data.shape[1]}")
    logger.info(f"Number of importances: {len(importances)}")
    logger.info("Feature ranking:")
    for i in range(min(training_data.shape[1], len(importances))):
        logger.info(f"{i + 1}. feature {indices[i]} ({importances[indices[i]]})")
    spinner.succeed('Feature importance analysis completed')


def objective(trial, X, y):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', *config.XGB_PARAM_RANGE['n_estimators']),
        'max_depth': trial.suggest_int('xgb_max_depth', *config.XGB_PARAM_RANGE['max_depth']),
        'learning_rate': trial.suggest_float('xgb_learning_rate', *config.XGB_PARAM_RANGE['learning_rate']),
        'subsample': trial.suggest_float('xgb_subsample', *config.XGB_PARAM_RANGE['subsample']),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', *config.XGB_PARAM_RANGE['colsample_bytree'])
    }

    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', *config.RF_PARAM_RANGE['n_estimators']),
        'max_depth': trial.suggest_int('rf_max_depth', *config.RF_PARAM_RANGE['max_depth']),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', *config.RF_PARAM_RANGE['min_samples_split']),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', *config.RF_PARAM_RANGE['min_samples_leaf']),
        'bootstrap': trial.suggest_categorical('rf_bootstrap', config.RF_PARAM_RANGE['bootstrap'])
    }

    try:
        model_pipeline = ImbPipeline(steps=[
            ('smote', SMOTE(random_state=config.SMOTE_RANDOM_STATE)),
            ('rfecv', RFECV(estimator=XGBClassifier(**xgb_params), step=1, cv=5, scoring='accuracy')),
            ('classifier', VotingClassifier(estimators=[
                ('xgb', XGBClassifier(**xgb_params)),
                ('rf', RandomForestClassifier(**rf_params))
            ], voting='soft'))
        ])

        cross_val_results = cross_val_score(model_pipeline, X, y, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cross_val_results}")

        return np.mean(cross_val_results)

    except Exception as e:
        logger.error(f"Trial failed with parameters: {trial.params}")
        logger.error(f"Error: {e}")
        return 0.0


def train_model(X_train, y_train, n_trials=config.N_TRIALS):
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    best_params = study.best_params

    logger.info(f"Best trial F1 score: {study.best_value}")
    logger.info(f"Best trial parameters: {best_params}")

    xgb_best_params = {
        'n_estimators': best_params['xgb_n_estimators'],
        'max_depth': best_params['xgb_max_depth'],
        'learning_rate': best_params['xgb_learning_rate'],
        'subsample': best_params['xgb_subsample'],
        'colsample_bytree': best_params['xgb_colsample_bytree']
    }

    rf_best_params = {
        'n_estimators': best_params['rf_n_estimators'],
        'max_depth': best_params['rf_max_depth'],
        'min_samples_split': best_params['rf_min_samples_split'],
        'min_samples_leaf': best_params['rf_min_samples_leaf'],
        'bootstrap': best_params['rf_bootstrap']
    }

    model_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=config.SMOTE_RANDOM_STATE)),
        ('rfecv', RFECV(estimator=XGBClassifier(**xgb_best_params), step=1, cv=5, scoring='accuracy')),
        ('classifier', VotingClassifier(estimators=[
            ('xgb', XGBClassifier(**xgb_best_params)),
            ('rf', RandomForestClassifier(**rf_best_params))
        ], voting='soft'))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def evaluate_model(model, X_val, y_val):
    try:
        cross_val_results = cross_val_score(model, X_val, y_val, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores on validation set: {cross_val_results}")

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        cm = confusion_matrix(y_val, y_pred)
        cr = classification_report(y_val, y_pred)

        return accuracy, f1, cm, cr

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        return 0.0, 0.0, np.array([[0, 0], [0, 0]]), "Evaluation failed"


def save_model(model, path):
    try:
        joblib.dump(model, path)
        logger.info('Model saved successfully')
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def analyze_trials(study_trials):
    trials = study_trials.trials
    trial_data = [(trial.number, trial.value, trial.params) for trial in trials if trial.value is not None]

    df = pd.DataFrame(trial_data, columns=['trial_number', 'f1_score', 'params'])
    df['mean_f1_score'] = df['f1_score'].apply(np.mean)
    df['std_f1_score'] = df['f1_score'].apply(np.std)

    df_sorted = df.sort_values(by='mean_f1_score', ascending=False)
    best_trial = df_sorted.iloc[0]
    worst_trial = df_sorted.iloc[-1]

    print("Best Trial:")
    print(best_trial)
    print("Worst Trial:")
    print(worst_trial)

    return df_sorted


def main():
    train_df, test_df = load_data(config.TRAIN_PATH, config.TEST_PATH)

    plot_target_distribution(train_df)
    plot_age_distribution(train_df)
    plot_feature_correlation(train_df)

    try:
        X_train, y_train = preprocess_data(train_df, is_train=True)
        X_test = preprocess_data(test_df, is_train=False)
    except Exception as e:
        logger.error(f"Data preprocessing failed with error: {e}")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2,
                                                                              random_state=42)

    model_pipeline = train_model(X_train_split, y_train_split, n_trials=config.N_TRIALS)

    accuracy, f1, cm, cr = evaluate_model(model_pipeline, X_val_split, y_val_split)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    save_model(model_pipeline, config.MODEL_SAVE_PATH)

    best_xgb_model = model_pipeline.named_steps['classifier'].estimators_[0]
    analyze_feature_importance(best_xgb_model, X_train_split)

    try:
        test_predictions = model_pipeline.predict(X_test)
        output = pd.DataFrame({'Id': test_df.index, 'target': test_predictions})
        output.to_csv(config.TEST_PREDICTIONS_PATH, index=False)
        logger.info('Test predictions saved successfully')
    except Exception as e:
        logger.error(f"Failed to predict on test set: {e}")

    # Analyze trials
    trial_results = analyze_trials(study)
    print(trial_results)


if __name__ == "__main__":
    main()
