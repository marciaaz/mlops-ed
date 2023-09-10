import os
import mlflow
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""
data = pd.read_csv('https://raw.githubusercontent.com/'
                   'renansantosmendes/lectures-cdas-2023/'
                   'master/fetal_health.csv')

"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""
features_to_remove = data.columns[7:]
X = data.drop(features_to_remove, axis=1)  # X matriz de entrada
y = data["fetal_health"]  # y vetor de saída
columns = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(X_df, y,
                                                    random_state=42,
                                                    test_size=0.3)


os.environ['MLFLOW_TRACKING_USERNAME'] = 'mar.f.azevedo'
os.environ['MLFLOW_TRACKING_PASSWORD'] = \
    'cc60d3538102b0977e5a062692efb453e35f5b65'

mlflow.set_tracking_uri('https://dagshub.com/mar.f.azevedo/'
                        'mlops-puc-20230821.mlflow')
mlflow.sklearn.autolog(log_models=True,
                       log_input_examples=True,
                       log_model_signatures=True)

"""# **Modelos Ensemble**"""
gradient_classifier = GradientBoostingClassifier(verbose=1,
                                                 max_depth=12,
                                                 n_estimators=150,
                                                 learning_rate=0.05)
with mlflow.start_run(run_name='gradient_boosting') as run:
    gradient_classifier.fit(X_train, y_train)
