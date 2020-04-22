import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import mlflow
import mlflow.sklearn
import hydra
import hydra.experimental

hydra.experimental.initialize()
cfg = hydra.experimental.compose(config_file="config.yaml")

experiment_name = cfg.training.experiment_name
mlflow.set_experiment(experiment_name)
tracking = mlflow.tracking.MlflowClient()
experiment = tracking.get_experiment_by_name(experiment_name)

def train_xgb():
    np.random.seed(0)
    # データの用意
    X,y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    run_name = cfg.training.run_name
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        xgbClassifier = xgb.XGBClassifier(max_depth=cfg.training.max_depth, 
                                          learning_rate=cfg.training.learning_rate, 
                                          n_estimators=cfg.training.n_estimators, 
                                          booster=cfg.training.booster, 
                                          subsample=cfg.training.subsample, 
                                          min_child_weight=cfg.training.min_child_weight
                                          )
        pipeline = make_pipeline(PolynomialFeatures(),xgbClassifier)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        # 評価値の計算
        accuracy = accuracy_score(y_test, pred)
        recall = recall_score(y_test, pred, average="weighted")
        precision = precision_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")
        # パラメータの保存
        mlflow.log_param("method_name",xgbClassifier.__class__.__name__)
        mlflow.log_param("max_depth", cfg.training.max_depth)
        mlflow.log_param("learning_rate", cfg.training.learning_rate)
        mlflow.log_param("n_estimators", cfg.training.n_estimators)
        mlflow.log_param("booster", cfg.training.booster)
        mlflow.log_param("subsample", cfg.training.subsample)
        mlflow.log_param("min_child_weight", cfg.training.min_child_weight)
        # 評価値の保存
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        # モデルの保存
        mlflow.sklearn.log_model(pipeline, "model")

if __name__ == "__main__":
    train_xgb()