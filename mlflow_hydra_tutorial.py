import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import mode
import random
from sklearn import metrics
from sklearn.metrics import log_loss, RocCurveDisplay, confusion_matrix, make_scorer, mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import feather
import mlflow
import mlflow.lightgbm
import hydra
import hydra.experimental
import shap
import os
from sklearn.datasets import load_boston

sns.set_style('whitegrid')
# plt.style.context('ggplot')
hydra.experimental.initialize()


def groupby_agg(df, key, agg_column, agg_func, mode=False, size=False, prefix=False, suffix=False):
    df = df.copy()
    agg_df = pd.DataFrame()
    
    for column in agg_column:
        df_tmp = df.groupby(key)[column].agg(agg_func)
        df_tmp.columns = ['{}_{}'.format(column, agg_name) for agg_name in agg_func]
        # add mode(最頻値)
        if mode:
            df_mode = df.groupby(key)[[column]].agg(mod)
            df_mode.columns = ['{}_mode'.format(column)]
            df_tmp = pd.concat([df_tmp, df_mode], axis=1)
        # key毎のログ数
        if size:
            df_size = df.groupby(key).size()
            df_tmp = pd.concat([df_tmp, df_size], axis=1)
            df_tmp = df_tmp.rename(columns={0:'{}_logsize'.format(column)})
        agg_df = pd.concat([agg_df, df_tmp], axis=1)
    
    # add prefix
    if prefix:
        agg_df = agg_df.add_prefix(prefix)
    # add suffix
    if suffix:
        agg_df = agg_df.add_suffix(suffix)
    
    agg_df = agg_df.reset_index()
    
    return agg_df

def mod(x):
    return mode(x)[0][0]


class LightgbmTrainer():
    def __init__(self, cfg_path):
        '''
        - Hydra・MLflowの初期化
        - Hydraに読み込ませるconfig.yamlの設定
        - MLflowのexperiment_name, run_idの設定
        
        arguments
        ------------
        cfg_path: path of config.yaml for hydra
        
        '''
        self.cfg_path = cfg_path
        # cwd = os.getcwd()
        # hydra.experimental.initialize(config_dir=cwd) # hydra initialize
       
        self.cfg = hydra.experimental.compose(config_file=self.cfg_path) # config.yaml
        
        # 保存先ディレクトリの設定
        # mlflow.log_artifactがうまくいかない
#         tracking_uri = '../mlruns'
#         mlflow.tracking.set_tracking_uri(tracking_uri)
        
        # MLflow experiment_nameの設定
        self.experiment_name = self.cfg.training.experiment_name # mlflow experiment name
        mlflow.set_experiment(self.experiment_name)
        
        # MLflow run_idの取得
        self.tracking = mlflow.tracking.MlflowClient()
        experiment = self.tracking.get_experiment_by_name(self.experiment_name)
        
        self.experiment_id = experiment.experiment_id # mlflow experiment id
        self.run_id = self.tracking.create_run(experiment.experiment_id).info.run_id # mlflow run id in experiment id
        
#         self.run_name = self.cfg.training.run_name # mlflow run name # run_idを指定すると設定できない
        
    def fit(self, X, y, sample_weight=None, log_model=False):
        '''
        - LightGBMの学習部分
        - config.yamlで記述されているハイパーパラメータ類を読み込み・MLflow側で保存
        
        arguments
        ------------
        X: Feature in train set
        y: Label in train set
        sample_weight: Weight in train set
        save_model: Log a LightGBM model as an MLflow artifact for the current run
        
        '''
        # Dataset
        self.X_train = X
        self.y_train = y
        
        # sample weight
        self.weight = None
        if sample_weight is not None:
            self.weight = sample_weight
        
        # cache
        self.models = []
        self.evals_results = []
        self.y_oof = np.zeros(len(self.y_train))
        
        # cross validation type
        if self.cfg.training.cv == 'kfold':
            self.cv = KFold(n_splits=self.cfg.training.num_fold, shuffle=True, random_state=1234)
            self.cv_splitter = self.cv.split(self.X_train)
        elif self.cfg.training.cv == 'stratified':
            self.cv = StratifiedKFold(n_splits=self.cfg.training.num_fold, shuffle=True, random_state=1234)
            self.cv_splitter = self.cv.split(self.X_train, self.y_train)
        
        # training
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
            for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter):
                print('Cross Validation : {} Fold'.format(fold + 1))

                evals_result_fold = {}

                X_train_fold, y_train_fold = self.X_train.iloc[train_idx], self.y_train.iloc[train_idx]
                X_valid_fold, y_valid_fold = self.X_train.iloc[valid_idx], self.y_train.iloc[valid_idx]

                if sample_weight is not None:
                    sample_weight_fold = self.weight[train_idx]
                    train_set = lgb.Dataset(X_train_fold, y_train_fold, weight=sample_weight_fold)
                else:
                    train_set = lgb.Dataset(X_train_fold, y_train_fold)
                valid_set = lgb.Dataset(X_valid_fold, y_valid_fold, reference=train_set)

                fold_model = lgb.train(train_set=train_set, 
                                       valid_sets=[train_set, valid_set], 
                                       evals_result=evals_result_fold, 
                                       params={
                                                'boosting_type': self.cfg.training.boosting_type,
                                                'learning_rate': self.cfg.training.learning_rate,
                                                'bagging_fraction': self.cfg.training.bagging_fraction,
                                                'lambda_l1': self.cfg.training.lambda_l1,
                                                'lambda_l2': self.cfg.training.lambda_l2,
                                                'metric': self.cfg.training.metric,
                                                'objective': self.cfg.training.objective, 
                                                'random_state': self.cfg.training.random_state,
                                                'num_threads': self.cfg.training.num_threads,
                                                # 'subsample': self.cfg.training.subsample,
                                                'subsample_freq': self.cfg.training.subsample_freq,
                                                'max_depth': self.cfg.training.max_depth,
                                                'num_leaves': self.cfg.training.num_leaves, 
                                                'colsample_bytree': self.cfg.training.colsample_bytree,
                                                'min_child_samples': self.cfg.training.min_child_samples,
                                                'verbose': self.cfg.training.verbose,
                                            },
                                       num_boost_round=self.cfg.training.num_boost_round,
                                       verbose_eval=self.cfg.training.verbose_eval,
                                       early_stopping_rounds=self.cfg.training.early_stopping_rounds,
                                      )

                self.models.append(fold_model)
                self.evals_results.append(evals_result_fold)

                y_oof_fold = fold_model.predict(X_valid_fold)
                self.y_oof[valid_idx] = y_oof_fold

            # save parameters
            mlflow.log_param('boosting_type', self.cfg.training.boosting_type)
            mlflow.log_param('learning_rate', self.cfg.training.learning_rate)
            mlflow.log_param('bagging_fraction', self.cfg.training.bagging_fraction)
            mlflow.log_param('lambda_l1', self.cfg.training.lambda_l1)
            mlflow.log_param('lambda_l2', self.cfg.training.lambda_l2)
            mlflow.log_param('metric', self.cfg.training.metric)
            mlflow.log_param('objective', self.cfg.training.objective)
            mlflow.log_param('random_state', self.cfg.training.random_state)
            mlflow.log_param('subsample', self.cfg.training.subsample)
            mlflow.log_param('subsample_freq', self.cfg.training.subsample_freq)
            mlflow.log_param('max_depth', self.cfg.training.max_depth)
            mlflow.log_param('num_leaves', self.cfg.training.num_leaves)
            mlflow.log_param('colsample_bytree', self.cfg.training.colsample_bytree)
            mlflow.log_param('min_child_samples', self.cfg.training.min_child_samples)
            mlflow.log_param('num_boost_round', self.cfg.training.num_boost_round)
            mlflow.log_param('early_stopping_rounds', self.cfg.training.early_stopping_rounds)
            mlflow.log_param('num_fold', self.cfg.training.num_fold)
            mlflow.log_param('CV', self.cfg.training.cv)
            # save metric
            if self.cfg.training.objective == 'regression':
                rmse = np.sqrt(mean_squared_error(self.y_train, self.y_oof))
                mlflow.log_metric('rmse_oof', rmse)
            if self.cfg.training.objective == 'binary':
                fpr, tpr, thresholds = metrics.roc_curve(self.y_train, self.y_oof)
                auc = metrics.auc(fpr, tpr)
                mlflow.log_metric('auc_oof', auc)
            if self.cfg.training.objective == 'multiclass':
                accuracy = accuracy_score(self.y_train, self.y_oof)
                mlflow.log_metric('accuracy_oof', accuracy)
            # save artifacts
            mlflow.log_artifact(self.cfg_path)
            # save model
            if log_model:
                for i in range(self.cfg.training.num_fold):
                    mlflow.lightgbm.log_model(self.models[i], 'lgb_model_{}'.format(i))
            
            return self.y_oof
        
    def predict(self, X, y=None):
        # test set
        self.X_test = X
        
        if y is not None:
            self.y_test = y
        
        # inference (average)
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
            self.y_pred = np.mean([model.predict(self.X_test) for model in self.models], axis=0)
            
            if y is not None:
                if self.cfg.training.objective == 'regression':
                    rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
                    mlflow.log_metric('rmse_pred', rmse)
                if self.cfg.training.objective == 'binary':
                    # roc-auc
                    fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self.y_pred)
                    auc = metrics.auc(fpr, tpr)
                    mlflow.log_metric('auc_pred', auc)
                    # confusion-matrix
                    y_pred_round = np.where(self.y_pred > 0.5, 1, 0)
                    cm = confusion_matrix(self.y_test, y_pred_round)
                    tn, fp, fn, tp = cm.flatten()
                    mlflow.log_metric('confusion_matrix - True Negative', tn)
                    mlflow.log_metric('confusion_matrix - False Positive', fp)
                    mlflow.log_metric('confusion_matrix - False Negative', fn)
                    mlflow.log_metric('confusion_matrix - True Positive', tp)

            return self.y_pred
        
    def groupkfold(self):
        pass
    
    def learning_curve(self):
        cols = 3
        if self.cfg.training.num_fold % cols != 0:
            rows = self.cfg.training.num_fold//cols+1
        else:
            rows = self.cfg.training.num_fold//cols
        
        fig = plt.figure(figsize=(25, 12))
        for i in range(len(self.evals_results)):
            ax = fig.add_subplot(rows, cols, i+1)
            lgb.plot_metric(self.evals_results[i], ax=ax)
            ax.set_title('Learning curve in Fold {}'.format(i+1))
        plt.tight_layout()
        plt.show();
        
    def feature_importance(self, importance_type='gain'):
        self.importance_type = importance_type
        cols = 2
        if self.cfg.training.num_fold % cols != 0:
            rows = self.cfg.training.num_fold//cols+1
        else:
            rows = self.cfg.training.num_fold//cols
        
        fig = plt.figure(figsize=(20, 40))
        for fold, model in enumerate(self.models):
            feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(importance_type=self.importance_type), 
                                                  model.feature_name()),reverse = True), 
                                       columns=['Value','Feature']).iloc[:20]
            ax = fig.add_subplot(rows, cols, fold+1)
            ax = sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title('Feature Importance in Fold {}'.format(fold + 1))
        plt.tight_layout()
        plt.show();
        
    def label_distribution(self, label=None, round=False, train=True):
        if round:
            cols = 3
            if train:
                label = self.y_train
                df_list = [pd.Series(self.y_oof), pd.Series(self.y_oof.round()), label]
                title = ['oof', 'round', 'label']
            else:
                self.label = label
                df_list = [pd.Series(self.y_pred), pd.Series(self.y_pred.round()), self.label]
                title = ['pred', 'round', 'label']           
        else:
            cols = 2
            if train:
                label = self.y_train
                df_list = [pd.Series(self.y_oof), label]
                title = ['oof', 'label']
            else:
                self.label = label
                df_list = [pd.Series(self.y_pred), self.label]
                title = ['pred', 'label']
        
        fig = plt.figure(figsize=(20, 5))
        for i, df in enumerate(df_list):
            ax = fig.add_subplot(1, cols, i+1)
            sns.distplot(df, kde=False, ax=ax)
            ax.set_title(title[i])
            ax.set_xlim(0, 1.0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
        plt.show();
        
    def shap_forceplot(self, plot_type):
        self.type = plot_type
        
        # load JS visualization code to notebook
        shap.initjs()
        
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(self.X_train)
        
        if self.type == 'single':
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            return shap.force_plot(explainer.expected_value, shap_values[0,:], self.X_train.iloc[0,:])
            
        elif self.type == 'all':
            return shap.force_plot(explainer.expected_value, shap_values, self.X_train)
        
    def shap_summaryplot(self):
        # load JS visualization code to notebook
        shap.initjs()

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(self.X_train)
        
        return shap.summary_plot(shap_values, self.X_train)
    
    def shap_dependenceplot(self, feature_name):
        self.feature_name = feature_name
        # load JS visualization code to notebook
        shap.initjs()

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(self.X_train)
        
        return shap.dependence_plot(self.feature_name, shap_values, self.X_train)
        
    def roc_curve(self, test_label=None, plot_type='test'):
        if test_label is not None:
            self.test_label = test_label
        if plot_type == 'test':
            predict = [self.y_pred]
            label = [self.test_label]
        elif plot_type == 'train':
            predict = [self.y_oof]
            label = [self.y_train]
        
        method_name = ['lgb']
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        for pred, label, method_name in zip(predict, label, method_name):
            fpr, tpr, thresholds = metrics.roc_curve(label, pred)
            auc = metrics.auc(fpr, tpr)
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=method_name)
            roc_display.plot(ax=ax)
            ax.set_title('ROC curve : LightGBM', fontsize=16)
        plt.show();
    
    def confusion_matrix(self, threshold, classes, test_label=None, plot_type='test', normalize=False, title=None, cmap=plt.cm.Blues):
        """
        Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if plot_type == 'train':
            y_true = self.y_train
            y_pred = np.where(self.y_oof > threshold, 1, 0)
        elif plot_type == 'test':
            y_true = test_label
            y_pred = np.where(self.y_pred > threshold, 1, 0)
        
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, fontsize=20)
        plt.yticks(tick_marks, fontsize=20)
        plt.xlabel('Predicted label',fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.title(title, fontsize=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.15)
        cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=20)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
    #            title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        fontsize=20,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        return ax
    
    def random_search(self):
        pass

def main():
    # dataset
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    y = pd.DataFrame(boston.target)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
    
    # training
    cfg_path = './config.yaml'
    trainer = LightgbmTrainer(cfg_path)
    y_oof = trainer.fit(X=X_train, y=y_train, log_model=True)

    # inference
    y_pred = trainer.predict(X_test, y_test)

if __name__ == '__main__':
    main()