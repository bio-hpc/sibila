# Import svm model
from sklearn import svm
from .BaseModel import *
from os.path import join
from joblib import dump
from Tools.TypeML import TypeML

#### kernel "linear", "poly", "rbf", "sigmoid", "precomputed", default="rbf"
PREFIX_OUT_SVM = '{}_{}_{}'  # Model, Dataset, numero de arboles, numero de profundidad, RANDOM_STATE
REMOVE_PARAMS_REGRESSOR = ['decision_function_shape', 'probability']


class SVM(BaseModel):
    def __init__(self, io_data, cfg, id_list):
        super(SVM, self).__init__(io_data, cfg, id_list)

        if self.cfg.get_params()['type_ml'].lower() == TypeML.CLASSIFICATION.value:
            self.model = svm.SVC(**self.cfg.get_params()['params'])
        elif self.cfg.get_params()['type_ml'].lower() == TypeML.REGRESSION.value:
            for i in REMOVE_PARAMS_REGRESSOR:
                if i in self.cfg.get_params()['params']:
                    del self.cfg.get_params()['params'][i]
                if i in self.cfg.get_params()['params_grid']:
                    del self.cfg.get_params()['params_grid'][i]
            self.model = svm.SVR(**self.cfg.get_params()['params'])
        else:
            print("Error: type_model not found ")
            exit()

    def get_prefix(self):
        return join(
            self.cfg.get_folder(),
            PREFIX_OUT_SVM.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset(),
                                  self.cfg.get_params()['params']['kernel']))

    def train(self, xtr, ytr):
        self.model_fit(xtr, ytr)
        #################################
        self.xtr = xtr
        self.ytr = ytr
        #################################

    def predict(self, xts):
        ypr = self.model_predict(xts)

        #################################
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        from sklearn.inspection import DecisionBoundaryDisplay

        _, ax = plt.subplots()
        DecisionBoundaryDisplay.from_estimator(
            self.model,
            self.xtr,
            cmap = cmap_light,
            ax = ax,
            response_method = "predict",
            plot_method = "pcolormesh",
            xlabel = 'eje x',
            ylabel = 'eje y',
            shading = "auto",
        )

        # Plot also the training points
        sns.scatterplot(
            x = self.xtr[:, 0],
            y = self.xtr[:, 1],
            hue = self.id_list[y],
            palette = cmap_bold,
            alpha = 1.0,
            edgecolor = "black",
        )
        plt.title("titulo")
        plt.savefig('svm_hyperplane.png', dpi=900)

        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=np.squeeze(self.xtr), y=self.ytr, hue=self.ytr, s=8)

        w = self.model.coef_[0]           # w consists of 2 elements
        b = self.model.intercept_[0]      # b consists of 1 element
        x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
        y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
        # Plotting a red hyperplane
        plt.plot(x_points, y_points, c='r')
        plt.savefig('svm_hyperplane.png', dpi=900)
        """
        #################################

        return ypr
