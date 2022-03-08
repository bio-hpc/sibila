from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from Models.Utils.CrossValidation import CrossValidation
from Tools.ToolsModels import is_regression


class TrainGrid:
    RANDOM_STATE = 500
    N_ITER = 4  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    N_JOBS = 4  # Number of jobs to run in parallel. None means 1 unless in
    N_SPLIT = 5  # Number of jobs to run in parallel. None means 1 unless in

    def train_random(
            self,
            model,
            grid_parameters,
            xtr,
            ytr,
            scoring='accuracy',
            #scoring='f1_macro',
            n_iter=N_ITER,
            n_jobs=N_JOBS,
            n_split=N_SPLIT,
            verbose=0,
            random_state=RANDOM_STATE):

        if is_regression(model):
            #    For the regression algorithms the scoring function of r2 is used
            scoring = "r2"
        """
              with the grid parameters it generates several runs by mixing them randomly
        """

        aux_model = RandomizedSearchCV(model,
                                       param_distributions=grid_parameters,
                                       n_iter=n_iter,
                                       n_jobs=n_jobs,
                                       scoring=scoring,
                                       cv=CrossValidation.group_kfold(xtr,
                                                                      ytr,
                                                                      n_splits=n_split,
                                                                      random_state=random_state),
                                       verbose=verbose,
                                       random_state=random_state)

        aux_model.fit(xtr, ytr)
        print("__________-")
        print(aux_model.cv_results_['mean_test_score'])
        print("__________-")
        return aux_model.best_params_

    def train_grid(self,
                   model,
                   grid_parameters,
                   xtr,
                   ytr,
                   scoring='accuracy',
                   n_iter=N_ITER,
                   n_split=N_SPLIT,
                   n_jobs=N_JOBS,
                   verbose=0,
                   random_state=RANDOM_STATE):
        """
             With the grid parameters generates as many runs as possible combinations to find the best
        """

        if is_regression(model):
            #    For the regression algorithms the scoring function of r2 is used
            scoring = "r2"
        aux_model = GridSearchCV(
            model,
            param_grid=grid_parameters,
            n_jobs=n_jobs,
            scoring=scoring,
            #cv=CrossValidation.stratified_kfold(xtr, ytr, n_splits=n_split, shuffle=True, random_state=random_state),
            cv=CrossValidation.group_kfold(xtr, ytr, n_splits=n_split, random_state=random_state),
            verbose=verbose,
        )
        aux_model.fit(xtr, ytr)
        print("__________-")
        print(aux_model.cv_results_['mean_test_score'])
        print("__________-")
        return aux_model.best_params_
