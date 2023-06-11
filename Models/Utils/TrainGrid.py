from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from .CrossValidation import CrossValidation
from Models.Utils.CrossValidation import CrossValidation
from Tools.ToolsModels import is_regression


class TrainGrid:
    RANDOM_STATE = 500
    N_ITER = 4  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    N_JOBS = 4  # Number of jobs to run in parallel. None means 1 unless in
    N_SPLIT = 5  # Number of jobs to run in parallel. None means 1 unless in


    def __init__(self, cv):
        self.cv = cv

    def get_scorer(self, model):
        if is_regression(model):
            return "r2"
        return None

    def print_search(self, src):
        print("__________-")
        print('Best estimator across searched params: ', src.best_estimator_)
        print('Best score across searched params:', src.best_score_)
        print('Best parameters across searched params:', src.best_params_)
        print('Mean test score:', src.cv_results_['mean_test_score'])
        print("__________-")

    def train_random(
            self,
            model,
            grid_parameters,
            xtr,
            ytr,
            n_iter=N_ITER,
            n_jobs=N_JOBS,
            n_split=N_SPLIT,
            verbose=0,
            random_state=RANDOM_STATE):

        """
              with the random parameters it generates several runs by mixing them randomly
        """
        random_src = RandomizedSearchCV(
                         model,
                         param_distributions = grid_parameters,
                         n_iter = n_iter,
                         n_jobs = n_jobs,
                         scoring = self.get_scorer(model),
                         cv = self.cv(xtr, ytr, n_splits=n_split, random_state=random_state),
                         verbose = 1,
                         random_state = random_state
                     )

        random_src.fit(xtr, ytr)
        self.print_search(random_src)
        return random_src.best_params_

    def train_grid(self,
                   model,
                   grid_parameters,
                   xtr,
                   ytr,
                   n_iter=N_ITER,
                   n_split=N_SPLIT,
                   n_jobs=N_JOBS,
                   verbose=0,
                   random_state=RANDOM_STATE):
        """
             With the grid parameters generates as many runs as possible combinations to find the best
        """
        grid_src = GridSearchCV(
            model,
            param_grid = grid_parameters,
            n_jobs = n_jobs,
            scoring = self.get_scorer(model),
            cv = self.cv(xtr, ytr, n_splits=n_split, random_state=random_state),
            verbose = 1
        )

        grid_src.fit(xtr, ytr)
        self.print_search(grid_src)
        return grid_src.best_params_

