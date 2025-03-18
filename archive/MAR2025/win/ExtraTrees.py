from sklearn.ensemble import ExtraTreesRegressor
from win.Trainer import WinModelTrainer

class ExtraTreesRegressorModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }

    def initialize_model(self):
        return ExtraTreesRegressor(**self.model_params)

    def train_models(self):
        self.train_all_targets()
