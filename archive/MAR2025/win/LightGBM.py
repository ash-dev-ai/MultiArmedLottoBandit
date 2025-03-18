from lightgbm import LGBMRegressor
from win.Trainer import WinModelTrainer

class LightGBMModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'objective': 'regression',
            'random_state': 42
        }

    def initialize_model(self):
        return LGBMRegressor(**self.model_params)

    def train_models(self):
        self.train_all_targets()
