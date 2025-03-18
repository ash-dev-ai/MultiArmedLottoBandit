from catboost import CatBoostRegressor
from win.Trainer import WinModelTrainer

class CatBoostModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'RMSE',
            'random_seed': 42
        }

    def initialize_model(self):
        return CatBoostRegressor(**self.model_params)

    def train_models(self):
        self.train_all_targets()
