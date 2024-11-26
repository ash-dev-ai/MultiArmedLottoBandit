from sklearn.linear_model import ElasticNet
from win.Trainer import WinModelTrainer

class ElasticNetModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'max_iter': 1000,
            'random_state': 42
        }

    def initialize_model(self):
        return ElasticNet(**self.model_params)

    def train_models(self):
        self.train_all_targets()
