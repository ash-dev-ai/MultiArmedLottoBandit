from sklearn.linear_model import BayesianRidge
from win.Trainer import WinModelTrainer

class BayesianRidgeModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'max_iter': 300,  
            'tol': 1e-3,
            'alpha_1': 1e-6,  
            'alpha_2': 1e-6,  
            'lambda_1': 1e-6,
            'lambda_2': 1e-6
        }

    def initialize_model(self):
        return BayesianRidge(**self.model_params)

    def train_models(self):
        self.train_all_targets()
