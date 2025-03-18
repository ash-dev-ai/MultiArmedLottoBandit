from sklearn.neural_network import MLPRegressor
from win.Trainer import WinModelTrainer

class MLPModelTrainer(WinModelTrainer):
    def __init__(self, dataset_name, root_dir):
        super().__init__(dataset_name, root_dir)
        self.model_params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42
        }

    def initialize_model(self):
        return MLPRegressor(**self.model_params)

    def train_models(self):
        self.train_all_targets()
