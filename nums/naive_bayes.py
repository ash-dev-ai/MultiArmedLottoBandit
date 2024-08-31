import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib
from nums.numModel_config import NumModelConfig

class NaiveBayesModelTrainer:
    def __init__(self, dataset_name, root_dir):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(root_dir, 'data')
        self.model_dir = os.path.join(root_dir, 'models', 'num', dataset_name, 'naive_bayes')
        os.makedirs(self.model_dir, exist_ok=True)
        self.data = self.load_data()
        self.config = NumModelConfig(model_type='naive_bayes', dataset=dataset_name)

    def load_data(self):
        """Loads the dataset into a pandas DataFrame."""
        file_path = os.path.join(self.data_dir, f'data_{self.dataset_name}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The dataset {self.dataset_name} was not found at {file_path}")
        return pd.read_csv(file_path)

    def create_target_column(self, num, target_type='Num'):
        """Creates the target column based on whether the number appears in the next draw."""
        if target_type == 'Num':
            return self.data.apply(lambda row: int(num in [row['num1'], row['num2'], row['num3'], row['num4'], row['num5']]), axis=1)
        elif target_type == 'A':
            return (self.data['numA'] == num).astype(int)
        else:
            raise ValueError(f"Invalid target type {target_type}. Must be 'Num' or 'A'.")

    def train_and_save_model(self, model_name, model_params, target_column):
        """Trains the Naive Bayes model and saves it to a file."""
        features = self.data[['mean', 'median', 'std_dev', 'numSum', 'totalSum']]
        target = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        # Remove the 'output_path' key from model_params before model initialization
        model_params_copy = model_params.copy()
        model_params_copy.pop('output_path', None)

        model = GaussianNB(**model_params_copy)
        model.fit(X_train, y_train)

        model_output_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        joblib.dump(model, model_output_path)

    def train_models(self):
        """Main method to train and save Naive Bayes models for each number."""
        for model_name, model_params in self.config.get_all_model_configs().items():
            if 'Num_' in model_name:
                num = int(model_name.split('_')[-1])
                target_column = f'num_{num}_target'
                self.data[target_column] = self.create_target_column(num, target_type='Num')
            elif 'A_' in model_name:
                num = int(model_name.split('_')[-1])
                target_column = f'numA_{num}_target'
                self.data[target_column] = self.create_target_column(num, target_type='A')
            self.train_and_save_model(model_name, model_params, target_column)

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = ['pb', 'mb', 'comb']
    for dataset in datasets:
        trainer = NaiveBayesModelTrainer(dataset_name=dataset, root_dir=root_dir)
        trainer.train_models()

if __name__ == '__main__':
    main()
