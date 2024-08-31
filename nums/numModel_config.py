import os

class NumModelConfig:
    def __init__(self, model_type='gbc', dataset='mb', num_range=range(1, 72), numA_range=range(1, 27), model_params=None):
        self.model_type = model_type  # Model type (e.g., 'gbc', 'rf', 'svm')
        self.dataset = dataset
        self.num_range = num_range  # Numbers 1-71
        self.numA_range = numA_range  # Numbers A1-A26
        
        # Base directory for saving models
        self.base_dir = os.path.join('models', 'num', dataset, model_type)
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Use provided model parameters, or set to an empty dict if none provided
        self.model_params = model_params if model_params is not None else {}

        # Set up individual model configs
        self.models_config = self._generate_model_configs()

    def _generate_model_configs(self):
        configs = {}
        
        # For Num1-71
        for num in self.num_range:
            model_name = f"{self.dataset}_{self.model_type}_Num_{num}"
            configs[model_name] = self.model_params.copy()
            configs[model_name]['output_path'] = os.path.join(self.base_dir, f"{self.model_type}_Num_{num}_model.pkl")
        
        # For NumA1-26
        for num in self.numA_range:
            model_name = f"{self.dataset}_{self.model_type}_A_{num}"
            configs[model_name] = self.model_params.copy()
            configs[model_name]['output_path'] = os.path.join(self.base_dir, f"{self.model_type}_A_{num}_model.pkl")

        return configs

    def get_model_config(self, model_name):
        return self.models_config.get(model_name, None)

    def get_all_model_configs(self):
        return self.models_config
