import os
import pandas as pd
import importlib
import joblib
import random
from datetime import timedelta, datetime

class Trial0:
    def __init__(self, dataset_name, root_dir, num_range):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(root_dir, 'data')
        self.meta_model_dir = os.path.join(root_dir, 'models', 'meta', dataset_name)
        self.num_range = num_range
        self.automata_rules = [30, 37, 42, 45, 73, 90, 110, 150, 254]

    def load_data(self):
        """Load the dataset into a pandas DataFrame."""
        file_path = os.path.join(self.data_dir, f"data_{self.dataset_name}.csv")
        return pd.read_csv(file_path)

    def load_meta_models(self):
        """Load the pre-trained meta-models for inference adjustments."""
        meta_models = {}
        targets = ['totalSum', 'numSum']
        for target in targets:
            model_path = os.path.join(self.meta_model_dir, f"meta_model_{target}.pkl")
            if os.path.exists(model_path):
                meta_models[target] = joblib.load(model_path)
                print(f"Loaded meta-model for {target} from {model_path}")
            else:
                print(f"Meta-model for {target} not found at {model_path}")
        return meta_models

    def get_next_draw_day(self, last_day):
        """Determine the next draw day based on the dataset's schedule."""
        draw_schedules = {
            'pb': [1, 3, 6],               # Powerball: Monday (1), Wednesday (3), Saturday (6)
            'mb': [2, 5],                  # Mega Millions: Tuesday (2), Friday (5)
            'combined': [1, 2, 3, 5, 6]    # Combined: Monday, Tuesday, Wednesday, Friday, Saturday
        }
        draw_days = draw_schedules.get(self.dataset_name, [])
        if not draw_days:
            raise ValueError(f"No draw schedule found for dataset {self.dataset_name}")

        today = datetime.now().weekday() + 1
        current_day = today if datetime.now().hour < 22 else (today % 7) + 1
        upcoming_days = [day for day in draw_days if day >= current_day]
        next_draw_day = min(upcoming_days) if upcoming_days else min(draw_days)

        days_until_next_draw = (next_draw_day - current_day) % 7
        next_draw_date = datetime.now() + timedelta(days=days_until_next_draw)

        return next_draw_date.strftime('%A')

    def generate_model_predictions(self, data, meta_models):
        last_date = pd.to_datetime(data['draw_date']).iloc[-1]
        next_draw_dates = [last_date + timedelta(days=day) for day in range(1, 4)]

        future_total_sum = pd.DataFrame({'ds': next_draw_dates})
        future_num_sum = pd.DataFrame({'ds': next_draw_dates})

        total_sum_forecast = meta_models['totalSum'].predict(future_total_sum)
        num_sum_forecast = meta_models['numSum'].predict(future_num_sum)

        total_sum_pred = total_sum_forecast['yhat'].iloc[0]
        numA_pred = num_sum_forecast['yhat'].iloc[0]

        predictions = [min(max(int(total_sum_pred / 5), 1), self.num_range[0]) for _ in range(5)]
        model_numA = min(max(int(numA_pred), 1), self.num_range[1])

        return {'num1-5': predictions, 'numA': model_numA}

    def get_adjustment_factor(self):
        """Returns a small random integer for controlled randomness, such as between -2 and 2."""
        return random.randint(-2, 2)

    def generate_adjusted_predictions(self, model_predictions, last_draw_day):
        """
        Generate adjusted predictions for each rule using the meta model as a guardrail.
        """
        adjusted_predictions = []
    
        for rule_number in self.automata_rules:
            rule_module_name = f"automata.Rule{rule_number}"
    
            try:
                rule_module = importlib.import_module(rule_module_name)
                rule_class = getattr(rule_module, f"Rule{rule_number}")
                rule_instance = rule_class(num_range=self.num_range)
    
                # Apply rule-specific logic starting from model predictions
                adjusted = rule_instance.apply_to_prediction(model_predictions)
    
                # Use the centralized get_adjustment_factor to introduce controlled randomness
                adjusted_predictions.append({
                    'num1': max(1, min(adjusted[0][0] + self.get_adjustment_factor(), self.num_range[0])),
                    'num2': max(1, min(adjusted[0][1] + self.get_adjustment_factor(), self.num_range[0])),
                    'num3': max(1, min(adjusted[0][2] + self.get_adjustment_factor(), self.num_range[0])),
                    'num4': max(1, min(adjusted[0][3] + self.get_adjustment_factor(), self.num_range[0])),
                    'num5': max(1, min(adjusted[0][4] + self.get_adjustment_factor(), self.num_range[0])),
                    'numA': max(1, min(adjusted[1] + self.get_adjustment_factor(), self.num_range[1])),
                    'next_draw_day': self.get_next_draw_day(last_draw_day),
                    'type': f'T-0-A-{rule_number}'
                })
    
            except (ModuleNotFoundError, AttributeError) as e:
                print(f"Error loading {rule_module_name}: {e}")
    
        return adjusted_predictions