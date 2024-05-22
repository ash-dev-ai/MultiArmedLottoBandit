import matplotlib.pyplot as plt
import seaborn as sns
from rfVis import plot_feature_importance, plot_predicted_vs_actual, plot_prediction_distribution

def plot_feature_importance(rf_model, feature_names):
    """
    Plot the feature importances for a given Random Forest model.

    Parameters:
        rf_model (RandomForestClassifier): The trained Random Forest model.
        feature_names (list): List of feature names.

    Returns:
        None
    """
    feature_importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_predicted_vs_actual(predictions, actual_values):
    """
    Plot the predicted values versus the actual values.

    Parameters:
        predictions (numpy array): The predicted values from the Random Forest model.
        actual_values (numpy array): The actual target values.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_values, predictions, alpha=0.6)
    plt.plot(actual_values, actual_values, color='red', linestyle='--')
    plt.title('Predicted Values vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def plot_prediction_distribution(predictions, actual_values):
    """
    Plot the distribution of predicted values and actual values.

    Parameters:
        predictions (numpy array): The predicted values from the Random Forest model.
        actual_values (numpy array): The actual target values.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions, label='Predicted Values', shade=True)
    sns.kdeplot(actual_values, label='Actual Values', shade=True)
    plt.title('Prediction Distribution')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

