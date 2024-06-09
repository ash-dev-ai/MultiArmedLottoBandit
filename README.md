---

# MultiArmedLotto

## Overview

The **MultiArmedLotto** project explores various machine learning and deep learning techniques to predict lottery outcomes using multi-armed bandit algorithms and a stacked ensemble of advanced models. This project involves creating, training, and evaluating multiple models on different datasets, with the aim to maximize the prediction accuracy for lottery draws.

Initially named bandit, but has subsequently grown. Archive has various other trials before this project on the same datasets.

## Project Structure

The project is organized into the following directories and scripts:

```
MultiArmedLottoBandit/
├── bandit/
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   ├── thompson_sampling.py
│   ├── reward_definition.py
│   ├── simulate.py
│   └── models/               # Models for bandit algorithms
├── chaos/
│   ├── chua.py
│   ├── henon.py
│   ├── logistic.py
│   ├── lorenz96.py
│   ├── rossler.py
│   └── vote.py
├── stacked/
│   ├── rnn_ensemble.py
│   ├── reservoir_ensemble.py
│   ├── deep_learning_ensemble.py
│   ├── meta_learner.py
│   └── models/               # Models for stacked ensembles
├── data/                      # Directory for storing datasets and predictions
│   ├── raw/                   # Raw datasets
│   ├── predictions/           # Prediction outputs
│   ├── train_combined.csv
│   ├── val_combined.csv
│   ├── test_combined.csv
│   ├── train_pb.csv
│   ├── val_pb.csv
│   ├── test_pb.csv
│   ├── train_mb.csv
│   ├── val_mb.csv
│   └── test_mb.csv
├── prep/
│   ├── data_split.py
│   ├── days.py
│   ├── export.py
├── bandit_main.py
├── stacked_main.py
└── README.md
```

## Installation

To get started with the project, follow these installation steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MultiArmedLottoBandit.git
   cd MultiArmedLottoBandit
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Here are some of the main packages used:
   - `numpy`
   - `pandas`
   - `tensorflow`
   - `keras`
   - `pyESN`
   - `reservoirpy`
   - `scikit-learn`
   - `matplotlib`
   - `tqdm`

## Usage

### Data Preparation

1. **Splitting and Preprocessing Data**:
   The `data_split.py` script in the `prep` directory splits the raw dataset into training, validation, and test sets. The `days.py` script adds a numerical representation of the days to the data.

   ```bash
   python prep/data_split.py
   python prep/days.py
   ```

2. **Exporting Data**:
   The `export.py` script saves the transformed datasets into CSV files for further analysis and modeling.

   ```bash
   python prep/export.py
   ```

### Model Training and Prediction

1. **Bandit Algorithms**:
   The bandit algorithms (ε-Greedy, UCB, Thompson Sampling) are implemented in the `bandit` directory. The `bandit_main.py` script trains these models and runs simulations to evaluate them.

   ```bash
   python bandit_main.py
   ```

2. **Stacked Ensemble**:
   The `stacked_main.py` script runs the ensemble models, which include recurrent neural networks, reservoir computing models, and deep learning models, along with a meta-learner to combine their predictions.

   ```bash
   python stacked_main.py
   ```

   This script calls the following:
   - `rnn_ensemble.py`: LSTM and GRU models.
   - `reservoir_ensemble.py`: Echo State Network (ESN) and Liquid State Machine (LSM) models.
   - `deep_learning_ensemble.py`: Physics-Informed Neural Networks (PINNs) and Deep Belief Networks (DBNs).
   - `meta_learner.py`: Combines predictions from the above models using a transformer model as the meta-learner.

### Visualization and Exploration

1. **Exploratory Data Analysis**:
   You can perform exploratory data analysis using scripts in the `explore` directory, which visualize the data and the model predictions.

   ```bash
   python explore/explore_main.py
   ```

## Project Components

### Bandit Algorithms

- **ε-Greedy Algorithm**: Chooses actions based on a balance between exploration and exploitation.
- **Upper Confidence Bound (UCB)**: Chooses actions with the highest upper confidence bound for the expected reward.
- **Thompson Sampling**: Chooses actions based on probability distributions of expected rewards.

### Stacked Ensemble Models

- **Recurrent Neural Networks**:
  - **LSTM**: Long Short-Term Memory networks for capturing long-range dependencies.
  - **GRU**: Gated Recurrent Units for a simpler and efficient alternative to LSTM.

- **Reservoir Computing**:
  - **ESN**: Echo State Networks, a type of recurrent neural network.
  - **LSM**: Liquid State Machines, a spiking neural network model.

- **Deep Learning**:
  - **PINN**: Physics-Informed Neural Networks, incorporating physical laws into the learning process.
  - **DBN**: Deep Belief Networks, a generative model with multiple layers of latent variables.

- **Meta-Learner**:
  - **Transformer Model**: Combines predictions from the above models to make final predictions.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push the branch to your forked repository.
5. Open a pull request to the main repository.

## License

None

## Acknowledgments

- Thanks to the contributors and the open-source community for providing the tools and libraries used in this project.

---