# Project 1: COVID-19 Patient Risk Classification

## Project Overview
The goal of this project was to analyze the **COVID-19 dataset** available on Kaggle ([link](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)), which contains information about patients and their symptoms.  
Based on these data, we aimed to create a **binary classification model** capable of predicting whether a given patient belongs to the **high-risk** or **low-risk** group.

The dataset was preprocessed for machine learning tasks, and a neural network was designed to perform binary classification. Through this project, we aimed to understand the entire pipeline -> from raw data to a functioning predictive model -> to experiment with performance-improving techniques such as **dropout**, **batch normalization**, **skip connections**, and **bottleneck layers**.

This project builds upon the theoretical background provided by the course **ZNEUS (Fundamentals of Neural Networks and Artificial Systems)** and utilizes the following tools:
- **Python**
- **PyTorch**
- **Pandas**
- **Matplotlib**
- **Weights & Biases (wandb)** for experiment tracking.

---

## Exploratory Data Analysis (EDA)

### Dataset Description
The original dataset contained **over one million records** and **21 attributes**.  
Most features were **binary**, representing symptoms, chronic diseases (such as asthma, hypertension, or obesity), or lifestyle factors (such as smoking or pregnancy).  
Basic demographic information included **age** and **gender**.

![Correlation_original](img/corr_1.png)

### Data Cleaning and Preparation
We began by detecting abnormal and outlier values using **boxplots**.  
Because most attributes were binary, **Gaussian normalization** was unnecessary.  
The dataset was relatively **balanced** between risk classes, and **feature correlations** were weak.

**Missing and invalid values** (NaN) were removed, as well as unrealistic age values (for example, 120 years).  
We applied the **Interquartile Range (IQR)** method to remove outliers, keeping the lower limit (age = 0) to include newborns.

Some missing or invalid values were encoded as **97**, **98**, or **99**, with 97 indicating “not applicable” (for instance, pregnancy in male patients). These were replaced with **0** to make the data consistent.  
Logical inconsistencies were also fixed for example, if a patient received home care (`MEDICAL_UNIT = 1`), attributes like **ICU** or **Intubed** could not be active.

![Dataset_attributes](img/data_distribution.png)

Wherever necessary, we **recoded binary variables** so that all used `1` for “yes/true” and `0` for “no/false” for better interpretability. After this the correlation matrix looked much more different and lost a track of previous correlations. We tried to examine if it had any impact on final model, but it did not.

![Correlation_cleaned](img/corr_2.png)

### Target Variable Definition
The original attribute **CLASSIFICATION_FINAL** indicated patient risk categories from 0 to 9.  
We reformulated the task into **binary classification**:
- Values **0–3** → **High risk (1)**
- Values **4+** → **Low risk (0)**

The **DATE_DIED** attribute was transformed into a binary feature **DIED**, where:
- `1` = patient died  
- `0` = patient survived  
(The value `9999-99-99` indicated survival in the original dataset.)

The **MEDICAL_UNIT** feature (hospital ID) showed no statistical correlation with other variables and was therefore removed.

After cleaning and transformation, we obtained a dataset of approximately **193,000 records**, ready for training.

---

## Model Development

### Model Architecture
A fully connected feed-forward neural network was implemented in **PyTorch** for binary classification.  
We developed a reusable function for dynamically constructing neural networks with varying numbers of layers and hidden units, allowing for flexible **hyperparameter sweeping**.

```
import torch.nn as nn

class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()

    def sweeping_build(self, input_size, output_size, hidden_size: list, activation_f, n_layers, dropout): # TODO: add dropout
        """
        Builds neural network from inputed parameters. Used for hyperparameter sweep
        :param input_size:
        :param output_size:
        :param hidden_size:
        :param activation_f:
        :param n_layers:
        :param dropout:
        """
        layers = []
        in_size = input_size
        for i in range(0, n_layers): # adjusted for input layer
            out_size = hidden_size[i]
            layers.append(nn.Linear(in_size, out_size))
            if activation_f == 'relu': layers.append(nn.ReLU())
            if activation_f == 'tanh': layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout)) # adding dropout layer before last linear layer
            in_size = out_size

        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)


    def add_configuration(self, network):
        self.net = network

    def forward(self, x):
        return self.net(x)
```
## Experiments and Observations

### Hyperparameter Sweep
To identify optimal hyperparameters, we used **Weights & Biases (wandb)** and performed a **hyperparameter sweep** — a systematic process of testing multiple combinations of learning rates, layer sizes, dropout rates, and optimizers.

We explored parameters such as:
- **Learning rate**
- **Batch size**
- **Number of layers**
- **Dropout probability**
- **Activation function**
- **Optimizer type**


```
#Hyperparameter configuration
#Random search
sweep_config = {
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
            "hidden_size_1": {"values": [32, 64, 128, 248]},
        "hidden_size_2": {"values": [32, 64, 128]},
        "hidden_size_3": {"values": [32, 64, 128]},
        "hidden_size_4": {"values": [32, 64, 128]},
        "hidden_size_5": {"values": [32, 64, 128]},
        "hidden_size_6": {"values": [32, 64, 128]},
        "n_layers": {"values": [2, 4, 6]},
        # "dropout": {"min": 0.0, "max": 0.5},
        "dropout": {"values": [0]},
        "optimizer": {"values": ["Adam"]},
        "activation": {"values": ["relu", "tanh"]},
        "batch_size": {"values": [2048]},
    }
}
```
![Grid_sweep](img/grid_sweep.png)
![Random_sweep](img/random_sweep.png)
![img.png](img/natural_select.png)
_grid and random sweeps show the accuracy plateau of 64% after selecting the most fit candidates from previous sweeps. 


![](img/random_sweep_droput.png)
![](img/random_sweep_optim.png)
![](img/random_sweep_last.png)

_metrics of correlation_
### Key Findings

[//]: # (- **Learning rate** needed to be small due to the large dataset size.  )
- **Few epochs** were sufficient for convergence; training longer did not significantly improve results.  
- The **first hidden layer** should be **larger than the input vector**, which improved feature representation.  
- The **SGD optimizer** showed **negative correlation** with validation performance.  
- **Adam optimizer consistently outperformed all others**, leading to the best results.
- The best activation function proved to be **tanh**
- The **number of layers** had little to no correlation with performance.  
- **Batch size** had a **positive correlation** with model accuracy.  
- **Dropout** negatively affected generalization.  

Final validation accuracy achieved: **~0.64**

Attempts to implement **bottleneck connections** (as used in autoencoders) negatively affected model performance.  
The reduced representational capacity limited feature propagation, leading to weaker generalization.


## Conclusions
The final model achieved an accuracy of approximately **64%**, which, given the complexity and potential noise in the dataset, is a reasonable baseline.
Despite the limitations, the project successfully demonstrated the full workflow — from **data exploration** and **preprocessing** to **model design**, **training**, and **evaluation** — while applying modern machine learning and experiment tracking tools.

---
