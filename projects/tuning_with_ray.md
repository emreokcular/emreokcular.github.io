# Hyperparameter Tuning for Neural Networks with Ray Tune

Author : *Emre Okcular*

Date : June 7st 2021

In this post, we will take a closer look at hyperparameters of a deep neural network such as listed below. Moreover, we will cover how to tune these parameters similar approach with scikit-learn GridSearch. [Ray Tune](https://docs.ray.io/en/master/tune/index.html) package is a Python library for experiment execution and hyperparameter tuning at any scale.

* Learning Rate
* Batch Size
* Dropout
* Number of Epochs
* Weight Decay


### Model Architecture
```
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class,d,M):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, M)
        self.layer_out = nn.Linear(M, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=d)
        self.batchnorm1 = nn.BatchNorm1d(M)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
```


### Parameter Space
```
param_space = {"lr": tune.grid_search([0.00003,0.0005, 0.001, 0.0007]), # Learning Rate
               "bs": tune.grid_search([8, 16, 32,64, 128 , 256,512]), # Batch Size
               "dr": tune.grid_search([0.3, 0.5,0.7,0.85]), # Dropout
               "m": tune.grid_search([30, 50, 100, 200, 300]), # Number of neurons
               "wd": tune.grid_search([0])} # Weight Decay for optimizer
```


```
print("Best config: ", analysis.get_best_config(mode ="max" ,metric="val_accuracy"))
```
```
Best config:  {'lr': 0.0005, 'bs': 128, 'dr': 0.3, 'm': 300, 'wd': 0}
```

#### Saving Results as DataFrame
```
analysis.dataframe().to_csv("df_ep_100.csv")
```

#### Loading old Results

```
analysis2 = Analysis("/Users/emre/ray_results/train_cancer_2021-05-01_22-32-12")
```