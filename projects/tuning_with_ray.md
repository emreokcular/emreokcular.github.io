# Hyperparameter Tuning for Neural Networks with Ray Tune

Author : *Emre Okcular*

Date : June 7st 2021

![5percent](/resources/mixer.jpg)

In this post, we will take a closer look at hyperparameters of a deep neural network listed below. Moreover, we will cover how to tune these parameters similar approach with scikit-learn GridSearch. [Ray Tune](https://docs.ray.io/en/master/tune/index.html) package is a Python library for experiment execution and hyperparameter tuning at any scale.

**Important hyperparameters for Neural Networks:**
* Learning Rate
* Dropout
* Weight Decay
* Batch Size
* Number of Epochs

#### Learning Rate

Learnin rate is the most important hyper-parameter to tune for training neural networks. If learning rates are too small it causes slow training, if learning rates are too large is causes training to diverge. As an initial approach we can fix the learning rate and then monotonically decrease during training.

#### Dropout

Dropout is a regularization technique that prevents overfitting in neural networks. At each training stage (batch), individual nodes are either dropped out of the net with probability p or kept with probability 1-p, so that a reduced network is left. 
**Training Phase**
For each hidden layer, for each training sample, for each iteration, ignore (zero out) a random fraction, p, of nodes (and corresponding activations).
**Testing Phase**
Use all activations, but reduce them by a factor 1-p (to account for the missing activations during training).

Neural networks learn co-adaptations of hidden units that work for the training data but do not generalize to unseen data.Random dropout breaks up co-adaptations making the presence of any particular hidden unit unreliable.
 
#### Weight Decay

It is usually a parameter in the optimization function such as ```torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)``` . In other words it is a L2 penalty that is added to error function. Larger values of λ will tend to shrink the weights toward zero: typically cross-validation is used to estimate λ.

#### Batch Size

Batch size is a number of samples processed before the model is updated.

**Why to use minibatch?**
* How much data can you fit on your GPU.
* Small batches can offer a regularizing effect, possibly due to the noise they add to the learning process.
* The model update frequency is higher with mini-batches.
* Larger batches provide more accurate estimate of the gradient.
 
#### Number of Epochs

Number of epochs is the number of complete passes through the training dataset.

While tuning these parameters manually, you might feel like a gradient decent in multidimensional space looking for some pattern(increase or decrease). In scikit-learn pipelines we use methods like ```HalvingGridSearchCV``` , ```RandomizedSearchCV``` , ```GridSearchCV``` to solve this search problem. It may take some time and CPU heat with big datasets but eventually you will have tuned model with best estimator and hyperparameters. With the same approach we can tune neural network parameters using Ray Tune package in efficient and distributed way.

We will import below packages from PyTorch and Ray.

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ray import tune # For hyperparameter tuning in Neural Networks.
from ray.tune import Analysis # For analyzing tuning results.
```

Let use a simple NN model architecture with one layer, dropout, ReLU activation function and batchnorm. 

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

To use tuning method we need to convert our training code to a function and add ```tune.report(training_accuracy=train_epoch_acc/len(train_loader),val_accuracy=val_epoch_acc/len(val_loader),training_loss=train_epoch_loss/len(train_loader),val_loss=val_epoch_loss/len(val_loader)``` to our training loop to store the metrics for each epoch.

```
def train(model,optimizer,train_loader,val_loader,EPOCHS,accuracy_stats,loss_stats):
    for e in tqdm(range(1, EPOCHS+1)):
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch, y_train_batch
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # VALIDATION    
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch, y_val_batch

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        tune.report(training_accuracy=train_epoch_acc/len(train_loader),val_accuracy=val_epoch_acc/len(val_loader),training_loss=train_epoch_loss/len(train_loader),val_loss=val_epoch_loss/len(val_loader))
    return accuracy_stats
```

```
def get_loaders():
    df = pd.read_csv("/Users/emre/Dev/GitHub/train_ml2_2021.csv")
    y=df['target']
    X=df.drop(columns='target')
    test = pd.read_csv('/Users/emre/Dev/GitHub/test0.csv')
    X_test = test.drop(columns=['target',"obs_id"])
    y_test = test["target"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    train_dataset = CancerDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = CancerDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = CancerDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]
    c = Counter(y_train)
    class_count = [i for i in [i[1] for i in sorted(c.items())]]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)
    return weighted_sampler, train_dataset, val_dataset,len(X.columns),y.nunique()
```

Finally, we will use train function to wrap all the helpers.

```
def train(config):
    test = []
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    w,tr,val,num_feat, num_cl = get_loaders()
    train_loader = DataLoader(dataset=tr,
                          batch_size=config["bs"],
                          sampler=w, drop_last=True)
    val_loader = DataLoader(dataset=val,shuffle=True, batch_size=1)
    model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES,d=config["dr"],M=config["m"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],weight_decay = config["wd"])
    train(model, optimizer, train_loader, val_loader,EPOCHS=EPOCH_FOR_TUNING,accuracy_stats=accuracy_stats,loss_stats=loss_stats)  ##EPOCH is hardcoded!
```

Before hyperparameter search we need to specify the search space in a dictionary.

```
param_space = {"lr": tune.grid_search([0.00003,0.0005, 0.001, 0.0007]), # Learning Rate
               "bs": tune.grid_search([8, 16, 32,64, 128 , 256,512]), # Batch Size
               "dr": tune.grid_search([0.3, 0.5,0.7,0.85]), # Dropout
               "m": tune.grid_search([30, 50, 100, 200, 300]), # Number of neurons
               "wd": tune.grid_search([0])} # Weight Decay for optimizer
```

When we run below method it will start to search parameter space by creatgin folders in the root library.

```analysis = tune.run(train, config=param_space,verbose=1, name="epoch_"+str(EPOCH_FOR_TUNING))```

You can see the folders in the path ```/Users/emre/ray_results/epoch_300``` Here ```epoch_300``` is the name for tuning process we specified above in the run() method.

```
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00000_0_bs=8,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00001_1_bs=16,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00002_2_bs=32,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00003_3_bs=64,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00004_4_bs=128,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00005_5_bs=256,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00007_7_bs=8,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00008_8_bs=16,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00009_9_bs=32,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-08
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00010_10_bs=64,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-35
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00011_11_bs=128,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-35
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00012_12_bs=256,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-36
drwxr-xr-x   8 emre  staff      256  6 May 11:35 train_cancer_c3378_00013_13_bs=512,dr=0.5,lr=3e-05,m=30,wd=0_2021-05-06_11-35-41
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00014_14_bs=8,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-35-47
drwxr-xr-x   7 emre  staff      224  6 May 11:35 train_cancer_c3378_00015_15_bs=16,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-35-53
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00016_16_bs=32,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-35-53
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00017_17_bs=64,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-36-05
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00018_18_bs=128,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-36-07
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00019_19_bs=256,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-36-19
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00021_21_bs=8,dr=0.85,lr=3e-05,m=30,wd=0_2021-05-06_11-36-29
drwxr-xr-x   8 emre  staff      256  6 May 11:36 train_cancer_c3378_00020_20_bs=512,dr=0.7,lr=3e-05,m=30,wd=0_2021-05-06_11-36-29
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00022_22_bs=16,dr=0.85,lr=3e-05,m=30,wd=0_2021-05-06_11-36-31
drwxr-xr-x   7 emre  staff      224  6 May 11:36 train_cancer_c3378_00023_23_bs=32,dr=0.85,lr=3e-05,m=30,wd=0_2021-05-06_11-36-37
drwxr-xr-x   2 emre  staff       64  6 May 11:36 train_cancer_c3378_00024_24_bs=64,dr=0.85,lr=3e-05,m=30,wd=0_2021-05-06_11-36-56
-rw-r--r--   1 emre  staff  2108895  6 May 11:36 experiment_state-2021-05-06_11-34-59.json
-rw-r--r--   1 emre  staff     6215  6 May 11:36 basic-variant-state-2021-05-06_11-34-59.json
drwxr-xr-x   8 emre  staff      256  6 May 17:52 train_cancer_c3378_00006_6_bs=512,dr=0.3,lr=3e-05,m=30,wd=0_2021-05-06_11-35-01
```

Lets take a look at one of the folder.

```
-rw-r--r--   1 emre  staff      63  6 May 11:35 params.json
-rw-r--r--   1 emre  staff      64  6 May 11:35 params.pkl
-rw-r--r--   1 emre  staff   80366  6 May 11:36 progress.csv
-rw-r--r--   1 emre  staff  208199  6 May 11:36 result.json
-rw-r--r--   1 emre  staff  169618  6 May 11:36 events.out.tfevents.1620326101.Emre-MacBook-Pro.local
```

You can find NN parameters, weights, model as pkl object, and logs.


```
print("Best config: ", analysis.get_best_config(mode ="max" ,metric="val_accuracy"))
```
```
Best config:  {'lr': 0.0005, 'bs': 128, 'dr': 0.3, 'm': 300, 'wd': 0}
```

#### Saving Results as DataFrame

|    |   training_accuracy |   val_accuracy |   training_loss |   val_loss |   time_this_iter_s | done   |   timesteps_total |   episodes_total |   training_iteration | experiment_id                    | date                |   timestamp |   time_total_s |   pid | hostname               | node_ip       |   time_since_restore |   timesteps_since_restore |   iterations_since_restore | trial_id    |   config/bs |   config/dr |   config/lr |   config/m |   config/wd | logdir                                                                                                                                      |\n|---:|--------------------:|---------------:|----------------:|-----------:|-------------------:|:-------|------------------:|-----------------:|---------------------:|:---------------------------------|:--------------------|------------:|---------------:|------:|:-----------------------|:--------------|---------------------:|--------------------------:|---------------------------:|:------------|------------:|------------:|------------:|-----------:|------------:|:--------------------------------------------------------------------------------------------------------------------------------------------|\n|  0 |             94.1897 |        59.8075 |        0.114618 |   1.14266  |           0.766135 | False  |               nan |              nan |                   20 | 8e1d3a2b785b487b820a4662a0e3607c | 2021-05-01_22-55-45 |  1619934945 |        19.5976 | 21455 | Emre-MacBook-Pro.local | 192.168.0.155 |              19.5976 |                         0 |                         20 | bf27d_00237 |         128 |         0.1 |      0.001  |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00237_237_bs=128,dr=0.1,lr=0.001,m=50,wd=0_2021-05-01_22-55-13  |\n|  1 |             84.9286 |        60.4091 |        0.303376 |   0.897925 |           0.665056 | False  |               nan |              nan |                   20 | ce123176b3ff4e858159e92cde518bf7 | 2021-05-01_23-00-29 |  1619935229 |        16.8075 | 21618 | Emre-MacBook-Pro.local | 192.168.0.155 |              16.8075 |                         0 |                         20 | bf27d_00279 |         512 |         0.3 |      0.0007 |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00279_279_bs=512,dr=0.3,lr=0.0007,m=50,wd=0_2021-05-01_23-00-01 |\n|  2 |             77.1693 |        57.7617 |        0.574762 |   1.00483  |           4.9946   | False  |               nan |              nan |                   20 | 3f27b2947d4b4526bf2b204c5c57a3e7 | 2021-05-01_22-39-17 |  1619933957 |        89.6643 | 20726 | Emre-MacBook-Pro.local | 192.168.0.155 |              89.6643 |                         0 |                         20 | bf27d_00064 |           4 |         0   |      0.0007 |         10 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00064_64_bs=4,dr=0,lr=0.0007,m=10,wd=0_2021-05-01_22-37-40      |\n|  3 |             88.9828 |        58.8448 |        0.250111 |   1.05108  |           0.772922 | False  |               nan |              nan |                   20 | 3c1df9f1e114480f9a2df855118b0340 | 2021-05-01_22-56-34 |  1619934994 |        18.7308 | 21494 | Emre-MacBook-Pro.local | 192.168.0.155 |              18.7308 |                         0 |                         20 | bf27d_00245 |         128 |         0.3 |      0.001  |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00245_245_bs=128,dr=0.3,lr=0.001,m=50,wd=0_2021-05-01_22-56-03  |\n|  4 |             82.5714 |        56.1974 |        0.374109 |   1.04833  |           0.672426 | False  |               nan |              nan |                   20 | 0d770e8147fc41609e1a395d782fe162 | 2021-05-01_22-53-19 |  1619934799 |        17.5119 | 21366 | Emre-MacBook-Pro.local | 192.168.0.155 |              17.5119 |                         0 |                         20 | bf27d_00215 |         512 |         0.3 |      0.0005 |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00215_215_bs=512,dr=0.3,lr=0.0005,m=50,wd=0_2021-05-01_22-52-54 |\n|  5 |             87.5391 |        65.4633 |        0.326149 |   0.913706 |           3.58599  | False  |               nan |              nan |                   20 | e77d7a68182f420888f9fceeab1c3048 | 2021-05-01_22-42-33 |  1619934153 |        65.926  | 20886 | Emre-MacBook-Pro.local | 192.168.0.155 |              65.926  |                         0 |                         20 | bf27d_00105 |           8 |         0.1 |      0.0005 |         30 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00105_105_bs=8,dr=0.1,lr=0.0005,m=30,wd=0_2021-05-01_22-41-22   |\n|  6 |             67.729  |        58.9651 |        0.776547 |   0.978256 |           2.14465  | False  |               nan |              nan |                   20 | 1d771b0b3f2744449c3cc617b6ea2e51 | 2021-05-01_23-45-55 |  1619937955 |        44.7988 | 20661 | Emre-MacBook-Pro.local | 192.168.0.155 |              44.7988 |                         0 |                         20 | bf27d_00048 |           4 |         0.3 |      0.001  |         10 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00048_48_bs=4,dr=0.3,lr=0.001,m=10,wd=0_2021-05-01_22-36-23     |\n|  7 |             80.6374 |        61.4922 |        0.474486 |   0.966504 |           5.03751  | False  |               nan |              nan |                   20 | 640c8da2886d4c9fb7fcff210521d8a6 | 2021-05-01_22-45-38 |  1619934338 |       106.718  | 20988 | Emre-MacBook-Pro.local | 192.168.0.155 |             106.718  |                         0 |                         20 | bf27d_00128 |           4 |         0   |      0.001  |         30 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00128_128_bs=4,dr=0,lr=0.001,m=30,wd=0_2021-05-01_22-43-45      |\n|  8 |             89.5862 |        58.3634 |        0.209558 |   1.00636  |           0.695875 | False  |               nan |              nan |                   20 | 29ca277421314d968a4c5b30aef04511 | 2021-05-01_22-48-50 |  1619934530 |        17.1981 | 21160 | Emre-MacBook-Pro.local | 192.168.0.155 |              17.1981 |                         0 |                         20 | bf27d_00174 |         256 |         0.1 |      0.0007 |         30 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00174_174_bs=256,dr=0.1,lr=0.0007,m=30,wd=0_2021-05-01_22-48-13 |\n|  9 |             77      |        50.0602 |        0.514854 |   1.14169  |           0.71833  | False  |               nan |              nan |                   20 | abb50c671be44c0da1c8756bb651e598 | 2021-05-01_22-54-14 |  1619934854 |        17.6418 | 21402 | Emre-MacBook-Pro.local | 192.168.0.155 |              17.6418 |                         0 |                         20 | bf27d_00223 |         512 |         0.5 |      0.0005 |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00223_223_bs=512,dr=0.5,lr=0.0005,m=50,wd=0_2021-05-01_22-53-47 |\n| 10 |             74.0627 |        60.1685 |        0.69363  |   0.926014 |           5.90874  | False  |               nan |              nan |                   20 | 9a8682fc71b3482697d757a8b2e9efa1 | 2021-05-01_22-58-40 |  1619935120 |       122.062  | 21499 | Emre-MacBook-Pro.local | 192.168.0.155 |             122.062  |                         0 |                         20 | bf27d_00248 |           4 |         0.5 |      0.001  |         50 |           0 | /Users/emre/ray_results/train_cancer_2021-05-01_22-32-12/train_cancer_bf27d_00248_248_bs=4,dr=0.5,lr=0.001,m=50,wd=0_2021-05-01_22-56-34    |

You can save the results as a dataframe.

```
analysis.dataframe().to_csv("df_ep_100.csv")
```

For example, you can draw accuracy and loss graphs using this data.

<center><img src="/resources/acc_val.png" width="50%" and height="50%"></center>

#### Loading old Results

If you want to take a look at previous tuning analysis you can load with below method.

```
analysis2 = Analysis("/Users/emre/ray_results/train_cancer_2021-05-01_22-32-12")
```

To wrap up, hyperparameter tuning in neural networks is an exhaustive search with lots of time and CPU power. Be careful with the parameter space that you used. It may give you the best parameters in somewhere in the parameter space which are not related with the exact solution.

Thank you for reading.