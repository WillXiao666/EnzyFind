#!/usr/bin/env python
# coding: utf-8

# # Training gradient boosting model for enzyme-substrate pair prediction with ESM-1b-vectors

# ### 1. Loading and preprocessing data for model training and evaluation
# ### 2. Hyperparameter optimization using a 5-fold cross-validation (CV)
# ### 3. Training and validating the final model

# In[1]:


import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import logging
from os.path import join
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, rand
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef


sys.path.append('./additional_code')
from data_preprocessing import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


# ## 1. Loading and preprocessing data for model training and evaluation

# In[2]:


def array_column_to_strings(df, column):
    df[column] = [str(list(df[column][ind])) for ind in df.index]
    return(df)

def string_column_to_array(df, column):
    df[column] = [np.array(eval(df[column][ind])) for ind in df.index]
    return(df)


# ### (a) Loading data: 
# Only keeping data points from the GO Annotation database with experimental evidence

# In[8]:


df_train = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","splits", "df_train_with_ESM1b_ts_GNN.pkl"))
df_train = df_train.loc[df_train["ESM1b"] != ""]
df_train = df_train.loc[df_train["type"] != "engqvist"]
df_train = df_train.loc[df_train["GNN rep"] != ""]
df_train.reset_index(inplace = True, drop = True)

df_test  = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","splits", "df_test_with_ESM1b_ts_GNN.pkl"))
df_test = df_test.loc[df_test["ESM1b"] != ""]
df_test = df_test.loc[df_test["type"] != "engqvist"]
df_test = df_test.loc[df_test["GNN rep"] != ""]
df_test.reset_index(inplace = True, drop = True)
print(len(df_train), len(df_test))

df_unimol = pd.read_pickle(join(CURRENT_DIR, ".." ,"data","substrate_data", "df_unimol.pkl"))

df_merged = pd.merge(df_train, df_unimol[['substrate ID', 'unimol']], on='substrate ID', how='left')
df_train = df_merged.loc[pd.notna(df_merged["unimol"]) & (df_merged["unimol"] != "")]

df_merged = pd.merge(df_test, df_unimol[['substrate ID', 'unimol']], on='substrate ID', how='left')
df_test = df_merged.loc[pd.notna(df_merged["unimol"]) & (df_merged["unimol"] != "")]
print(len(df_train), len(df_test))
df_train.reset_index(inplace = True, drop = True)
df_test.reset_index(inplace = True, drop = True)

df_train.keys()
train_length, test_length = len(df_train), len(df_test)




# In[4]:


# 打印 df_train 的所有列名
print("df_train 的列名:")
print(list(df_train.columns))

print("\n" + "="*30 + "\n") # 打印一个分隔符

# 打印 df_test 的所有列名
print("df_test 的列名:")
print(list(df_test.columns))


# ### (b) Splitting training set into 5-folds for hyperparameter optimization:
# The 5 folds are created in such a way that the same enzyme does not occure in two different folds

# In[5]:


def split_dataframe(df, frac):
    df1 = pd.DataFrame(columns = list(df.columns))
    df2 = pd.DataFrame(columns = list(df.columns))
    try:
        df.drop(columns = ["level_0"], inplace = True)
    except: 
        pass
    df.reset_index(inplace = True)
    
    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) +len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))

                while n_old != len(train_indices):
                    n_old = len(train_indices)

                    training_seqs= list(set(df["ESM1b"].loc[train_indices]))

                    train_indices = train_indices + (list(df.loc[df["ESM1b"].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))
                
            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices))

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs= list(set(df["ESM1b"].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df["ESM1b"].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))
                
        ind +=1
    return(df.loc[train_indices], df.loc[test_indices])


# In[ ]:


'''
data_train2 = df_train.copy()
data_train2 = array_column_to_strings(data_train2, column = "ESM1b")

data_train2, df_fold = split_dataframe(df = data_train2, frac=5)
indices_fold1 = list(df_fold["index"])
print(len(data_train2), len(indices_fold1))#

data_train2, df_fold = split_dataframe(df = data_train2, frac=4)
indices_fold2 = list(df_fold["index"])
print(len(data_train2), len(indices_fold2))

data_train2, df_fold = split_dataframe(df = data_train2, frac=3)
indices_fold3 = list(df_fold["index"])
print(len(data_train2), len(indices_fold3))

data_train2, df_fold = split_dataframe(df = data_train2, frac=2)
indices_fold4 = list(df_fold["index"])
indices_fold5 = list(data_train2["index"])
print(len(data_train2), len(indices_fold4))


fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]

train_indices = [[], [], [], [], []]
test_indices = [[], [], [], [], []]

for i in range(5):
    for j in range(5):
        if i != j:
            train_indices[i] = train_indices[i] + fold_indices[j]
            
    test_indices[i] = fold_indices[i]
    
np.save(join(CURRENT_DIR, ".." ,"data","splits", "CV_train_indices.npy"), np.array(train_indices, dtype=object))
np.save(join(CURRENT_DIR, ".." ,"data","splits", "CV_test_indices.npy"), np.array(test_indices, dtype=object))
'''


# In[7]:


train_indices = list(np.load(join(CURRENT_DIR, ".." ,"data","splits", "CV_train_indices.npy"),  allow_pickle=True))
test_indices = list(np.load(join(CURRENT_DIR, ".." ,"data","splits", "CV_test_indices.npy"),  allow_pickle=True))
print(len(test_indices[0]) + len(train_indices[0]))
train_indices = [ [idx for idx in row if idx < train_length] for row in train_indices]
test_indices = [ [idx for idx in row if idx < train_length] for row in test_indices]
print(len(train_indices[0]))
print(len(test_indices[0]) + len(train_indices[0]))


# ## 2. Hyperparameter optimization using a 5-fold cross-validation (CV)

# ### (a) ECFP and ESM1b:

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b"][ind]
        # ecfp = np.array(list(df["ECFP"][ind])).astype(int)
        ### unimol转换后的表征向量本身就是np.array类型，不用再单独转换
        ecfp = np.array(df["unimol"][ind]).squeeze()
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)



train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


# feature_names =  ["ECFP_" + str(i) for i in range(1024)]
### unimol的表征向量是768维的
feature_names =  ["ECFP_" + str(i) for i in range(768)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)


# #### (ii) Performing hyperparameter optimization

# In[ ]:


def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['eval_metric'] = 'logloss'
    # 可以有不同的metric
    # param['eval_metric'] = 'error'
    param['objective'] = 'binary:logistic'
    #定义一个随机种子，保证可复现
    param['seed'] = 42
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.loguniform("learning_rate", np.log(0.05), np.log(0.2)),
    "max_depth": hp.choice("max_depth", [10,12,13,14]),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 250, 400),
    "weight" : hp.uniform("weight", 0.1,0.5)}


# Performing a random grid search:

# In[11]:


# 1. 初始化 Trials 对象来记录结果
trials = Trials()

# 2. 定义评估的总次数
MAX_EVALS = 2000 # 你可以根据需要调整这个值，2000次会非常耗时

# 3. 运行超参数优化
# fmin 会自动寻找让目标函数返回值最小的超参数组合
best_hyperparams = fmin(fn=cross_validation_neg_acc_gradient_boosting, # 目标函数
                        space=space_gradient_boosting,                  # 超参数搜索空间
                        algo=rand.suggest,                              # 使用随机搜索算法
                        max_evals=MAX_EVALS,                            # 最大评估次数
                        trials=trials)                                  # 记录器

# 4. 优化完成后，打印最终的最佳结果
print("调优完成！")
print("最佳损失值 (负准确率):", trials.best_trial['result']['loss'])
print("找到的最佳超参数组合:", best_hyperparams)


# Best set of hyperparameters:

# In[15]:


###原始paramater
'''
param = {'learning_rate': 0.12771337495138718,
         'max_delta_step': 3.080851382419611,
         'max_depth': 13,
         'min_child_weight': 2.68947814956559,
         'num_rounds': 332.92969059815346,
         'reg_alpha': 1.4293630231664674,
         'reg_lambda': 0.12220981612600046,
         'weight': 0.11412319177763543}
         '''
param = best_hyperparams


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[16]:



num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'logloss'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]


loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ECFP.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ECFP.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ECFP.npy"), np.array(ROC_AUC))


# #### (iv) 3. Training and validating the final model

# Training the model and validating it on the test set:

# In[14]:


dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

#np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ECFP.npy"), bst.predict(dtest))
#np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ECFP.npy"), test_y)


# ### (b) ESM1b and GNN (pre-trained):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
        
    for ind in df.index:
        emb = df["ESM1b"][ind]
        ecfp = df["GNN rep (pretrained)"][ind]
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["GNN rep_" + str(i) for i in range(50)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 200, 400),
    "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''
# 1. 初始化 Trials 对象来记录结果
trials = Trials()

# 2. 定义评估的总次数
MAX_EVALS = 2000 # 你可以根据需要调整这个值，2000次会非常耗时

# 3. 运行超参数优化
# fmin 会自动寻找让目标函数返回值最小的超参数组合
best_hyperparams = fmin(fn=cross_validation_neg_acc_gradient_boosting, # 目标函数
                        space=space_gradient_boosting,                  # 超参数搜索空间
                        algo=rand.suggest,                              # 使用随机搜索算法
                        max_evals=MAX_EVALS,                            # 最大评估次数
                        trials=trials)                                  # 记录器

# 4. 优化完成后，打印最终的最佳结果
print("调优完成！")
print("最佳损失值 (负准确率):", trials.best_trial['result']['loss'])
print("找到的最佳超参数组合:", best_hyperparams)
'''


# Best set of hyperparameters:

# In[ ]:


param = {'learning_rate': 0.09207371208675638,
         'max_delta_step': 1.6501026095681381,
         'max_depth': 12, 
         'min_child_weight': 4.385828776339477,
         'num_rounds': 361.4040208821599,
         'reg_alpha': 2.8139614176935313,
         'reg_lambda': 1.1521733347508363,
         'weight': 0.14162338902155536}


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_GNN_pretrained.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_GNN_pretrained.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_GNN_pretrained.npy"), np.array(ROC_AUC))
'''


# #### (iv) 3. Training and validating the final model
# Training the model and validating it on the test set:

# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_GNN_pretrained.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_GNN_pretrained.npy"), test_y)
'''


# ### (c) ESM1b_ts and ECFP:

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["ECFP_" + str(i) for i in range(1024)]
feature_names = feature_names + ["ESM1b_ts_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)


# #### (ii) Performing hyperparameter optimization

# In[ ]:


def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                            "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
                            "reg_lambda": hp.uniform("reg_lambda", 0, 5),
                            "reg_alpha": hp.uniform("reg_alpha", 0, 5),
                            "max_delta_step": hp.uniform("max_delta_step", 0, 5),
                            "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
                            "num_rounds":  hp.uniform("num_rounds", 200, 400),
                            "weight" : hp.uniform("weight", 0.1,0.33)}


# Performing a random grid search:

# In[ ]:


'''
# 1. 初始化 Trials 对象来记录结果
trials = Trials()

# 2. 定义评估的总次数
MAX_EVALS = 1 # 你可以根据需要调整这个值，2000次会非常耗时

# 3. 运行超参数优化
# fmin 会自动寻找让目标函数返回值最小的超参数组合
best_hyperparams = fmin(fn=cross_validation_neg_acc_gradient_boosting, # 目标函数
                        space=space_gradient_boosting,                  # 超参数搜索空间
                        algo=rand.suggest,                              # 使用随机搜索算法
                        max_evals=MAX_EVALS,                            # 最大评估次数
                        trials=trials)                                  # 记录器

# 4. 优化完成后，打印最终的最佳结果
print("调优完成！")
print("最佳损失值 (负准确率):", trials.best_trial['result']['loss'])
print("找到的最佳超参数组合:", best_hyperparams)
'''


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.31553117247348733,
         'max_delta_step': 1.7726044219753656,
         'max_depth': 10,
         'min_child_weight': 1.3845040588450772,
         'num_rounds': 342.68325188584106,
         'reg_alpha': 0.531395259755843,
         'reg_lambda': 3.744980563764689,
         'weight': 0.26187490421514203}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_ECFP.npy"), np.array(ROC_AUC))
'''


# #### (iv) 3. Training and validating the final model
# Training the model and validating it on the test set:

# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_ECFP.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_ECFP.npy"), test_y)
'''


# ### (d) ESM1b_ts and ECFP (512-dimensional):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP_512"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)

feature_names =  ["ECFP_" + str(i) for i in range(512)]
feature_names = feature_names + ["ESM1b_ts_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                            "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
                            "reg_lambda": hp.uniform("reg_lambda", 0, 5),
                            "reg_alpha": hp.uniform("reg_alpha", 0, 5),
                            "max_delta_step": hp.uniform("max_delta_step", 0, 5),
                            "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
                            "num_rounds":  hp.uniform("num_rounds", 200, 400),
                            "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''trials = Trials()

for i in range(1,2000):
    best = fmin(fn = cross_validation_neg_acc_gradient_boosting, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    logging.info(i)
    logging.info(trials.best_trial["result"]["loss"])
    logging.info(trials.argmin)''';


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.20031456821679422,
         'max_delta_step': 3.723458003552047,
         'max_depth': 10,
         'min_child_weight': 2.0109208762032678,
         'num_rounds': 347.78525681188614,
         'reg_alpha': 2.213525607682663,
         'reg_lambda': 4.5822546393906025,
         'weight': 0.16604653557737126}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_ECFP_512.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_ECFP_512.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_ECFP_512.npy"), np.array(ROC_AUC))
'''


# #### (iv) 3. Training and validating the final model
# Training the model and validating it on the test set:

# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_ECFP_512.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_ECFP_512.npy"), test_y)
'''


# ### (e) ESM1b_ts and ECFP (2048-dimensional):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP_2048"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)

feature_names =  ["ECFP_" + str(i) for i in range(2048)]
feature_names = feature_names + ["ESM1b_ts_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                            "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
                            "reg_lambda": hp.uniform("reg_lambda", 0, 5),
                            "reg_alpha": hp.uniform("reg_alpha", 0, 5),
                            "max_delta_step": hp.uniform("max_delta_step", 0, 5),
                            "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
                            "num_rounds":  hp.uniform("num_rounds", 200, 400),
                            "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''trials = Trials()

for i in range(1,2000):
    best = fmin(fn = cross_validation_neg_acc_gradient_boosting, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    logging.info(i)
    logging.info(trials.best_trial["result"]["loss"])
    logging.info(trials.argmin)''';


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.38572930012069273,
         'max_delta_step': 2.3691090709866254,
         'max_depth': 13,
         'min_child_weight': 0.11946441222742316,
         'num_rounds': 185.5041665008057,
         'reg_alpha': 0.4492202436042625,
         'reg_lambda': 0.8927451484733545,
         'weight': 0.10881477775175043}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_ECFP_2048.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_ECFP_2048.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_ECFP_2048.npy"), np.array(ROC_AUC))
'''


# #### (iv) 3. Training and validating the final model
# Training the model and validating it on the test set:

# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_ECFP_2048.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_ECFP_2048.npy"), test_y)
'''


# ### (f) ESM1b_ts and GNN (pretrained):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = df["GNN rep (pretrained)"][ind]
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["GNN rep_" + str(i) for i in range(50)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 200, 400),
    "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''trials = Trials()

for i in range(1,2000):
    best = fmin(fn = cross_validation_neg_acc_gradient_boosting, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    logging.info(i)
    logging.info(trials.best_trial["result"]["loss"])
    logging.info(trials.argmin)''';


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.19789627044374644,
 'max_delta_step': 3.815106738298364,
 'max_depth': 12,
 'min_child_weight': 0.9568708633806051,
 'num_rounds': 358.42154962618235,
 'reg_alpha': 0.3726209284173021,
 'reg_lambda': 4.442065146895246,
 'weight': 0.11281944917093198}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_GNN_pretrained.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_GNN_pretrained.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_GNN_pretrained.npy"), np.array(ROC_AUC))
'''


# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_GNN_pretrained.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_GNN_pretrained.npy"), test_y)
'''


# ### (g) ESM1b_ts (mean representaion) and GNN (pretrained):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts_mean"][ind]
        ecfp = df["GNN rep (pretrained)"][ind]
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["GNN rep_" + str(i) for i in range(50)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 200, 400),
    "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''trials = Trials()

for i in range(1,2000):
    best = fmin(fn = cross_validation_neg_acc_gradient_boosting, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    logging.info(i)
    logging.info(trials.best_trial["result"]["loss"])
    logging.info(trials.argmin)''';


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.1156069031228949,
         'max_delta_step': 2.6918966267028415,
         'max_depth': 12,
         'min_child_weight': 1.9758135134127655,
         'num_rounds': 127.079967978755,
         'reg_alpha': 2.3233749696436714,
         'reg_lambda': 2.163811098458429,
         'weight': 0.28341863683389795}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_mean_GNN_pretrained.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_mean_GNN_pretrained.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_mean_GNN_pretrained.npy"), np.array(ROC_AUC))
'''


# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_mean_GNN_pretrained.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_mean_GNN_pretrained.npy"), test_y)
'''


# ### (h) ESM1b_ts and GNN (not pre-trained):

# #### (i) Creating numpy arrays with input vectors and output variables

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = df["GNN rep"][ind]
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["GNN rep_" + str(i) for i in range(50)]
feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# #### (ii) Performing hyperparameter optimization

# In[ ]:


'''
def cross_validation_neg_acc_gradient_boosting(param):
    num_round = param["num_rounds"]
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])
    
    del param["num_rounds"]
    del param["weight"]
    
    loss = []
    for i in range(5):
        train_index, test_index  = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                         label = np.array(train_y[train_index]))
        dvalid = xgb.DMatrix(np.array(train_X[test_index]))
        bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
        y_valid_pred = np.round(bst.predict(dvalid))
        validation_y = train_y[test_index]
    
        false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
        false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
        logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
        loss.append(2*(false_negative**2) + false_positive**1.3)
    return(np.mean(loss))

#Defining search space for hyperparameter optimization
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", [9,10,11,12,13]),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 200, 400),
    "weight" : hp.uniform("weight", 0.1,0.33)}
'''


# Performing a random grid search:

# In[ ]:


'''trials = Trials()

for i in range(1,2000):
    best = fmin(fn = cross_validation_neg_acc_gradient_boosting, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    logging.info(i)
    logging.info(trials.best_trial["result"]["loss"])
    logging.info(trials.argmin)''';


# Best set of hyperparameters:

# In[ ]:


'''
param = {'learning_rate': 0.18444025726334898,
         'max_delta_step': 3.2748796106084117,
         'max_depth': 13,
         'min_child_weight': 3.1946753845027738,
         'num_rounds': 314.1036429221291,
         'reg_alpha': 0.48821021807600673,
         'reg_lambda': 2.6236829011598073,
         'weight': 0.1264521266931227}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in df_train["Binding"]])

del param["num_rounds"]
del param["weight"]
'''


# #### (iii) Repeating 5-fold CV for best set of hyperparameters

# In[ ]:


'''
loss = []
accuracy = []
ROC_AUC = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(np.array(train_X[train_index]), weight = weights[train_index],
                     label = np.array(train_y[train_index]))
    dvalid = xgb.DMatrix(np.array(train_X[test_index]))
    bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
    y_valid_pred = np.round(bst.predict(dvalid))
    validation_y = train_y[test_index]

    #calculate loss:
    false_positive = 100*(1-np.mean(np.array(validation_y)[y_valid_pred == 1]))
    false_negative = 100*(np.mean(np.array(validation_y)[y_valid_pred == 0]))
    logging.info("False positive rate: " + str(false_positive)+ "; False negative rate: " + str(false_negative))
    loss.append(2*(false_negative**2) + false_positive**1.3)
    #calculate accuracy:
    accuracy.append(np.mean(y_valid_pred == np.array(validation_y)))
    #calculate ROC-AUC score:
    ROC_AUC.append(roc_auc_score(np.array(validation_y), bst.predict(dvalid)))
    
print("Loss values: %s" %loss) 
print("Accuracies: %s" %accuracy)
print("ROC-AUC scores: %s" %ROC_AUC)

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "acc_CV_xgboost_ESM1b_ts_GNN.npy"), np.array(accuracy))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "loss_CV_xgboost_ESM1b_ts_GNN.npy"), np.array(loss))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "ROC_AUC_CV_xgboost_ESM1b_ts_GNN.npy"), np.array(ROC_AUC))
'''


# #### (iv) 3. Training and validating the final model
# Training the model and validating it on the test set:

# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))

np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_pred_xgboost_ESM1b_ts_GNN.npy"), bst.predict(dtest))
np.save(join(CURRENT_DIR, ".." ,"data", "training_results", "y_test_true_xgboost_ESM1b_ts_GNN.npy"), test_y)
'''


# ## 3.Training the best model with training and test set (production mode):

# In[ ]:


'''
def create_input_and_output_data(df):
    X = ();
    y = ();
    
    for ind in df.index:
        emb = df["ESM1b_ts"][ind]
        ecfp = np.array(list(df["ECFP"][ind])).astype(int)
                
        X = X +(np.concatenate([ecfp, emb]), );
        y = y + (df["Binding"][ind], );

    return(X,y)

train_X, train_y =  create_input_and_output_data(df = df_train)
test_X, test_y =  create_input_and_output_data(df = df_test)


feature_names =  ["ECFP_" + str(i) for i in range(1024)]
feature_names = feature_names + ["ESM1b_ts_" + str(i) for i in range(1280)]

train_X = np.array(train_X)
test_X  = np.array(test_X)

train_y = np.array(train_y)
test_y  = np.array(test_y)
'''


# In[ ]:


'''
train_X = np.concatenate([train_X, test_X])
train_y = np.concatenate([train_y, test_y])
'''


# In[ ]:


'''
param = {'learning_rate': 0.31553117247348733,
         'max_delta_step': 1.7726044219753656,
         'max_depth': 10,
         'min_child_weight': 1.3845040588450772,
         'num_rounds': 342.68325188584106,
         'reg_alpha': 0.531395259755843,
         'reg_lambda': 3.744980563764689,
         'weight': 0.26187490421514203}

num_round = param["num_rounds"]
param["tree_method"] = "gpu_hist"
param["sampling_method"] = "gradient_based"
param['objective'] = 'binary:logistic'
weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in train_y])

del param["num_rounds"]
del param["weight"]
'''


# In[ ]:


'''
dtrain = xgb.DMatrix(np.array(train_X), weight = weights, label = np.array(train_y),
                feature_names= feature_names)
dtest = xgb.DMatrix(np.array(test_X), label = np.array(test_y),
                    feature_names= feature_names)

bst = xgb.train(param,  dtrain, int(num_round), verbose_eval=1)
y_test_pred = np.round(bst.predict(dtest))
acc_test = np.mean(y_test_pred == np.array(test_y))
roc_auc = roc_auc_score(np.array(test_y), bst.predict(dtest))
mcc = matthews_corrcoef(np.array(test_y), y_test_pred)

print("Accuracy on test set: %s, ROC-AUC score for test set: %s, MCC: %s"  % (acc_test, roc_auc, mcc))
'''


# In[ ]:


'''
y_train_pred = np.round(bst.predict(dtrain))
acc_train = np.mean(y_train_pred == np.array(train_y))
roc_auc = roc_auc_score(np.array(train_y), bst.predict(dtrain))
mcc = matthews_corrcoef(np.array(train_y), y_train_pred)

print("Accuracy on train set: %s, ROC-AUC score for train set: %s, MCC: %s"  % (acc_train, roc_auc, mcc))
'''


# In[ ]:


'''
pickle.dump(bst, open(join(CURRENT_DIR, ".." ,"data", "model_weights",
                           "xgboost_model_production_mode.dat"), "wb"))
'''

