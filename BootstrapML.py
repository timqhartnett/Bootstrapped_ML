#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:53:07 2021

First attempt at objected oriented ML bootstrapping 

@author: timothy
"""
import pandas as pd
import numpy as np

class data_analysis:
    '''
    class to handle data analysis and visualization prior to model creation
    in general it is good practice to visualize your response distribution as well as the feature distribution
    consider expanding to include elimination of correlated features along with PCA and other analysis techniques
    '''
    def __init__(self, data,response): #Will accept pandas dataframe
        self.data = data
        self.response = data.loc[:,response] #choose which column is the response and which are the features
        self.features = data.drop(columns=response)
    def correlation(self): #use for feature corrleation
        correlation_matrix = self.features.corr()
        return(correlation_matrix)
    def feature_distribution(self):
        histograms = []
        for feature in self.features:
            histograms.append(self.features.hist(column=feature))
        return histograms #setup should return a list containing each histogram, depending on your python setup it may or may not also print the images directly
    def response_distribution(self):
        histogram = self.response.hist()
        return histogram
    def identify_outliers(self): #outliers are identified by their z-score and will appear as a dataframe
        temp = pd.DataFrame()
        for column in list(self.features.columns):
            col_zscore = column + '_zscore'
            temp[col_zscore] = (self.features[column]-self.features[column].mean())/self.features[column].std(ddof=0)
        outliers = temp.loc[(temp>3).any(1)]
        return outliers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class utilities:
    '''
    classs to handle ML utilities including train/test split, bootstrapped train_test split, bootstarpped prediction
    and data analysis
    for bootstrapped problems the test split is initialized at 30% test, 70% train. can adjust split but not
    eliminate split so far
    '''
    def __init__(self, data,response): #Will accept pandas dataframe
        self.data = data
        self.response = data.loc[:,response] #choose which column is the response and which are the features
        self.features = data.drop(columns=response)
    def split(self,size_test = 0.3):#test size initialized to 30% but can be adjusted by user input
        features_train,features_test,response_train,response_test = train_test_split(self.features,self.response,test_size=size_test)
        output = [features_train,features_test,response_train,response_test]
        return output
    def split_bagging(self,number_bags,size_test = 0.3):
        features_train,features_test,response_train,response_test = train_test_split(self.features,self.response,test_size=size_test)
        bagged_features = []
        bagged_response = []
        out_of_bag_features = []
        out_of_bag_response = []
        for i in range(number_bags): #split into 6 dataframes train_bagged(features and responses)[2], train out of bag(features and response)[2], test(features and response)
            bagged_sample_index=np.random.choice(features_train.index,size = features_train.shape[0],replace=True)
            print(bagged_sample_index)
            print('features')
            print(features_train.index)
            print('response')
            print(response_train)
            bagged_features.append(features_train.loc[bagged_sample_index,:])
            bagged_response.append(response_train.loc[bagged_sample_index])
            out_of_bag_features.append(features_train.drop(np.unique(bagged_sample_index),axis=0))
            out_of_bag_response.append(response_train.drop(np.unique(bagged_sample_index)))
            bagged_features,bagged_response,out_of_bag_features,out_of_bag_response,features_test,response_test
            output = [bagged_features,bagged_response,out_of_bag_features,out_of_bag_response,features_test,response_test]
        return output
    def boot_predict(self,bagged_models,test_features,test_response): #use list of models as
        predictions = pd.DataFrame()
        for i in range(len(bagged_models)):
            predictions[i] = bagged_models[i].predict(test_features)
        labels = test_response.astype('category').cat.categories.tolist()
        for label in labels:
            predictions[label]  = predictions.apply(lambda row: sum(row[0:len(bagged_models)] == label),axis=1)
        output = predictions.iloc[:,len(bagged_models):]
        output['final'] = output.idxmax(axis=1)
        return output #returns only sum of all predictions from models and final decision
    def clf_analysis(self,prediction,actual):
        confusion = confusion_matrix(actual,prediction) #returns np.array
        labels = actual.astype('category').cat.categories.tolist() #associated classes in problem
        confMat = pd.DataFrame(confusion,index=labels,columns=labels) #adds classes to Confusion matrix
        accuracy = accuracy_score(actual,prediction)
        recall = recall_score(actual,prediction,average=None)
        precision = precision_score(actual,prediction,average=None)
        f1 =  2*precision*recall/(precision+recall)
        return confMat,accuracy,recall,precision,f1
    def oob_score(self,oob_error):
        oob_accuracy = 0
        for i in range(len(oob_error)):
            oob_accuracy = oob_accuracy+oob_error[i][1]
        oob_accuracy = oob_accuracy/len(oob_error)
        return oob_accuracy
    def virtual_predict(self,bagged_models,virtual_set,classes):
        predictions = pd.DataFrame()
        for i in range(len(bagged_models)):
            predictions[i] = bagged_models[i].predict(virtual_set)
        labels = list(classes)
        for label in labels:
            predictions[label]  = predictions.apply(lambda row: sum(row[0:len(bagged_models)] == label),axis=1)
        output = predictions.iloc[:,len(bagged_models):]
        output['final'] = output.idxmax(axis=1)
        return output
        
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class bootstrapped_SVC(utilities):
    '''
    trains ensemble of Bootstrapped SVCs
    bagged samples will have same length as input meaning some data points will be seen multiple times
    used for active learning problems where variance is desired in the model
    for classification problems probability is the prediction output bassed on the classification from each model
    Potential expansion to other kernels: linear, polynomial, and sigmoid
    '''
    def __init__(self,data,response):
        super().__init__(data,response)#subclass to utilities should allow for split bagging 
    def train_radial(self,bagging_output):
        param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']} #same grid used in online example can change
        models = []
        oob_error = []
        for i in range(len(bagging_output[0])):
            grid = GridSearchCV(SVC(),param_grid,refit=True)
            models.append(grid.fit(bagging_output[0][i],bagging_output[1][i]))
            oob_prediction = grid.predict(bagging_output[2][i])
            oob_error.append(self.clf_analysis(oob_prediction,bagging_output[3][i]))
        test_prediction = self.boot_predict(models,bagging_output[4],bagging_output[5])
        confMat,accuracy,recall,precision,f1 = self.clf_analysis(test_prediction['final'],bagging_output[5])
        return models,oob_error,test_prediction,confMat,accuracy,recall,precision,f1
    
from sklearn.naive_bayes import GaussianNB

class naive_bayes(utilities):
    '''
    trains naive bayes classifier
    implemented algorightms Gaussian bayes classifier
    possible exansion to Multinomial? (mostly used for NLP)
    '''   
    def __init__(self,data,response):
        super().__init__(data,response)#for Naive Bayes there is no need for bagging
    def gaussian_bayes(self,split_output):
        gnb = GaussianNB()
        gnb.fit(split_output[0],split_output[2])
        predictions = gnb.predict(split_output[1])
        confMat,accuracy,recall,precision,f1 = self.clf_analysis(predictions,split_output[3])
        return gnb,confMat,accuracy,recall,precision,f1

from sklearn.neighbors import KNeighborsClassifier

class bootstrapped_knn(utilities):
    '''
    trains ensemble of bootstrapped k-nearest neighbor models, see above bootstrapped svc
    knn models are optimized for k which is the number of neighboring datapoints to consider
    '''
    def __init__(self,data,response):
        super().__init__(data,response)#subclass to utilities should allow for split bagging 
    def train_knn(self, bagging_output):
        k_range = range(1,30)
        models = []
        oob_accuracy = []
        for i in range(len(bagging_output[0])):
            scores = []
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(bagging_output[0][i],bagging_output[1][i])
                y_pred = knn.predict(bagging_output[2][i])
                scores.append(accuracy_score(bagging_output[3][i], y_pred))
            best_k = scores.index(max(scores))+1
            oob_accuracy.append(max(scores))
            knn_best = KNeighborsClassifier(n_neighbors = best_k)
            model = knn_best.fit(bagging_output[0][i],bagging_output[1][i])
            models.append(model)
        test_prediction = self.boot_predict(models,bagging_output[4],bagging_output[5])
        confMat,accuracy,recall,precision,f1 = self.clf_analysis(test_prediction['final'],bagging_output[5])
        return models,oob_accuracy,test_prediction,confMat,accuracy,recall,precision,f1

from sklearn.ensemble import RandomForestClassifier
    
class random_forest(utilities):
    '''
    trains random forest classifier
    '''
    def __init__(self,data,response):
        super().__init__(data,response)
    def train_rf(self, split_output):
        paramgrid = {
            'max_features':list(range(1,6)),
            'n_estimators':list(range(100,1000,50)),
            }
        grid = GridSearchCV(RandomForestClassifier(), paramgrid, refit = False)
        grid_models = grid.fit(split_output[0],split_output[2])
        model = RandomForestClassifier(max_features=grid_models.best_params_['max_features'],
                                       n_estimators=grid_models.best_params_['n_estimators'],
                                       oob_score=True)
        model.fit(split_output[0],split_output[2])
        predictions = model.predict(split_output[1])
        confMat,accuracy,recall,precision,f1 = self.clf_analysis(predictions,split_output[3])
        return model,confMat,accuracy,recall,precision,f1
    
