#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:22:25 2022

@author: monty
"""

import numpy as np


class Gradient_Descent:
    def __init__(self, data, labels, hidden_layers, activation_funcs, iterations, alpha):
        self.m=len(labels[0])
        self.n=len(data[0])
        self.X_train=np.array(data).T
        self.Y_train=np.array(labels)
        self.layers=hidden_layers
        self.iterations=iterations
        self.alpha=alpha
        self.activation_funcs=activation_funcs
        self.funcs={"ReLU":[self.ReLU, self.deriv_ReLU],"sigmoid":[self.sigmoid, self.deriv_sigmoid],"softmax":[self.softmax]}
        
        
        self.W1=np.random.rand(self.layers[1],self.layers[0])-0.5
        self.b1=np.random.rand(self.layers[1],1)-0.5
        self.W2=np.random.rand(self.layers[2],self.layers[1])-0.5
        self.b2=np.random.rand(self.layers[2],1)-0.5
    
    def forward_prop(self, X):
        Z1=self.W1.dot(X)+self.b1
        A1=self.funcs[self.activation_funcs[0]][0](Z1)
        Z2=self.W2.dot(A1)+self.b2
        A2=self.funcs[self.activation_funcs[1]][0](Z2)        
        return Z1, A1, Z2, A2
    
    def back_prop(self, Z1, A1, Z2, A2):
        dZ2=2*(A2-self.Y_train)
        dW2=1/self.m*(dZ2.dot(A1.T))
        db2=1/self.m*np.sum(dZ2,1)
        dZ1=(self.W2.T.dot(dZ2))*self.funcs[self.activation_funcs[0]][1](Z1)
        dW1=1/self.m*(dZ1.dot(self.X_train.T))
        db1=1/self.m*np.sum(dZ1,1)
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * np.reshape(db1, (self.layers[1],1))
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * np.reshape(db2, (self.layers[2],1))
    
    def ReLU(self, Z):
        return np.maximum(Z,0)
    
    def deriv_ReLU(self, Z):
        return Z > 0
    
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def deriv_sigmoid(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    
    def softmax(self, Z):
        exp = np.exp(Z - np.max(Z)) 
        return exp / exp.sum(axis=0)
    
    def get_predictions_round(self, Z):
        return np.round(Z)
    
    def get_predictions_argmax(self, Z):
        return np.argmax(Z, 0)
    
    def get_accuracy(self, predictions):
        return np.sum(predictions == self.get_predictions_argmax(self.Y_train))/self.m

    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        return self.get_predictions_argmax(A2)
    
    def test_prediction_index(self, index):
        #print("Data: ", self.X_train.T[index])
        print("Prediction: ", self.make_predictions(self.X_train)[index])
        print("Label: ", self.get_predictions_argmax(self.Y_train.T[index]))
        
    def test_prediction(self, datum):
        print("Data: ", datum)
        print("Prediction: ", self.make_predictions(np.reshape(np.array(datum), (self.layers[0],1)))[0][0])
    
    def gdescend(self, monitor=False):
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forward_prop(self.X_train)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2)
            self.update_params(dW1, db1, dW2, db2)
            if i%50 == 0 and monitor:
                print("Iteration: ",i)
                predictions=self.get_predictions_argmax(A2)
                print("Accuracy: ", self.get_accuracy(predictions))

#____MAIN____#

if __name__ == "__main__":
    print("this is a class silly")
    

