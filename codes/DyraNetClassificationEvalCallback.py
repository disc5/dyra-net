#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates the (classification) accuracy of the model on validation data

@author: dschaefer
"""
import keras
import numpy as np

class DyraNetClassificationEvalCallback(keras.callbacks.Callback):
            
    def __init__(self, X_inst_features, y_class_labels):
        self.X = X_inst_features
        self.y = y_class_labels
        self.num_classes = np.unique(y_class_labels).shape[0]
        
    def on_epoch_end(self, epoch, logs={}):
        accuracy = self.get_accuracy()
        print('\n{{"metric": "Validation (Classification) Accuracy", "value": {}}}'.format(accuracy))

    def get_accuracy(self):
        '''
            Calculates the classication accuracy
        '''
        n_examples = len(self.y)
        accuracy = 0
        for idx in range(n_examples):
            ct_inst = self.X[idx]
            true_cl = self.y[idx]+1
            pred_cl, scores = self.predict_class(ct_inst, self.num_classes)
            if true_cl == pred_cl:
                accuracy = accuracy + 1
        accuracy = accuracy / n_examples
        return accuracy
    
    def predict_class(self, instance, num_classes):
        '''
            Uses a trained ranking model classify a given instance vector.
            The model takes inputs
                X: instance
                Y: one-of-K encoded class vector
            
            returns a class
        '''
        scores = list()
        all_classes = np.eye(num_classes)
        inst = np.reshape(instance,(1,28,28,1))
        for i in range(len(all_classes)):
            cl = all_classes[i]
            cl = np.reshape(cl,(1,num_classes))
            scores.append(self.model.predict([inst, cl]))
        
        sas = np.squeeze(np.asarray(scores))
        ordering = sas.argsort()[::-1][:num_classes] + 1
        return ordering[0], scores
    