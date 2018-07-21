#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print metrics callback to improve the output

@author: dschaefer
"""
import warnings
import keras

class PrintMetricsCallback(keras.callbacks.Callback):
            
    def __init__(self):
        super(PrintMetricsCallback, self).__init__()
        
    def on_epoch_end(self, epoch, logs={}):
        print('\n{{"metric": "Epoch", "value": {}}}'.format(epoch))
        
        current = logs.get('val_loss')
        if current is None:
            warnings.warn(
                'Metric not available', RuntimeWarning
            )
            return
        val_loss = current
        print('\n{{"metric": "Validation Loss", "value": {}}}'.format(val_loss))