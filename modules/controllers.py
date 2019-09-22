# -*- coding: utf-8 -*-
"""

Controllers for diffferent models

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import models
import optimizers

#from scipy.optimize import minimize

dtype = theano.config.floatX


class ControlNeuralHawkesAdaptiveBaseCTSM_time(object):
    #
    def __init__(self, settings):
        print "building controller ... "
        '''
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        #self.seq_time_to_end = tensor.matrix(
        #    dtype=dtype, name='seq_time_to_end'
        #)
        self.seq_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_time_to_current'
        )
        self.seq_type_event = tensor.imatrix(
            name='seq_type_event'
        )
        #self.seq_time_rep = tensor.tensor3(
        #    dtype=dtype, name='seq_time_rep'
        #)
        self.seq_time_values = tensor.matrix(
            dtype=dtype, name='seq_time_values'
        )
        #
        self.time_since_start_to_end = tensor.vector(
            dtype=dtype, name='time_since_start_to_end'
        )
        self.num_sims_start_to_end = tensor.vector(
            dtype=dtype, name='num_sims_start_to_end'
        )
        self.seq_mask = tensor.matrix(
            dtype=dtype, name='seq_mask'
        )
        self.seq_sims_time_to_current = tensor.matrix(
            dtype=dtype, name='seq_sims_time_to_current'
        )
        self.seq_sims_index_in_hidden = tensor.imatrix(
            name='seq_sims_index_in_hidden'
        )
        self.seq_sims_mask = tensor.matrix(
            dtype=dtype, name='seq_sims_mask'
        )
        self.time_diffs = tensor.vector(
            dtype=dtype, name='time_diffs'
        )
        #
        #

        self.hawkes_ctsm = models.NeuralHawkesCTLSTM(
            settings
        )
        list_constrain = [0]



        print "train with prediction ... "
        #TODO: need to add switch for less memory
        #or faster speed
        #self.hawkes_ctsm.compute_prediction_loss(
        self.hawkes_ctsm.compute_prediction_loss_lessmem(
            self.seq_type_event,
            self.seq_time_values,
            self.seq_mask,
            self.time_diffs
        )


        self.adam_optimizer = optimizers.Adam(
            adam_params=None
        )
        #
        if 'learn_rate' in settings:
            print "learn rate is set to : ", settings['learn_rate']
            self.adam_optimizer.set_learn_rate(
                settings['learn_rate']
            )
        #
        self.adam_optimizer.compute_updates(
            self.hawkes_ctsm.params, self.hawkes_ctsm.grad_params,
            list_constrain = list_constrain
        )
        # in this version, no hard constraints on parameters
        #

        print "optimize prediction ... "
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                self.seq_type_event,
                self.seq_time_values,
                self.seq_mask,
                self.time_diffs
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_type_predict,
                self.hawkes_ctsm.num_of_errors,
                self.hawkes_ctsm.square_errors,
                self.hawkes_ctsm.num_of_events
                #self.hawkes_ctsm.abs_grad_params
            ],
            updates = self.adam_optimizer.updates,
            on_unused_input='ignore'
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                self.seq_type_event,
                self.seq_time_values,
                self.seq_mask,
                self.time_diffs
            ],
            outputs = [
                self.hawkes_ctsm.log_likelihood_type_predict,
                self.hawkes_ctsm.num_of_errors,
                self.hawkes_ctsm.square_errors,
                self.hawkes_ctsm.num_of_events,
                self.hawkes_ctsm.time_pred,
                self.hawkes_ctsm.type_pred
                #self.hawkes_ctsm.abs_grad_params
                #
            ],
            on_unused_input='ignore'
        )
        #
        #
        self.get_model = self.hawkes_ctsm.get_model
        self.save_model = self.hawkes_ctsm.save_model
        #
    #
#
#
