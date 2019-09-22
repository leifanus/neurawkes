# -*- coding: utf-8 -*-
"""

Here are the models
continuous-time sequential model (CTSM)

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

dtype=theano.config.floatX


#
#
class NeuralHawkesCTLSTM(object):
    '''
    This model uses:
    Adative base rate, interaction and decay
    Continuous-time LSTM
    Scale parameter s_k for softrelu curvature adjustment
    Reduced version -- delta param is D * D, not D * D * K
    '''
    #
    def __init__(self, settings):
        self.size_batch = settings['size_batch']
        self.coef_l2 = settings['coef_l2']
        self.time_pred = None
        self.type_pred = None
        #
        #
        print "initializing Neural Hawkes with Continuous-time LSTM ... "
        if settings['path_pre_train'] == None:
            self.dim_process = settings['dim_process']
            self.dim_time = settings['dim_time']
            # the dimension of time representations
            # this is useless in cont-time lstm
            self.dim_model = settings['dim_model']
            # initialize variables
            self.scale = theano.shared(
                numpy.ones(
                    (self.dim_process,), dtype=dtype
                ), name='scale'
            )
            #
            # the 0-th axis -- self.dim_model
            # is for dot product with hidden units
            # dot(h, W_delta) --> delta of size:
            # dim_model * dim_process
            #
            self.W_alpha = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_process
                ), name='W_alpha'
            )
            # + 1 cuz there is a special BOS event
            self.Emb_event = theano.shared(
                utils.sample_weights(
                    self.dim_process+numpy.int32(1), self.dim_model
                ), name='Emb_event'
            )
            #
            self.W_recur = theano.shared(
                utils.sample_weights(
                    2*self.dim_model, 7*self.dim_model
                ), name='W_recur'
            )
            '''
            2 input :
            event rep, hidden state
            7 outputs :
            4 regular LSTM gates
            2 -- input_bar and forget_bar gate
            1 -- cell memory decay gate
            '''
            self.b_recur = theano.shared(
                numpy.zeros(
                    (7*self.dim_model,), dtype=dtype
                ), name='b_recur'
            )
            #
        else:
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #with open(settings['path_pre_train'], 'rb') as f:
            #    model_pre_train = pickle.load(f)
            self.dim_process = model_pre_train['dim_process']
            self.dim_model = model_pre_train['dim_model']
            self.dim_time = model_pre_train['dim_time']
            #
            self.scale = theano.shared(
                model_pre_train['scale'], name='scale'
            )
            #
            self.W_alpha = theano.shared(
                model_pre_train['W_alpha'], name='W_alpha'
            )
            self.Emb_event = theano.shared(
                model_pre_train['Emb_event'], name='Emb_event'
            )
            #
            self.W_recur = theano.shared(
                model_pre_train['W_recur'], name='W_recur'
            )
            self.b_recur = theano.shared(
                model_pre_train['b_recur'], name='b_recur'
            )
        #
        self.h_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='h_0'
        )
        self.c_0 = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0'
        )
        self.c_0_target = theano.shared(
            numpy.zeros(
                (self.dim_model, ), dtype=dtype
            ), name='c_0_target'
        )
        self.expand = theano.shared(
            numpy.ones(
                (self.size_batch, ), dtype=dtype
            ), name='expand'
        )
        # alpha & delta, i-row j-col is the effect of j to i
        #
        self.params = [
            #self.mu, #self.delta,
            self.scale, # scale parameter
            self.W_alpha,
            self.Emb_event,
            self.W_recur, self.b_recur
            #self.h_0, self.c_0
        ]
        #
        self.grad_params = None
        self.cost_to_optimize = None
        #
        #
        self.log_likelihood_seq = None
        self.log_likelihood_type = None
        self.log_likelihood_time = None
        #
        self.norm_l2 = numpy.float32(0.0)
        for param in self.params:
            self.norm_l2 += tensor.sum( param ** 2 )
        self.term_reg = self.coef_l2 * self.norm_l2
        #
        #
    #

    def soft_relu(self, x):
        # x is a symbolic tensor
        # tensor[(x == 0).nonzeros()]
        #v_max = numpy.float32(1e9)
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        #a = tensor.switch(z>=v_max, v_max, z)
        #y[(x>=100.0).nonzeros()] = x[(x>=100.0).nonzeros()]
        #np.finfo(np.float32).max
        return z
    #
    #
    def soft_relu_scale(self, x):
        # x is symbolic tensor
        # last dim is dim_process
        # this is important !
        x /= self.scale
        y = tensor.log(numpy.float32(1.0)+tensor.exp(x) )
        z = tensor.switch(x>=100.0, x, y)
        z *= self.scale
        return z
    #
    #
    def rnn_unit(
        self,
        emb_event_im1, time_interval_im1,
        hidden_t_im1, cell_t_im1, cell_im1_target
    ):
        '''
        This LSTM unit is working in continuous-time
        What a regular LSTM does :
        Take h_{i-1}, and update to h_i
        What a CT-LSTM does :
        Take h(t_{i-1}), which decays to t_{i-1}
        Use it and upate to h_i
        h_i is then used to compute Hawkes params
        #
        input:
        emb_event_imt = x_{i-1}
        time_interval_i = t_i - t_{i-1}
        h(t_{i-1}) right before THIS update
        c(t_{i-1}) right before THIS update
        c_{i-1}_target before THIS update
        output: ( # stands for not output it )
        h(t_i) right before NEXT update at t_i
        c(t_i) right before NEXT update at t_i
        c_i_target over ( t_{i-1}, t_i ]
        #h_i = h( t_{i-1} <-- t ) right after THIS update
        c_i = c( t_{i-1} <-- t ) right after THIS update
        decay_rate over ( t_{i-1}, t_i ]
        gate_output over ( t_{i-1}, t_i ]
        '''
        #TODO: update LSTM state at t_{i-1}
        pre_transform = tensor.concatenate(
            [emb_event_im1, hidden_t_im1],
            axis = 1
        )
        post_transform = tensor.dot(
            pre_transform, self.W_recur
        ) + self.b_recur
        # 4 regular LSTM gates
        gate_input = tensor.nnet.sigmoid(
            post_transform[:, :self.dim_model]
        )
        gate_forget = tensor.nnet.sigmoid(
            post_transform[:, self.dim_model:2*self.dim_model]
        )
        gate_output = tensor.nnet.sigmoid(
            post_transform[
                :, 2*self.dim_model:3*self.dim_model
            ]
        )
        gate_pre_c = tensor.tanh(
            post_transform[
                :, 3*self.dim_model:4*self.dim_model
            ]
        )
        # 2 -- input_bar and forget_bar gates
        gate_input_target = tensor.nnet.sigmoid(
            post_transform[
                :, 4*self.dim_model:5*self.dim_model
            ]
        )
        gate_forget_target = tensor.nnet.sigmoid(
            post_transform[
                :, 5*self.dim_model:6*self.dim_model
            ]
        )
        # cell memory decay
        decay_cell = self.soft_relu(
            post_transform[
                :, 6*self.dim_model:
            ]
        )
        # size : size_batch * dim_model
        #TODO: decay cell memory
        cell_i = gate_forget * cell_t_im1 + gate_input * gate_pre_c
        cell_i_target = gate_forget_target * cell_im1_target + gate_input_target * gate_pre_c
        #
        cell_t_i = cell_i_target + (
            cell_i - cell_i_target
        ) * tensor.exp(
            -decay_cell * time_interval_im1[:, None]
        )
        hidden_t_i = gate_output * tensor.tanh(
            cell_t_i
        )
        #TODO: get the hidden state right after this update, which is used to compute Hawkes params
        hidden_i = gate_output * tensor.tanh(
            cell_i
        )
        return hidden_t_i, cell_t_i, cell_i_target, cell_i, decay_cell, gate_output
        #return hidden_t_i, cell_t_i, cell_i_target, hidden_i, cell_i, decay_cell, gate_output
        #
    #
    #
    def compute_loss(
        self,
        seq_time_to_current,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        time_since_start_to_end,
        num_sims_start_to_end,
        seq_mask,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute log likelihood
        seq_time_to_current : T * size_batch -- t_i - t_i-1
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_values : (T+1) * size_batch -- t_i - t_i-1 starting as 0.0 at BOS event
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        seq_mask : T * size_batch -- 1/0
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        Warning: There is overlap between seq_time_values and seq_time_to_current, so in this function, we happen not to use both. So we need to put on_unused_input='warn' in theano.function to avoid error message.
        '''
        print "computing loss function of Neural Hawkes model with continuous-time LSTM ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        # T * size_batch * dim_model
        '''
        No need to pass time values through thresholds
        Use time_values directly
        '''
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_time_values start with
        # a special BOS event -- ( K, 0.0 )
        # to initialize the h, c and \bar{c}
        '''
        seq_cell_target, seq_cell : cell right AFTER THIS occurrence, including BOS
        seq_decay_cell, seq_gate_output : decay and gates AFTER THIS and BEFORE NEXT
        seq_hidden_t, seq_cell_t : hidden and cell right BEFORE NEXT occurrence
        '''
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_to_current,
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        # size of outputs of this scan :
        # T * size_batch * dim_model
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t), c(t), and decay(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        #
        shape_hidden = seq_cell_target.shape
        # [ T , size_batch , dim_model ]
        shape_sims_index = seq_sims_index_in_hidden.shape
        # [ N, size_batch ]
        #
        seq_cell_target_sims = seq_cell_target.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_cell_sims = seq_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_decay_cell_sims = seq_decay_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_gate_output_sims = seq_gate_output.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        #
        seq_cell_with_time_sims = seq_cell_target_sims + (
            seq_cell_sims - seq_cell_target_sims
        ) * tensor.exp(
            -seq_decay_cell_sims * seq_sims_time_to_current[:, :, None]
        )
        seq_hidden_with_time_sims = seq_gate_output_sims * tensor.tanh(
            seq_cell_with_time_sims
        )
        #
        lambda_over_seq_sims_tilde = tensor.tensordot(
            seq_hidden_with_time_sims, self.W_alpha,
            (2, 0)
        )
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        lambda_sum_over_seq_sims = tensor.sum(
            lambda_over_seq_sims, axis=2
        )
        lambda_sum_over_seq_sims *= seq_sims_mask
        # N * size_batch
        term_3 = tensor.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end / num_sims_start_to_end
        #
        #
        term_2 = numpy.float32(0.0)
        #
        #
        # compute term_1
        # as the same procedure as term_3, but easier
        # since we can directly use
        # seq_hidden_t : T * size_batch * dim_model
        #
        lambda_over_seq_tilde = tensor.tensordot(
            seq_hidden_t, self.W_alpha,
            (2, 0)
        )
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde
        )
        # T * size_batch * dim_process
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis = 2
        )
        # T * size_batch
        new_shape_0 = lambda_over_seq.shape[0]*lambda_over_seq.shape[1]
        new_shape_1 = lambda_over_seq.shape[2]
        #
        back_shape_0 = lambda_over_seq.shape[0]
        back_shape_1 = lambda_over_seq.shape[1]
        #
        lambda_target_over_seq = lambda_over_seq.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            seq_type_event[1:,:].flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        #
        log_lambda_target_over_seq = tensor.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask
        #
        log_lambda_sum_over_seq = tensor.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask
        #
        term_1 = tensor.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = tensor.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = tensor.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = tensor.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        self.cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        #
        self.log_likelihood_seq = log_likelihood_seq_batch
        self.log_likelihood_type = log_likelihood_type_batch
        self.log_likelihood_time = log_likelihood_time_batch
        #
        self.num_of_events = tensor.sum(seq_mask)
        #
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        #
        #
    #
    #
    def compute_lambda(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_sims_time_to_current,
        seq_sims_index_in_hidden,
        seq_sims_mask
    ):
        '''
        use this function to compute intensity
        seq_type_event : (T+1) * size_batch -- k_i
        seq_time_rep : (T+1) * size_batch * dim_time --
        for each data and each time step, track the time features of event k_i
        seq_sims_time_to_current : N * size_batch -- s_j - t_i
        seq_sims_index_in_hidden : N * size_batch -- int32
        seq_sims_mask : N * size_batch -- 1/0
        '''
        print "computing loss function of Neural Hawkes model ... "
        #
        # we first process the past history of events with LSTM
        seq_emb_event = self.Emb_event[seq_type_event, :]
        '''
        seq_type_event is (T + 1) * size_batch
        the 0-th is BOS event
        the 1-to-T is regular event
        regular event id is 0, 1, 2, ..., K-1
        the BOS is K
        this setting is easier for the use of seq_type_event
        '''
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        #
        '''
        # This tensor is used to compute effect/decay term
        # it will be used to compute term_1 and term_3
        # the (t, m, d) entry of this tensor is :
        # in m-th data in batch, before t-th event happen,
        # at the d-th dimention, the value of hidden unit
        '''
        #
        # first compute the 3rd term in loss
        # self.delta : dim_model * dim_process
        #
        '''
        while using simulation, we should feed in follows:
        seq_sims_time_to_current : time of t-t_recent_event at each simulation time for each seq in batch
        seq_sims_index_in_hidden : index of the hidden units
        at each time of simulation, so that we can extract the right h(t)
        to do this, we need to be sure the indexing is correct:
        a) reshape T * size_batch * dim_model
        to (T*size_batch) * dim_model
        b) flatten seq_sims_index_in_hidden N * size_batch
        to (N*size_batch) * null
        c) indexing to get (N*size_batch) * dim_model
        d) reshape it back to N * size_batch * dim_model
        the crucial part is to fill in the seq_sims_index_in_hidden correctly !!!
        '''
        #
        shape_hidden = seq_cell_target.shape
        # [ T, size_batch, dim_model ]
        shape_sims_index = seq_sims_index_in_hidden.shape
        #
        seq_cell_target_sims = seq_cell_target.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_cell_sims = seq_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_decay_cell_sims = seq_decay_cell.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        seq_gate_output_sims = seq_gate_output.reshape(
            (shape_hidden[0]*shape_hidden[1], shape_hidden[2])
        )[
            seq_sims_index_in_hidden.flatten(), :
        ].reshape(
            (
                shape_sims_index[0], shape_sims_index[1], shape_hidden[2]
            )
        )
        # N * size_batch * dim_model
        #
        seq_cell_with_time_sims = seq_cell_target_sims + (
            seq_cell_sims - seq_cell_target_sims
        ) * tensor.exp(
            -seq_decay_cell_sims * seq_sims_time_to_current[:, :, None]
        )
        seq_hidden_with_time_sims = seq_gate_output_sims * tensor.tanh(
            seq_cell_with_time_sims
        )
        #
        lambda_over_seq_sims_tilde = tensor.tensordot(
            seq_hidden_with_time_sims, self.W_alpha,
            (2, 0)
        )
        # N * size_batch * dim_process
        lambda_over_seq_sims = self.soft_relu_scale(
            lambda_over_seq_sims_tilde
        )
        # N * size_batch * dim_process
        # (2,0,1) --> dim_process * N * size_batch
        '''
        this block is to compute intensity
        '''
        self.lambda_samples = lambda_over_seq_sims.transpose((2,0,1)) * seq_sims_mask[None,:,:]
        self.num_of_samples = tensor.sum(seq_sims_mask)
        #
        #
    #
    #
    def compute_prediction_loss(
        self,
        seq_type_event, #seq_time_rep,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss for neural Hawkes with continuous-time LSTM ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        # seq_hidden_t : T * size_batch * dim_model
        seq_cell_with_time = seq_cell_target[
            :, :, :, None
        ] + (
            seq_cell[:, :, :, None] - seq_cell_target[:, :, :, None]
        ) * tensor.exp(
            -seq_decay_cell[:, :, :, None] * time_diffs[
                None, None, None, :
            ]
        )
        # T * size_batch * dim_model * M
        seq_hidden_with_time = seq_gate_output[
            :, :, :, None
        ] * tensor.tanh(
            seq_cell_with_time
        )
        # T * size_batch * dim_model * M
        lambda_over_seq_tilde = tensor.sum(
            seq_hidden_with_time[
                :, :, :, None, :
            ] * self.W_alpha[
                None, None, :, :, None
            ], axis = 2
        )
        # T * size_batch * dim_process * M
        # each time stamp, each seq in batch
        # each process, each simulation for prediction
        lambda_over_seq = self.soft_relu_scale(
            lambda_over_seq_tilde.dimshuffle(3,0,1,2)
        ).dimshuffle(1,2,3,0)
        #
        # T * size_batch * dim_process * M
        lambda_sum_over_seq = tensor.sum(
            lambda_over_seq, axis=2
        )
        # T * size_batch * M
        term_1 = time_diffs
        # M *
        #
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_over_seq, axis = 2
                ) / cum_num[None, None, :]
            ) * time_diffs[
                None, None, :
            ]
        )
        # T * size_batch * M
        term_3 = lambda_sum_over_seq
        # T * size_batch * M
        density = term_2 * term_3
        # T * size_batch * M
        time_prediction = tensor.mean(
            term_1[None, None, :] * density,
            axis = 2
        ) * time_diffs[-1]
        # T * size_batch
        lambda_over_seq_over_sims = lambda_over_seq[
            :, :, :, :
        ] * density[
            :, :, None, :
        ] / lambda_sum_over_seq[
            :, :, None, :
        ]
        # T * size_batch * dim_process * M
        prob_over_seq_over_type = tensor.mean(
            lambda_over_seq_over_sims, axis = 3
        ) * time_diffs[-1]
        # T * size_batch * dim_process
        prob_over_seq_over_type /= tensor.sum(
            prob_over_seq_over_type,
            axis=2,
            keepdims=True
        )
        # T * size_batch * dim_process
        #type_prediction = tensor.argmax(
        #    prob_over_seq_over_type, axis = 2
        #)
        # T * size_batch
        # Now we have :
        # time_prediction, type_prediction, seq_mask
        # all of -- T * size_batch
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #diff_type = tensor.abs_(
        #    target_type - type_prediction
        #) * seq_mask
        #diff_type = tensor.switch(
        #    diff_type >= numpy.float32(0.5),
        #    numpy.float32(1.0), numpy.float32(0.0)
        #)
        #
        #self.num_of_errors = tensor.sum(diff_type)
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    #TODO: memory efficient version of prediction loss
    def predict_each_step(
        self,
        cell_target, cell,
        decay_cell, gate_output,
        time_diffs
    ):
        # seqs : size_batch * dim_model
        # time_diffs : M
        cell_with_time = cell_target[
            :, :, None
        ] + (
            cell[:, :, None] - cell_target[:, :, None]
        ) * tensor.exp(
            -decay_cell[:, :, None] * time_diffs[
                None, None, :
            ]
        )
        # size_batch * dim_model * M
        hidden_with_time = gate_output[
            :, :, None
        ] * tensor.tanh(
            cell_with_time
        )
        # size_batch * dim_model * M
        lambda_tilde = tensor.sum(
            hidden_with_time[
                :, :, None, :
            ] * self.W_alpha[
                None, :, :, None
            ], axis = 1
        )
        # size_batch * dim_process * M
        lambda_each_step = self.soft_relu_scale(
            lambda_tilde.dimshuffle(2, 0, 1)
        ).dimshuffle(1, 2, 0)
        lambda_sum_each_step = tensor.sum(
            lambda_each_step, axis=1
        )
        # size_batch * M
        #TODO: compute integral
        term_1 = time_diffs
        cum_num = tensor.arange(
            time_diffs.shape[0]+numpy.int32(1)
        )[1:] * numpy.float32(1.0)
        # M
        term_2 = tensor.exp(
            (
                -1.0 * tensor.extra_ops.cumsum(
                    lambda_sum_each_step, axis=1
                ) / cum_num[None, :]
            ) * time_diffs[None, :]
        )
        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3
        # size_batch * M
        time_prediction_each_step = tensor.mean(
            term_1[None, :] * density, axis=1
        ) * time_diffs[-1]
        # size_batch
        lambda_each_step_over_sims = lambda_each_step[
            :, :, :
        ] * density[
            :, None, :
        ] / lambda_sum_each_step[
            :, None, :
        ]
        # size_batch * dim_process * M
        prob_over_type = tensor.mean(
            lambda_each_step_over_sims, axis=2
        ) * time_diffs[-1]
        # size_batch * dim_process
        prob_over_type /= tensor.sum(
            prob_over_type, axis=1, keepdims=True
        )
        # size_batch * dim_process
        return prob_over_type, time_prediction_each_step
    #
    #
    def compute_prediction_loss_lessmem(
        self,
        seq_type_event,
        seq_time_values,
        seq_mask,
        time_diffs
    ):
        #
        print "computing predictions loss of neural Hawkes with continuous-time LSTM ... "
        print "memory efficient version ... "
        seq_emb_event = self.Emb_event[seq_type_event, :]
        #
        initial_hidden_mat = tensor.outer(
            self.expand, self.h_0
        )
        initial_cell_mat = tensor.outer(
            self.expand, self.c_0
        )
        initial_cell_target_mat = tensor.outer(
            self.expand, self.c_0_target
        )
        # size_batch * dim_model
        # seq_emb_event and seq_emb_time start with
        # a special BOS event,
        # to initialize the h and c
        [seq_hidden_t, seq_cell_t, seq_cell_target, seq_cell, seq_decay_cell, seq_gate_output], _ = theano.scan(
            fn = self.rnn_unit,
            sequences = [
                dict(
                    input=seq_emb_event[:-1, :, :],
                    taps=[0]
                ),
                dict(
                    input=seq_time_values[1:, :],
                    taps=[0]
                )
            ],
            outputs_info = [
                dict(initial=initial_hidden_mat, taps=[-1]),
                dict(initial=initial_cell_mat, taps=[-1]),
                dict(initial=initial_cell_target_mat, taps=[-1]),
                None, None, None
            ],
            non_sequences = None
        )
        #
        #TODO: predict time and type for each step
        [prob_over_seq_over_type, time_prediction], _ = theano.scan(
            fn = self.predict_each_step,
            sequences = [
                dict(input=seq_cell_target, taps=[0]),
                dict(input=seq_cell, taps=[0]),
                dict(input=seq_decay_cell, taps=[0]),
                dict(input=seq_gate_output, taps=[0])
            ],
            outputs_info = [
                None, None
            ],
            non_sequences = time_diffs
        )
        #
        target_type = seq_type_event[1:, :]
        target_time = seq_time_values[1:, :]
        # Type first
        new_shape_0 = target_type.shape[0] * target_type.shape[1]
        new_shape_1 = self.dim_process
        back_shape_0 = target_type.shape[0]
        back_shape_1 = target_type.shape[1]
        #
        prob_over_seq = prob_over_seq_over_type.reshape(
            (new_shape_0, new_shape_1)
        )[
            tensor.arange(new_shape_0),
            target_type.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        log_prob_over_seq = tensor.log(
            prob_over_seq + numpy.float32(1e-9)
        )
        log_prob_over_seq *= seq_mask
        self.log_likelihood_type_predict = tensor.sum(
            log_prob_over_seq
        )
        #
        # Time
        diff_time = (
            target_time - time_prediction
        )**2
        diff_time *= seq_mask
        self.square_errors = tensor.sum(diff_time)
        self.num_of_events = tensor.sum(seq_mask)
        #TODO: Hamming loss for prediction checking
        #
        type_prediction = tensor.argmax(
            prob_over_seq_over_type, axis = 2
        )
        diff_type = tensor.abs_(
            target_type - type_prediction
        ) * seq_mask
        diff_type = tensor.switch(
            diff_type >= numpy.float32(0.5),
            numpy.float32(1.0), numpy.float32(0.0)
        )
        self.time_pred = time_prediction
        self.type_pred = type_prediction
        self.num_of_errors = tensor.sum(diff_type)
        #
        self.cost_to_optimize = -self.log_likelihood_type_predict / self.num_of_events + self.square_errors / self.num_of_events + self.term_reg
        #self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg
        self.grad_params = tensor.grad(
            self.cost_to_optimize, self.params
        )
        self.abs_grad_params = 0.0
        for grad_param in self.grad_params:
            self.abs_grad_params += tensor.sum(
                tensor.abs_(
                    grad_param
                )
            )
        #
        #
    #
    #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_process'] = self.dim_process
        model_dict['dim_time'] = self.dim_time
        model_dict['dim_model'] = self.dim_model
        return model_dict
        #
    #
    #
    def save_model(self, file_save):
        model_dict = self.get_model()
        print "saving model ... "
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
        #
    #
#
#

