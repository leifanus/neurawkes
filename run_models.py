# -*- coding: utf-8 -*-
"""
Created on Mar 18th 10:58:37 2016

run models, including training and validating

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import pandas as pd
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers
import modules.organizers as organizers

dtype=theano.config.floatX


#
def train_generalized_neural_hawkes_ctsm_time(
    input_train, tag_neural_type = 'general'
):
    '''
    this function is called to train
    generalized neural hawkes ctsm
    tag can be : general, adaptive, simple
    though simple is deprecated
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_train['seed_random']
    )
    #
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': input_train['save_file_path'],
        'mode': 'create', 'compile_time': None,
        'max_dev_log_likelihood': -1e6,
        'min_dev_error_rate': 1e6,
        'min_dev_rmse': 1e6,
        #
        'what_to_track': input_train['what_to_track'],
        #
        'args': input_train['args'],
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        #
        'tracked_best': {},
        #
        'tracked': {
            'track_cnt': None,
            'train_log_likelihood': None,
            'dev_log_likelihood': None,
            'train_log_likelihood_time': None,
            'dev_log_likelihood_time': None,
            'train_log_likelihood_type': None,
            'dev_log_likelihood_type': None,
            #
            'train_log_likelihood_type_predict': None,
            'dev_log_likelihood_type_predict': None,
            'train_rmse': None,
            'dev_rmse': None,
            'train_error_rate': None,
            'dev_error_rate': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': input_train['path_rawdata'],
            'size_batch': input_train['size_batch'],
            'to_read': [
                'train', 'dev'
            ]
        }
    )
    #
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    #
    # print "get time quantiles ... "
    # data_process.get_time_quantiles()
    # #

    model_settings = {
        'model': input_train['model'],
        'loss_type': input_train['loss_type'],
        'dim_process': data_process.dim_process,
        #
        'dim_time': data_process.dim_time,
        'dim_model': input_train['dim_model'],
        #
        'coef_l2': input_train['coef_l2'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train'],
        # 'threshold_time': numpy.copy(
        #     data_process.time_quantiles
        # ),
        'learn_rate': input_train['learn_rate'],
        'predict_lambda': input_train['predict_lambda']
    }
    #

    control = controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(
            model_settings
            )


    #
    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        total_log_likelihood = 0.0
        total_log_likelihood_time = 0.0
        total_log_likelihood_type = 0.0
        total_log_likelihood_type_predict = 0.0
        total_num_of_events = 0.0
        total_num_of_errors = 0.0
        total_square_errors = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'neural',
                multiple = numpy.int32(
                    input_train['multiple_sample_for_train']
                ),
                predict_first = input_train['predict_first']
            )
            #
            time_diffs_numpy = numpy.float32(
                numpy.array(
                    sorted(
                        numpy.random.exponential(
                            scale=1.0,
                            size=(100,)
                        )
                    )
                )
            )
            #
            log_likelihood_numpy = 0.0
            log_likelihood_time_numpy = 0.0
            log_likelihood_type_numpy = 0.0
            log_likelihood_type_predict_numpy = 0.0
            num_of_events_numpy = 0.0
            num_of_errors_numpy = 0.0
            square_errors_numpy = 0.0
            #
            if input_train['loss_type'] == 'loglikehood':
                log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_time_to_current_numpy,
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.time_since_start_to_end_numpy,
                    data_process.num_sims_start_to_end_numpy,
                    data_process.seq_mask_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_index_in_hidden_numpy,
                    data_process.seq_sims_mask_numpy
                )
            else:
                log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy = control.model_learn(
                    data_process.seq_type_event_numpy,
                    data_process.seq_time_values_numpy,
                    data_process.seq_mask_numpy,
                    time_diffs_numpy
                )
                #print "gradient absoluate value : ", grad_numpy
            #
            #
            log_dict['iteration'] += 1
            #
            total_log_likelihood += log_likelihood_numpy
            total_log_likelihood_time += log_likelihood_time_numpy
            total_log_likelihood_type += log_likelihood_type_numpy
            total_log_likelihood_type_predict += log_likelihood_type_predict_numpy
            total_num_of_events += num_of_events_numpy
            total_num_of_errors += num_of_errors_numpy
            total_square_errors += square_errors_numpy
            #
            #
            log_dict['tracked']['train_log_likelihood'] = round(
                total_log_likelihood / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_time'] = round(
                total_log_likelihood_time / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type'] = round(
                total_log_likelihood_type / total_num_of_events, 4
            )
            log_dict['tracked']['train_log_likelihood_type_predict'] = round(
                total_log_likelihood_type_predict / total_num_of_events, 4
            )
            log_dict['tracked']['train_rmse'] = round(
                numpy.sqrt(
                    total_square_errors / total_num_of_events
                ), 8
            )
            log_dict['tracked']['train_error_rate'] = round(
                total_num_of_errors / total_num_of_events, 4
            )
            train_end = time.time()
            #
            log_dict['tracked']['train_time'] = round(
                (train_end - train_start)*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: get the dev loss values
                total_log_likelihood_dev = 0.0
                total_log_likelihood_time_dev = 0.0
                total_log_likelihood_type_dev = 0.0
                total_log_likelihood_type_predict_dev = 0.0
                total_num_of_events_dev = 0.0
                total_num_of_errors_dev = 0.0
                total_square_errors_dev = 0.0
                #
                for step_dev in range(data_process.max_nums['dev']):
                    #
                    #
                    data_process.process_data(
                        tag_batch = 'dev',
                        idx_batch_current = step_dev,
                        tag_model = 'neural',
                        multiple = numpy.int32(
                            input_train[
                                'multiple_sample_for_dev'
                            ]
                        ),
                        predict_first = input_train['predict_first']
                    )
                    #
                    time_diffs_numpy = numpy.float32(
                        numpy.array(
                            sorted(
                                numpy.random.exponential(
                                    scale=1.0,
                                    size=(100,)
                                )
                            )
                        )
                    )
                    #
                    #
                    log_likelihood_numpy = 0.0
                    log_likelihood_time_numpy = 0.0
                    log_likelihood_type_numpy = 0.0
                    log_likelihood_type_predict_numpy = 0.0
                    num_of_events_numpy = 0.0
                    num_of_errors_numpy = 0.0
                    square_errors_numpy = 0.0
                    #
                    #
                    if input_train['loss_type'] == 'loglikehood':
                        log_likelihood_numpy, log_likelihood_time_numpy, log_likelihood_type_numpy, num_of_events_numpy = control.model_dev(
                            data_process.seq_time_to_current_numpy,
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.time_since_start_to_end_numpy,
                            data_process.num_sims_start_to_end_numpy,
                            data_process.seq_mask_numpy,
                            data_process.seq_sims_time_to_current_numpy,
                            data_process.seq_sims_index_in_hidden_numpy,
                            data_process.seq_sims_mask_numpy
                        )
                    else:
                        log_likelihood_type_predict_numpy, num_of_errors_numpy, square_errors_numpy, num_of_events_numpy, time_pred, type_pred = control.model_dev(
                            data_process.seq_type_event_numpy,
                            data_process.seq_time_values_numpy,
                            data_process.seq_mask_numpy,
                            time_diffs_numpy
                        )
                        #print "gradient absoluate value : ", grad_numpy
                        #
                    #
                    time1 = str(time.time())
                    time_pred = pd.DataFrame(time_pred)
                    time_pred.to_csv('./tmp/time_pred_' + time1 + '.csv')
                    type_pred = pd.DataFrame(type_pred)
                    type_pred.to_csv('./tmp/type_pred_' + time1 + '.csv')


                    total_log_likelihood_dev += log_likelihood_numpy
                    total_log_likelihood_time_dev += log_likelihood_time_numpy
                    total_log_likelihood_type_dev += log_likelihood_type_numpy
                    total_log_likelihood_type_predict_dev += log_likelihood_type_predict_numpy
                    total_num_of_events_dev += num_of_events_numpy
                    total_num_of_errors_dev += num_of_errors_numpy
                    total_square_errors_dev += square_errors_numpy
                    #
                    if step_dev % 10 == 9:
                        print "in dev, the step is out of ", step_dev, data_process.max_nums['dev']
                #
                #
                log_dict['tracked']['dev_log_likelihood'] = round(
                    total_log_likelihood_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_time'] = round(
                    total_log_likelihood_time_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_log_likelihood_type'] = round(
                    total_log_likelihood_type_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_log_likelihood_type_predict'] = round(
                    total_log_likelihood_type_predict_dev / total_num_of_events_dev, 4
                )
                #
                log_dict['tracked']['dev_error_rate'] = round(
                    total_num_of_errors_dev / total_num_of_events_dev, 4
                )
                log_dict['tracked']['dev_rmse'] = round(
                    numpy.sqrt(
                        total_square_errors_dev / total_num_of_events_dev
                    ), 8
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round(
                    dev_end - dev_start, 0
                )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                if log_dict['what_to_track'] == 'loss':
                    if log_dict['tracked']['dev_log_likelihood'] > log_dict['max_dev_log_likelihood']:
                        #
                        #name_file = 'model_'+str(log_dict['tracked']['track_cnt'])+'.pkl'
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                elif log_dict['what_to_track'] == 'rmse':
                    if log_dict['tracked']['dev_rmse'] < log_dict['min_dev_rmse']:
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                elif log_dict['what_to_track'] == 'rate':
                    if log_dict['tracked']['dev_error_rate'] < log_dict['min_dev_error_rate']:
                        name_file = 'model.pkl'
                        save_file = os.path.abspath(
                            log_dict['save_file_path']
                        ) + '/'+name_file
                        #
                        control.save_model(save_file)
                else:
                    print "what tracker ? "
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
    #
