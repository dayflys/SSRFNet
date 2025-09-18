import os
import itertools

def get_args():
    args = {
        # expeirment info
        'project'               : 'SSRFNet',
        'name'                  : 'test_code',
        'tags'                  : ['Release'],
        'description'           : '',
        'result'                : './', # final destination: result + project + name
        
        #FIXME
        #---------------------------------
        # dataset path 
        'trained_model'         : 'paht_trained_ssrfnet',
        
        'path_vox2_train'       : '/data/list2/vox2_train_samples.txt',
        'path_vox_O_trials'     : '/data/list2/vox2_testO_trials.txt',
        'path_vox_E_trials'     : '/data/list2/vox2_testE_trials.txt',
        'path_vox_H_trials'     : '/data/list2/vox2_testH_trials.txt',
        'path_vcmix_trials'     : '/data/list2/vcmix_test_trial.txt',
        'path_voxsrc_trials'    : '/data/list2/voxsrc23_test_trial.txt',
        'path_voices_dev_trials': '/data/list2/voices_dev_test_trial.txt',
        'path_voices_eval_trials': '/data/list2/voices_eval_test_trial.txt',

        # huggingface model
        'huggingface_url'       : 'microsoft/wavlm-large',
        #---------------------------------

        # experiment
        'lr_patience'           : 5,
        'stop_patience'         : 10,
        'batch_size'            : 128,
        'num_gpu'               : 2,
        
        # model
        'hidden_size'           : 1024,
        'seq_len'               : 149,
        'channel'               : 512,
        'num_hidden_layers'     : 12,
        'embedding_size'        : 192,
        'aam_margin'            : 0.2,
        'aam_scale'             : 30,
        'topk_panalty'          : (5, 0.1),
        'merge_layer_num'       : 4,
        'insert_feature_num'    : 4,

        # data processing
        'num_seg'               : 10,
        'crop_size'             : 16000 * 3, # 3sec
        'augment_probability'   : 0.8,
    
        # learning rate
        'lr'                    : 2e-4,
        'lr_gamma'              : 0.5,
        'weight_decay'          : 2e-5,
    }

    return args