import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy
import json

from params import *


def run_experiments(args, save_dir):

    os.environ['search_space'] = args.search_space

    from nas_algorithms import run_nas_algorithm
    from data import Data

    trials = args.trials
    
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.search_space)
    algorithm_params = algo_params(args.algo_params)
    num_algos = len(algorithm_params) 
    
    # set up search space
    mp = copy.deepcopy(metann_params)
    ss = mp.pop('search_space')
    dataset = mp.pop('dataset')
    search_space = Data(ss, dataset=dataset)

    for i in range(num_algos):
        results = {}
        #walltimes = []
        #run_data = []
        alg = algorithm_params[i]
        logging.info('[{}/{}] Running algorithm: {}'.format(i+1, num_algos, alg))
        filename = os.path.join(save_dir, '{}_{}_{}-{}.json'.format(alg['algo_name'], alg['total_queries'], 
                                                                    args.save_type, trials))
        tmpname = os.path.join(save_dir, '{}_{}_{}-{}.json.tmp'.format(alg['algo_name'], alg['total_queries'], 
                                                                    args.save_type, trials))
        s_j = 0
        # TODO:load previous results
        if os.path.exists(tmpname):
            with open(tmpname, 'r') as json_file:
                results = json.load(json_file)
            s_j = max(results.keys())
            logging.info("Experiment will be resumed from #{}".format(s_j))
        
        if os.path.exists(filename):
            logging.info("Experiment already saved! No more trials required for {}".format(alg))
        else:
            for j in range(s_j, trials):
                # run NAS algorithm
                result = {}
                result['error'] = []
                result['exec_time'] = []
                result['opt_time'] = []
                result['train_epoch'] = []

                starttime = time.time()
                algo_result, run_datum = run_nas_algorithm(alg, search_space, mp)
                #algo_result = np.round(algo_result, 5)

                # remove unnecessary dict entries that take up space
                for d in run_datum:
                    if args.save_type == 'valid':
                        result['error'].append(d['val_loss'] / 100.0)
                    elif args.save_type == 'test':
                        result['error'].append(d['test_loss'] / 100.0)
                    
                    result['opt_time'].append(d['opt_time'])
                    result['exec_time'].append(d['training_time'])                
                    result['train_epoch'].append(d['epochs'])

                    if not save_specs:
                        d.pop('spec')
                    for key in ['encoding', 'adjacency', 'path', 'dist_to_min']:
                        if key in d:
                            d.pop(key)

                results[str(j)] = result
                # saving temporary JSON result
                with open(tmpname, 'w') as json_file:
                    json_file.write(json.dumps(results))
                
                walltime = time.time()-starttime
                logging.info("{} - trial #{} takes {:.1f} sec".format(alg, j, walltime))
            
            # saving final JSON result
            with open(filename, 'w') as json_file:
                json_file.write(json.dumps(results))
        
        if os.path.exists(tmpname):
            os.remove(tmpname)

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = save_dir + '/' + args.search_space + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--search_space', type=str, default='nasbench_201_cifar100', \
        help='nasbench or darts')
    parser.add_argument('--algo_params', type=str, default='dngo', help='which parameters to use')
    #parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_type', type=str, default='valid', help='set valid or test')
    parser.add_argument('--save_dir', type=str, default='results', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')    

    args = parser.parse_args()
    main(args)
