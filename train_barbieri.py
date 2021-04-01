
"""
Train Barbieri on synthetic or BCH data. Return path to saved network. 
"""

import os
import sys
import argparse
import pickle
import copy 
import shutil


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nrrd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

from model_barbieri import Self_Supervised_FCN
import svtools as sv

#check conda environment and activate environment if necessary 
def check_conda(): 
    stream = os.popen("echo $CONDA_DEFAULT_ENV")
    output = stream.read()
    if output == 'tch1\n': 
        print("Please activate correct conda env by `conda activate barbieri_temp")
        sys.exit()

def check_savedir(args):
    savedir = args.trained_weights_dir+args.experiment_name+"/"
    if os.path.exists(savedir) and not args.debug:
        print("A folder with the same experiment name already exists! Please rename it or remove before training")
        sys.exit()
    print(f"Trained weights will be saved to: {savedir}")
    return savedir
        
#generate synthetic data 
def gen_synthetic_data(args): 
    # define ivim function
    def _ivim(b, Dp, Dt, Fp):
        return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)
    # define b values
    b_values = np.array([0,50,100,200,400,600,800])
    # training data
    if args.debug: 
        num_samples = 10 
    else: 
        num_samples = args.synthesize_size
    X_train = np.zeros((num_samples, len(b_values)))
    for i in range(len(X_train)):
        Dp = np.random.uniform(0.01, 0.1)
        Dt = np.random.uniform(0.0005, 0.002)
        Fp = np.random.uniform(0.1, 0.4)
        X_train[i, :] = _ivim(b_values, Dp, Dt, Fp)
    # add some noise
    X_train_real = X_train + np.random.normal(scale=0.01, size=(num_samples, len(b_values)))
    X_train_imag = np.random.normal(scale=0.01, size=(num_samples, len(b_values)))
    X_train = np.sqrt(X_train_real**2 + X_train_imag**2)
    
    return X_train

def gen_bch60_data(args):
    
    """Get 60 patient cases, mask them and vectorize in correct shape to match barbieri input requirements"""
    
    #training cases 
    training_data_json = "/home/ch215616/code/RoAR/JSON_subject_lists/train_n_test_60cases.json" #updated version of the JSON 
    cases = sv.read_from_json(training_data_json)
    training_cases = cases['cases_train']
    #full path to training cases
    training_cases_path = ["/home/ch215616/code/IVIM_data/train_60_cases/"+case+"/average6/acquired_signal/" for case in training_cases]

    num_train_images = args.bch_train_size if not args.debug else 1
    
    X_all = []
    for n in range(0,num_train_images+1): 
        print(n)
        # get b-value images 
        ims, header = sv.get_dwi_images(training_cases_path[n], '_averaged')
        # get mask image 
        mask, mask_header = nrrd.read(training_cases_path[n] + "mask.nrrd")
        # find non zero indices in the mask 
        x,y,z = np.nonzero(mask)
        # select these non-zero indices in the array and create two dimensions array with each list (particular b-value) item containing all non-zero (masked) voxels
        ims_ = [im[x,y,z] for im in ims]
        #transform list into array 
        ims_ = np.asarray(ims_)
        #normalize the data by dividing everything by the first b-value 
        ims_norm = ims_/ims_[0,:]
        # swap axes and feed into X_train 
        ims_norm_swap = np.swapaxes(ims_norm,0,1)
        X_all.append(ims_norm_swap)
    #add all training sets together 
    X_train = np.concatenate(X_all,axis=0)

    return X_train    

# instantiate the network and dataloader 
def initialize_network(args): 
    # define b values
    b_values = np.array([0,50,100,200,400,600,800])
    # Network
    b_values_no0 = torch.FloatTensor(b_values[1:])
    net = Self_Supervised_FCN(b_values_no0)
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learningrate) 
    
    return net, criterion, optimizer
    
# create batch queues 
def batch_queues(X_train,args):
    # debug 
    if args.debug and len(X_train)>100000:
        X_train = X_train[0:100000,:]
        
        
    batch_size = args.batchsize
    num_batches = len(X_train) // batch_size
    X_train = X_train[:,1:] # exlude the b=0 value as signals are normalized
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)
    return trainloader

# train the network 
def train(trainloader, net, criterion, optimizer,args): 
    # Best loss
    best_loss = 1e16
    num_bad_epochs = 0
    patience = args.patience if not args.debug else 1
    
    # Train
    for epoch in range(1000): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.
        
        

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best_loss:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best_loss = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best_loss))
                break
    print("Done")
    # Restore best model
    net.load_state_dict(final_model)  
    trained_model = {'net':net,'best_loss':best_loss,'epoch':epoch,'optimizer':optimizer}
    
    return trained_model
    
def save_network(savedir, trained_model, args): 
    os.makedirs(savedir,exist_ok=True)
    torch.save({
                'epoch': trained_model['epoch'],
                'model_state_dict': trained_model['net'].state_dict(),
                'optimizer_state_dict': trained_model['optimizer'].state_dict(),
                'loss': trained_model['best_loss'],
                'input_type':args.input,
                'args':args,
                }, savedir+"final_model.pt")
    
def gather_options(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='synthetic', choices=['synthetic','miccai10','bch60'], help='choose what data to train')
    parser.add_argument('--batchsize', type=int, default=128)    
    parser.add_argument('--learningrate', type=int, default=0.001)    
    parser.add_argument('--patience', type=int, default=10, help='choose threshold for early stopping (stop after N instances of increasing loss)')            
    parser.add_argument('--debug', action='store_true', help='choose this flag to limit training sample to 10 voxels. ')                    
    parser.add_argument('--synthesize_size', type=int, default=100000, help='size of synthetic datapoints dataset to generate')
    parser.add_argument('--experiment_name','-name', type=str, required=True, help='experiment name. Used for saving the network weights. ')    
    parser.add_argument('--bch_train_size', type=int, default=58, help='number of patient training cases to load, range of 1-58')        
    parser.add_argument('--trained_weights_dir', type=str, default="/home/ch215616/code/barbieri/trained_weights/", help='rootdir for saved weights')        
    args = parser.parse_args()
    
    return args 

def save_source(args,savedir):
    
    # create source dir
    savedir = savedir+"source/"
    os.makedirs(savedir,exist_ok=True)
    
    # save args     
    with open(savedir+"args.pkl", 'wb') as f:
        pickle.dump(args, f)
        
    # save this script  
    train_file = sys.argv[0]
    base,name = os.path.split(train_file)
    shutil.copy(train_file,savedir+name)
    
    # save imported model (assuming we are using Self_Supervised_FCN)
    import_module_name = Self_Supervised_FCN.__module__
    module_path = __import__(import_module_name).__file__
    base,name = os.path.split(module_path)
    shutil.copy(train_file,savedir+name) 

if __name__ == "__main__": 
    args = gather_options()
    
    #check if the correct conda environment is initialized (as torch will result in error)
    check_conda()
    
    # exit if experiment folder already exists 
    savedir = check_savedir(args)
    
    # load training data 
    if args.input == 'synthetic': 
        X_train = gen_synthetic_data(args)
    elif args.input == 'bch60':
        X_train = gen_bch60_data(args)
    else:
        print("size input type is not implemented")
        sys.exit()
    
    # create data loader 
    trainloader = batch_queues(X_train,args)
    # initialize network 
    net, criterion, optimizer = initialize_network(args)
    
    # train
    trained_model = train(trainloader, net, criterion, optimizer,args)
    
    # save network to file 
    save_network(savedir, trained_model, args)
    
    # save source files and 
    save_source(args,savedir)
    
    
    
