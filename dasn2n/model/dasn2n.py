'''
Copyright (c) 2024 Sacha Lapins, University of Bristol

This file is part of the dasn2n package.

The dasn2n package is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The dasn2n package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the dasn2n package. If not, see <https://www.gnu.org/licenses/>.

'''


import os
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import gc
import random
from skimage.util import view_as_blocks, view_as_windows # For cutting up data into patches
import tdms_reader


class dasn2n(nn.Module):
    
    def __init__(self):
        super().__init__()

        '''
        Define DAS-N2N model layers 
        '''
        
        self.INPUT_SHAPE = [128,96] # Hard code input shape from paper: 128 time samples, 96 DAS channels
        self.conv00 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv10 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv01a = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv01b = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.out01 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.down = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, inp):
        
        '''
        Forward pass of DAS-N2N model
        '''
        
        x = torch.unsqueeze(inp, dim=1) # Expand dims
                    
        # Encoder layer
        x = self.act(self.conv00(x))
        enc_skip = x
        
        # Middle layer
        x = self.act(self.conv10(self.down(x)))

        # Decoder layer
        x = torch.cat((enc_skip, self.up(x)), -3)
        x = self.act(self.conv01a(x))
        x = self.act(self.conv01b(x))
        x = self.out01(x)
        
        return torch.reshape(x, (inp.shape))
    
    
    def load_weights(self, weights_path):
        
        ''' 
        Function to load pre-trained DAS-N2N model weights. 
        '''
        
        self.load_state_dict(torch.load(weights_path))
        
    
    
    def denoise_numpy(self, das_numpy_array, batch_size = 64, overlap = True, step = 0.95, normalize = True, remove_mean_axis = None, std_norm_axis = None, rmean_on_end = True, track_processing_time = False):

        '''
        Function to run DAS-N2N model on a 2D numpy array containing DAS data.

        Inputs:
        - das_numpy_array: 2D numpy array containing DAS data, shape = (time_samples, das_channels).
        - batch_size: number of 128x96 patches to process at a time (depends on available memory)
        - overlap: if True, run model on overlapping patches of data
        - step: step size (proportion of patch size) for overlap. Values close to 1 discard only edges and are most efficient.
        - track_processing_time: if True, prints the time taken at various steps of function code.

        '''
                
        if track_processing_time:
            function_start = datetime.now()
            print("Starting function: " + str(function_start))

        # Set model to eval mode:
        self.eval()

        # Standardise data (by std)
        if normalize:
            das_numpy_array = das_numpy_array - np.mean(das_numpy_array, axis=remove_mean_axis, keepdims=True)
            norm_factor = np.std(das_numpy_array, axis=std_norm_axis, keepdims=True)
            das_numpy_array = das_numpy_array / norm_factor
            
        # Make sure data type is float32 (required for torch model):
        if das_numpy_array.dtype != 'float32':
            das_numpy_array = das_numpy_array.astype('float32')

        if track_processing_time:
            print("After normalisation: " + str(datetime.now()))

        # Pad and split data into blocks for model processing
        das_array_shape = das_numpy_array.shape
        
        if overlap:
            edge_size_t = np.max([0,int(self.INPUT_SHAPE[-2] * (1. - step)) // 2])
            edge_size_c = np.max([0,int(self.INPUT_SHAPE[-1] * (1. - step)) // 2])

            new_block_size = (self.INPUT_SHAPE[-2] - (edge_size_t * 2), self.INPUT_SHAPE[-1] - (edge_size_c * 2))

            # Pad array
            das_array_pad = np.pad(das_numpy_array, 
                                   ((edge_size_t, new_block_size[0] - (das_numpy_array.shape[-2] % new_block_size[0]) + edge_size_t), 
                                    (edge_size_c, new_block_size[1] - (das_numpy_array.shape[-1] % new_block_size[1]) + edge_size_c)), 
                                   mode='reflect')

            # Window data (overlapping)
            das_array_blocks = view_as_windows(das_array_pad, self.INPUT_SHAPE, step=new_block_size)
        else:
            # Pad array
            das_array_pad = np.pad(das_numpy_array, 
                                   ((0, self.INPUT_SHAPE[-2] - (das_numpy_array.shape[-2] % self.INPUT_SHAPE[-2])), 
                                    (0, self.INPUT_SHAPE[-1] - (das_numpy_array.shape[-1] % self.INPUT_SHAPE[-1]))), 
                                   mode='reflect')

            # Window data (not overlapping)
            das_array_blocks = view_as_blocks(das_array_pad, self.INPUT_SHAPE)

        das_array_pad_shape = das_array_pad.shape
        das_array_blocks_shape = das_array_blocks.shape

        # Clear some memory
        del(das_numpy_array)
        del(das_array_pad)
        gc.collect()

        # Reshape to 3D array to feed into model (no_blocks, samples, channels)
        model_in = np.reshape(das_array_blocks, (-1, das_array_blocks.shape[2], das_array_blocks.shape[3]))

        # Clear some memory
        del(das_array_blocks)
        gc.collect()

        if track_processing_time:
            print("After padding and reshaping data: " + str(datetime.now()))

        # Create dataloader
        model_in_dataset = ArrayDataset(model_in)
        model_in_loader = torch.utils.data.DataLoader(model_in_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=True)
        if track_processing_time:
            print("After creating dataloader: " + str(datetime.now()) + ". Running model.")

        # Run model
        if track_processing_time:
            before = datetime.now()

        try:
            model_device = next(self.parameters()).device.type # Inherit device from model
        except:
            model_device = 'cpu'

        num_elements = len(model_in_loader.dataset)
        num_batches = len(model_in_loader)
        batch_size = model_in_loader.batch_size
        das_array_out = torch.zeros(model_in.shape)

        with torch.no_grad():
            for i, patches in enumerate(model_in_loader):
                start = i * batch_size
                end = start + batch_size
                if i == num_batches - 1:
                    end = num_elements
                if model_device != 'cpu':
                    patches = patches.to(model_device)
                preds = self(patches)
                das_array_out[start:end, :, :] = preds

        das_array_out = das_array_out.numpy()

        if track_processing_time:
            after = datetime.now()
            print("After running model: " + str(after) + ". Running model took " + str((after-before).total_seconds()) + " seconds.")

        # Clear memory
        del(model_in)
        del(model_in_dataset)
        del(model_in_loader)
        gc.collect()

        # Reshape and do inverse of view_as_blocks
        das_array_out = np.reshape(das_array_out, (das_array_blocks_shape[0], das_array_blocks_shape[1], das_array_blocks_shape[2], das_array_blocks_shape[3]))

        if overlap:
            # Keep only middle samples/channels to form non-overlapping smaller blocks and invert back to original shape as before
            das_array_out = das_array_out[:, :, edge_size_t:-edge_size_t, edge_size_c:-edge_size_c]
            das_array_new_shape = (das_array_pad_shape[0] - (2 * edge_size_t), das_array_pad_shape[1] - (2 * edge_size_c))
            das_array_out = das_array_out.transpose(0,2,1,3).reshape(das_array_new_shape)

            # Pad channels then cut back to unpadded size:
            das_array_out = np.pad(das_array_out, 
                                  ((0, edge_size_t), 
                                   (0, edge_size_c)), 
                                  mode='constant') # Pad end channels with zeros
        else:
            das_array_out = das_array_out.transpose(0,2,1,3).reshape(das_array_pad_shape)

        # Cut back to unpadded size:
        das_array_out = das_array_out[:das_array_shape[0],:das_array_shape[1]]

        # Un-normalise
        if normalize:
            das_array_out = das_array_out * norm_factor

        # Remove channel-wise mean?:
        if rmean_on_end:
            das_array_out = das_array_out - np.mean(das_array_out, axis=0, keepdims=True)

        if track_processing_time:
            function_end = datetime.now()
            print("After re-scaling predictions and reshaping back to original signal dims: " + str(function_end))
            print("TOTAL TIME TAKEN: " + str((function_end-function_start).total_seconds()) + " seconds.")

        return das_array_out
    
    
    def train_model(self, 
                    file_list,    # List of tdms or numpy array files containing DAS data
                    file_dict,    # Dictionary indicating which channels to use as input/target data for each file
                    no_epochs,    # Number of training epochs
                    loss = "mse",    # Loss function to use (currently only mean squared error ("mse") or mean absolute error ("mae")
                    model_save_dir = ".",    # Path to dir to save model weights
                    file_batch_size = 2,    # Number of tdms/numpy files to read in at a time (128x96 sections of data shuffled within files)
                    mini_batch_size = 24,    # Number of 128x96 sections per training batch
                    skip_epochs = 0,    # Number of training epochs to skip (e.g., if continuing training from checkpoint)
                    augmentations = True,    # Use paper training augmentation (random flipping + random swapping input/target data)
                    remove_mean_axis = None,    # Axis along which to normalise data (paper uses None but normalisation along each DAS channel, i.e. axis=0, recommended)
                    std_norm_axis = None,    # Axis along which to normalise data (paper uses None but normalisation along each DAS channel, i.e. axis=0, recommended)
                    init_lr = 1e-4,    # Initial learning rate
                    min_lr = 1e-6,    # End learning rate
                    use_grad_clip = True,    # Use gradient clipping during training
                    use_scheduler = True,    # Cosine anneal learning rate from init_lr to min_lr over no_epochs
                    save_every_five_file_batches = False,    # Save after every (5 * file_batch_size) files
                    save_on_epoch_end = True,    # Save at the end of every epoch (i.e., each iteration through all training files)
                    verbose = True    # Verbosity flag
                   ):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.train() # Set to training mode
        
        # Inherit device from model
        model_device = next(self.parameters()).device.type
                    
        # Set optimizer and scheduler
        lr = min_lr + (0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * (skip_epochs / no_epochs))))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=0)
        if use_scheduler:
            scheduler_steps = int(np.ceil(len(file_list)/file_batch_size)) * (no_epochs - skip_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_steps, eta_min=min_lr)
            
        # Check model_save_dir doesn't have trailing slash:
        if model_save_dir[-1] == "/":
            model_save_dir = model_save_dir[:-1]
                
        train_loss = [] # List to hold training loss for each file_batch
                    
        # Loop through number of training epochs, training model on all files in file_list in each epoch
        for epoch in range(no_epochs):
            
            print(f"Epoch: {(epoch+1):>5d} of {no_epochs:>5d}")
            print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            
            if epoch < skip_epochs:
                print("Skipping epoch.")
                continue
            
            running_loss = 0.
            no_of_samples_processed = 0
            
            # Shuffle files with training data and break into chunks of length file_batch_size
            train_paths = file_list
            random.shuffle(train_paths)
            train_paths_chunks = list(chunk_list(train_paths, file_batch_size))
            
            # Loop through each file batch size generating training samples and then training model
            n = 0
            for f in train_paths_chunks:
                n += 1
                if verbose:
                    if n % 5 == 0:
                        print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                    print(f"File batch: {(n):>5d} of {len(train_paths_chunks):>5d}")

                # Generate training samples from file batch
                tdms_gen = training_data_generator(file_list = f,
                                                   file_dict = file_dict,
                                                   batch_size = file_batch_size, 
                                                   remove_mean_axis = remove_mean_axis,
                                                   std_norm_axis = std_norm_axis,
                                                   random_samp_start = augmentations, 
                                                   random_chan_start = augmentations, 
                                                   random_flip = augmentations,
                                                   randomise_sides = augmentations,
                                                   pad_data = True
                                              )

                # Create new dataloader for model training (shuffles data)
                x, y = tdms_gen[0]
                    
                train_data = training_set_batched(x, y)
                dataloader = torch.utils.data.DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
                size = len(dataloader.dataset)

                file_batch_loss = 0.

                # Loop through training batches and train model
                for batch, (X, y) in enumerate(dataloader):
                    X, y = X.to(model_device), y.to(model_device)
                    X, y = X.float(), y.float()

                    # Compute prediction error
                    pred = self.forward(X) # forward pass: predict outputs for each image
                    if loss == "mse":
                        batch_loss = torch.mean((pred - y)**2) # calculate MSE loss          
                    else:
                        batch_loss = torch.mean(abs(pred - y)) # calculate MAE loss          

                    # Backpropagation
                    optimizer.zero_grad() # clear gradients (otherwise they accumulate and explode)
                    batch_loss.backward() # backward pass: compute gradient of the loss wrt model parameters
                    if use_grad_clip:
                        nn.utils.clip_grad_norm_(self.parameters(), 0.01) # gradient clipping
                    optimizer.step() # update parameters

                    # Gather data and report
                    running_loss += (batch_loss.item() * len(X))
                    file_batch_loss += (batch_loss.item() * len(X))

                    # Report batch loss after every 10 batches
                    if verbose:
                        if batch % 10 == 0:
                            batch_loss, current = batch_loss.item(), batch * len(X)
                            print(f"loss: {batch_loss:>7f}  [{current:>5d}/{size:>5d}]")

                no_of_samples_processed += size
                train_loss.append(file_batch_loss / size)
                
                # Apply Cosine Annealing LR scheduler after every file batch
                if use_scheduler:
                    scheduler.step()
                
                if save_every_five_file_batches:
                    # Save model after every 5 file batches:
                    if n % 5 == 0:
                        if not os.path.exists(model_save_dir):
                            print("Model save directory does not exist, attempting to create directory...")
                            os.mkdir(model_save_dir)
                            if not os.path.exists(model_save_dir):
                                print("Could not make save directory, saving model state to current directory instead.")
                                model_save_dir = "."
                        model_path = model_save_dir + '/model_{}_{}_{}'.format(timestamp, epoch, n)
                        torch.save(self.state_dict(), model_path)
            
            # Save model and optimizer state at each epoch end
            if save_on_epoch_end:
                if not os.path.exists(model_save_dir):
                    print("Model save directory does not exist, attempting to create directory...")
                    os.mkdir(model_save_dir)
                    if not os.path.exists(model_save_dir):
                        print("Could not make save directory, saving model state to current directory instead.")
                        model_save_dir = "."

                model_path = model_save_dir + '/model_{}_{}'.format(timestamp, epoch)
                torch.save(self.state_dict(), model_path)

                optimizer_path = model_save_dir + '/optimizer_{}_{}'.format(timestamp, epoch)
                torch.save(optimizer.state_dict(), optimizer_path)

                # Save training loss
                open_file = open(model_save_dir + '/model_{}_train_loss.pkl'.format(timestamp), "wb")
                pickle.dump(train_loss, open_file)
                open_file.close()
    
    
    
class ArrayDataset(torch.utils.data.Dataset):
    '''
    Simple class/functions to convert numpy array to torch Dataset. 
    Required to utilise torch DataLoader for subsequent model processing.
    '''
    
    def __init__(self, patches):
        super().__init__()

        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
    
    
def chunk_list(lst, n):
    '''
    Function to split a list into chunks of size n
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
class training_set_batched(torch.utils.data.Dataset):
    '''
    Required to create mini-batches from training set
    '''
    def __init__(self, x, y):
        super().__init__()
        self._x = x
        self._y = y

    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, index):
        x = self._x[index, :, :]
        y = self._y[index, :, :]
        
        return x, y

    
class training_data_generator(torch.utils.data.Dataset):
    '''
    Generator function for creating training patches
    '''
    def __init__(self, 
                 file_list, 
                 file_dict,
                 batch_size = 1, 
                 remove_mean_axis = None,
                 std_norm_axis = None,
                 random_samp_start = False, 
                 random_chan_start = False,
                 random_flip = False,
                 randomise_sides = False, 
                 pad_data = True
                ):
        super().__init__()
        
        self.x = file_list
        self.file_dict = file_dict        
        self.batch_size = batch_size
        self.remove_mean_axis = remove_mean_axis
        self.std_norm_axis = std_norm_axis
        self.random_samp_start = random_samp_start
        self.random_chan_start = random_chan_start
        self.random_flip = random_flip
        self.randomise_sides = randomise_sides
        self.pad_data = pad_data
        
        self.SIGNAL_LENGTH = 128
        self.N_CHANNELS = 96
            
    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]

        # Get data
        n = 0
        for file_name in batch_x:
            n += 1
            x_out = []
            y_out = []
            
            input_channels = self.file_dict[file_name.split('/')[-2]]['input_channels']
            target_channels = self.file_dict[file_name.split('/')[-2]]['target_channels']
            
            # Check input and target channels both lists:
            if type(input_channels) is not list:
                print("Input channels are not a list for file " + file_name + ". Skipping file.")
                continue

            if type(target_channels) is not list:
                print("Target channels are not a list for file " + file_name + ". Skipping file.")
                continue

            # Check number of input channels and target channels are the same
            if len(input_channels) != len(target_channels):
                print("Input and target lists are not of equal length for file " + file_name + ". Skipping file.")
                continue

            # Check each element of list are tuples and that input and target channels are equal size
            for i in range(len(input_channels)):
                if (type(input_channels[i]) is not tuple) or (type(target_channels[i]) is not tuple):
                    print("One of input_channels[" + str(i) + "] or target_channels[" + str(i) + "] is not a tuple for file " + file_name + ". Skipping file.")
                    break
                if (input_channels[i][1] - input_channels[i][0]) != (target_channels[i][1] - target_channels[i][0]):
                    print("No. of channels in input_channels[" + str(i) + "] and target_channels[" + str(i) + "] not equal for file " + file_name + ". Skipping file.")
                    break
            else:
                pass
            if (input_channels[i][1] - input_channels[i][0]) != (target_channels[i][1] - target_channels[i][0]) or (type(input_channels[i]) is not tuple) or (type(target_channels[i]) is not tuple):
                continue
        
            # Read in data and split into input and target data        
            x_out, y_out = self.get_training_data_from_file(file_name, input_channels, target_channels, self.remove_mean_axis, self.std_norm_axis)
            
            # View input and target data as smaller patches for model training
            x_out, y_out = self.split_training_data_into_blocks(x_out, y_out, self.SIGNAL_LENGTH, self.N_CHANNELS, self.random_samp_start, self.random_chan_start, self.pad_data)

            # Randomly swap blocks between input and target data (if randomise_sides = True)
            if self.randomise_sides:
                x_out, y_out = self.random_swap_input_target_blocks(x_out, y_out)
                                
            # Randomly flip blocks up-down and left-right
            if self.random_flip:
                x_out, y_out = self.random_flip_updown_leftright(x_out, y_out)
                                            
            # If file_batch_size > 1, concatenate batch
            if self.batch_size > 1:
                if n > 1:
                    x_out_final = np.concatenate((x_out_final, x_out), axis=0)
                    del(x_out)
                    y_out_final = np.concatenate((y_out_final, y_out), axis=0)
                    del(y_out)
                else:
                    x_out_final = x_out
                    del(x_out)
                    y_out_final = y_out
                    del(y_out)

            else:
                x_out_final = x_out
                del(x_out)
                y_out_final = y_out
                del(y_out)

            gc.collect()

        # Output
        return torch.from_numpy(np.float32(x_out_final)), torch.from_numpy(np.float32(y_out_final))
    
    
    def get_training_data_from_file(self, file_name, input_channels, target_channels, remove_mean_axis=None, std_norm_axis=None):

        if file_name.split('.')[-1] == "tdms":
            x_in = tdms_reader.TdmsReader(file_name)
            x_read = x_in.get_data()
        elif file_name.split('.')[-1] == "npy":
            x_read = np.load(file_name, allow_pickle=True)
        
        # Get input/target data
        x_input = x_read[:,input_channels[0][0]:input_channels[0][1]]
        x_target = np.fliplr(x_read[:,target_channels[0][0]:target_channels[0][1]])
        
        if len(input_channels) > 1:
            for i in range(1, len(input_channels)):                
                # Concatenate input/target data
                x_input = np.concatenate((x_input, x_read[:,input_channels[i][0]:input_channels[i][1]]), axis=1)
                x_target = np.concatenate((x_target, np.fliplr(x_read[:,target_channels[i][0]:target_channels[i][1]])), axis=1)
                
        # Normalise data
        if remove_mean_axis is None:
            offset = np.mean(np.concatenate((x_input, x_target), axis=1), axis=None, keepdims=True)
            x_input = x_input - offset
            x_target = x_target - offset
        else:
            x_input = x_input - np.mean(x_input, axis=remove_mean_axis, keepdims=True)
            x_target = x_target - np.mean(x_input, axis=remove_mean_axis, keepdims=True)
            
        if std_norm_axis is None:
            norm_factor = np.std(np.concatenate((x_input, x_target), axis=1), axis=None, keepdims=True)
            x_input = x_input / norm_factor
            x_target = x_target / norm_factor
        else:
            x_input = x_input / np.std(x_input, axis=std_norm_axis, keepdims=True)
            x_target = x_target / np.std(x_target, axis=std_norm_axis, keepdims=True)

        return x_input, x_target
    
    
    def split_training_data_into_blocks(self, input_data, target_data, signal_length, n_channels, random_samp_start=True, random_chan_start=True, pad_data=False):

        # Get number of blocks
        no_of_sig_blocks = input_data.shape[0] // signal_length
        no_of_cha_blocks = input_data.shape[1] // n_channels

        # Signal might not be exactly divisible by `signal_length`
        if random_samp_start:
            rand_samp_start = random.randint(0,max(0,(input_data.shape[0] % signal_length)-1))
        else:
            rand_samp_start = 0

        # Same with no. of DAS channels...
        if random_chan_start:
            rand_chan_start = random.randint(0,max(0,(input_data.shape[1] % n_channels)-1))
        else:
            rand_chan_start = 0
            
        # Reflect pad data to complete final block:
        if pad_data:
            pad_samp = rand_samp_start + ((no_of_sig_blocks + 1) * signal_length) - input_data.shape[0]
            pad_chan = rand_chan_start + ((no_of_cha_blocks + 1) * n_channels) - input_data.shape[1]
            pad_samp_l = random.randint(0, pad_samp)
            pad_samp_tuple = (pad_samp_l, pad_samp - pad_samp_l)
            pad_chan_l = random.randint(0, pad_chan)
            pad_chan_tuple = (pad_chan_l, pad_chan - pad_chan_l)
            input_data = np.pad(input_data, (pad_samp_tuple, pad_chan_tuple), 'reflect')
            target_data = np.pad(target_data, (pad_samp_tuple, pad_chan_tuple), 'reflect')

            input_data = input_data[rand_samp_start:(rand_samp_start + ((no_of_sig_blocks + 1) * signal_length)), rand_chan_start:(rand_chan_start + ((no_of_cha_blocks + 1) * n_channels))]
            target_data = target_data[rand_samp_start:(rand_samp_start + ((no_of_sig_blocks + 1) * signal_length)), rand_chan_start:(rand_chan_start + ((no_of_cha_blocks + 1) * n_channels))]
        else:
            input_data = input_data[rand_samp_start:(rand_samp_start + (no_of_sig_blocks * signal_length)), rand_chan_start:(rand_chan_start + (no_of_cha_blocks * n_channels))]
            target_data = target_data[rand_samp_start:(rand_samp_start + (no_of_sig_blocks * signal_length)), rand_chan_start:(rand_chan_start + (no_of_cha_blocks * n_channels))]
                            
        # Use skimage.util.view_as_blocks to cut up data:
        # Input data
        input_data = view_as_blocks(input_data, (signal_length, n_channels))
        x_out = np.reshape(input_data, (input_data.shape[0] * input_data.shape[1], input_data.shape[2], input_data.shape[3]))
        del(input_data) # Clear memory
        # Target data
        target_data = view_as_blocks(target_data, (signal_length, n_channels))
        y_out = np.reshape(target_data, (target_data.shape[0] * target_data.shape[1], target_data.shape[2], target_data.shape[3]))
        del(target_data) # Clear memory
        gc.collect()
        
        return x_out, y_out
        
    
    def random_swap_input_target_blocks(self, input_data, target_data):
        
        # Join input_data and target_data along channels axis
        x_y_join = np.concatenate([input_data, target_data], axis=-1)

        # Randomly flip channels (target_data will randomly swap over with input_data, but obviously channels randomly reversed again)
        join_flips = [(slice(None, None, None), # signal dim - don't flip
                       slice(None, None, random.choice([-1, None]))) # channel dim - randomly flip
                      for _ in range(x_y_join.shape[0])] # For each block
        x_y_join = np.array([ex[flip] for ex, flip in zip(x_y_join, join_flips)])

        # Split back out to input and target data:
        x_out, y_out = np.array_split(x_y_join, 2, axis=-1)
        
        return x_out, y_out
    
    
    def random_flip_updown_leftright(self, input_data, target_data):
        
        # Randomly flip blocks up-down and left-right
        # Generate random signal / channel flips to apply same to both input and target
        flips = [(slice(None, None, random.choice([-1, None])), # signal dim - randomly flip or not
                  slice(None, None, random.choice([-1, None]))) # channel dim - randomly flip or not
                 for _ in range(input_data.shape[0])] # For each block

        # Randomly flip (same for input and target)
        x_out = np.array([ex[flip] for ex, flip in zip(input_data, flips)])
        y_out = np.array([ex[flip] for ex, flip in zip(target_data, flips)])
        
        return x_out, y_out
    