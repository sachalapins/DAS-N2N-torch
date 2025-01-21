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
import importlib.resources as pkg_resources



class DASN2N(nn.Module):
    
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
    
    
    def load_weights(self, weights_path=None):
        
        ''' 
        Function to load pre-trained DAS-N2N model weights. 
        '''

        model_device = next(self.parameters()).device # Identify model device
        if weights_path:
            # Load weights from user specified path
            self.load_state_dict(torch.load(weights_path, weights_only=True, map_location=model_device))
        else:
            # Load default weights from package
            with pkg_resources.files("dasn2n").joinpath("weights/original.pt") as package_weights:
                self.load_state_dict(torch.load(package_weights, weights_only=True, map_location=model_device))
    

    def denoise_numpy(self, das_numpy_array, batch_size = 64, overlap = True, step = 0.95, normalize = True, remove_mean_axis = None, std_norm_axis = None, channel_block_width = 1, rmean_on_end = True, track_processing_time = False):

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
            offset = np.mean(das_numpy_array, axis=remove_mean_axis, keepdims=True)
            if (remove_mean_axis != None) & (channel_block_width > 1):
                if (int(remove_mean_axis) == 0):
                    offset = np.array([np.pad(offset[norm_offset:], (0, norm_offset), mode='reflect') for norm_offset in range(channel_block_width)])
            das_numpy_array = das_numpy_array - offset
            norm_factor = np.std(das_numpy_array, axis=std_norm_axis, keepdims=True)
            if (remove_mean_axis != None) & (channel_block_width > 1):
                if (int(remove_mean_axis) == 0):
                    norm_factor = np.array([np.pad(norm_factor[norm_offset:], (0, norm_offset), mode='reflect') for norm_offset in range(channel_block_width)])
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
            