# name: LeGall 5-3 GPU 
# author: Declan McIntosh
# contact: contact@declanmcintosh.com
# paper: https://www.declanmcintosh.com/projects/wavelet-preprocessing-for-cnns                         

from numba import cuda
from numba import njit, jit
import numpy as np
import time

'''
Requirements:
    - CUDA enabled Nvidia GPU (for CUDA functions)
    - Python > 3.6.6
    - Numba
    - Numpy 
'''

@jit
def vert_CPU(Batch, result_L, result_H):
    for batch in range(Batch.shape[0]):
        for channel in range(Batch.shape[3]):
            for i in range(Batch.shape[1]//2):
                for j in range(Batch.shape[2]):
                    index_a = i*2-2 
                    index_b = i*2-1 
                    index_c = i*2 
                    index_d = i*2+1
                    index_e = i*2+2   
                    # High Pass 
                    s = Batch[batch,index_c,j,channel]*0.5
                    if index_b > 0:
                        s = s -Batch[batch,index_b,j,channel]*0.25 
                    if index_d < Batch.shape[1]//2:
                        s = s -Batch[batch,index_d,j,channel]*0.25
                    result_H[batch,i,j,channel] = s

                    # Low pass
                    s = -s + Batch[batch,index_c,j,channel]*1.25
                    if index_a > 0:
                        s = s -Batch[batch,index_a,j,channel]*0.125
                    if index_e < Batch.shape[1]//2:
                        s = s -Batch[batch,index_e,j,channel]*0.125
                    result_L[batch,i,j,channel] = s

    return result_L, result_H

@jit
def hor_CPU(Batch, result_L, result_H):
    for batch in range(Batch.shape[0]):
        for channel in range(Batch.shape[3]):
            for i in range(Batch.shape[1]):
                for j in range(Batch.shape[2]//2):
                    index_a = j*2-2 
                    index_b = j*2-1 
                    index_c = j*2 
                    index_d = j*2+1
                    index_e = j*2+2

                    # High Pass 
                    s = Batch[batch,i,index_c,channel]*0.5
                    if index_b > 0:
                        s = s -Batch[batch,i,index_b,channel]*0.25 
                    if index_d < Batch.shape[2]//2:
                        s = s -Batch[batch,i,index_d,channel]*0.25
                    result_H[batch,i,j,channel] = s

                    # Low pass
                    s = -s + Batch[batch,i,index_c,channel]*1.25
                    if index_a > 0:
                        s = s -Batch[batch,i,index_a,channel]*0.125
                    if index_e < Batch.shape[2]//2:
                        s = s -Batch[batch,i,index_e,channel]*0.125
                    result_L[batch,i,j,channel] = s   

    return result_L, result_H

def dwt_5_3_CPU(Batch):
    H = np.empty((Batch.shape[0],Batch.shape[1]//2,Batch.shape[2],Batch.shape[3]), dtype=np.float32)
    L = np.empty(shape=(Batch.shape[0],Batch.shape[1]//2,Batch.shape[2],Batch.shape[3]), dtype=np.float32)
    vert_CPU(Batch, L, H)
    LL = np.empty(shape=(Batch.shape[0],Batch.shape[1]//2,Batch.shape[2]//2,Batch.shape[3]), dtype=np.float32)
    LH = np.empty(shape=(Batch.shape[0],Batch.shape[1]//2,Batch.shape[2]//2,Batch.shape[3]), dtype=np.float32)
    HL = np.empty(shape=(Batch.shape[0],Batch.shape[1]//2,Batch.shape[2]//2,Batch.shape[3]), dtype=np.float32)
    HH = np.empty(shape=(Batch.shape[0],Batch.shape[1]//2,Batch.shape[2]//2,Batch.shape[3]), dtype=np.float32)
    LL, LH = hor_CPU(L, LL, LH)       
    HL, HH = hor_CPU(H, HL, HH)

    return np.concatenate((LL,LH,HL,HH),axis=3)




@cuda.jit
def vert(result_H, result_L, image):
    '''
    Takes in an image and retuns back an image of half vertical dimension. 
    Performs high pass filter over the given image.
    '''

    i, j, img =  cuda.grid(3)

    result_rows, result_cols, channels, images = result_H.shape
    image_rows, image_cols, channels, images  = image.shape
    
    if i < result_rows and j < result_cols  and img < images:
        for x in range(0,channels):
            index_a = i*2-2 
            index_b = i*2-1 
            index_c = i*2 
            index_d = i*2+1
            index_e = i*2+2

            # High Pass 
            s = image[index_c,j,x,img]*0.5
            if index_b > 0:
                s = s -image[index_b,j,x,img]*0.25 
            if index_d < result_rows:
                s = s -image[index_d,j,x,img]*0.25
            result_H[i,j,x,img] = s

            # Low pass
            s = -s + image[index_c,j,x,img]*1.25
            if index_a > 0:
                s = s -image[index_a,j,x,img]*0.125
            if index_e < result_rows:
                s = s -image[index_e,j,x,img]*0.125
            result_L[i,j,x,img] = s

@cuda.jit
def hor(result_L, result_H, image):
    '''
    Takes in an image and retuns back an image of half vertical dimension. 
    Performs high pass filter over the given image.
    '''

    i, j, img =  cuda.grid(3)
    result_rows, result_cols, channels, images = result_H.shape
    image_rows, image_cols, channels, images = image.shape

    if i < result_rows and j < result_cols and img < images:
        for x in range(0,channels):  
            index_a = j*2-2 
            index_b = j*2-1 
            index_c = j*2 
            index_d = j*2+1
            index_e = j*2+2

            # High Pass 
            s = image[i,index_c,x,img]*0.5
            if index_b > 0:
                s = s -image[i,index_b,x,img]*0.25 
            if index_d < result_rows:
                s = s -image[i,index_d,x,img]*0.25
            result_H[i,j,x,img] = s

            # Low pass
            s = -s + image[i,index_c,x,img]*1.25
            if index_a > 0:
                s = s -image[i,index_a,x,img]*0.125
            if index_e < result_rows:
                s = s -image[i,index_e,x,img]*0.125
            result_L[i,j,x,img] = s

def dwt_5_3(Batch):
    '''
    This function takes an entire batch as a numpy array before passing into the network, likely after any data agumentation and generates 4 coppies of each image with 
    the corresponding single pass LeGall 5/3 DWT. This function passes the values to the GPU, most cost is in the transfer to the GPU, it is most effecient with large batches.
    Inputs
        Batch - 4D Numpy array of shape (Rows, Cols, Channels, Batches) Rows and Cols must be multiples of 2.
    Outputs 
        Batch_DWT - 4D Numpy array of shape (Rows/2, Cols/2, Channels*4, Batches)
    '''
    # Send batch array to GPU
    Batch_cuda = cuda.to_device(Batch)
    assert len(Batch.shape) == 4, "Incompatable number of dimensions!"

    Rows, Cols, Channels, Batches = Batch.shape

    assert Rows > Channels and Cols > Channels, "Appears you have you channels missordered, performance will be degraded!"

    # Create target containers on GPU
    L = np.zeros_like(Batch, shape=(Rows//2, Cols, Channels, Batches))
    L_cuda = cuda.device_array_like(L)
    H_cuda = cuda.device_array_like(L)

    block_dim = (16,16,2)
    grid_dim = (L.shape[0]//block_dim[0]+1, L.shape[1]//block_dim[1]+1, L.shape[3]//block_dim[2]+1)

    vert[grid_dim,block_dim](H_cuda, L_cuda, Batch_cuda)

    LL = np.zeros_like(Batch, shape=(Rows//2, Cols//2, Channels, Batches))
    LL_cuda = cuda.device_array_like(LL)
    HL_cuda = cuda.device_array_like(LL)
    LH_cuda = cuda.device_array_like(LL)
    HH_cuda = cuda.device_array_like(LL)

    block_dim = (16,16,2)
    grid_dim = (LL.shape[0]//block_dim[0]+1, LL.shape[1]//block_dim[1]+1, LL.shape[3]//block_dim[2]+1)

    hor[grid_dim,block_dim](LH_cuda, LL_cuda, L_cuda)
    hor[grid_dim,block_dim](HH_cuda, HL_cuda, H_cuda)
    result = np.concatenate((LL_cuda.copy_to_host(), LH_cuda.copy_to_host(), HL_cuda.copy_to_host(), HH_cuda.copy_to_host()), axis=2)
    cuda.profile_stop()
    return result


if __name__ == '__main__':
    resolution = 512
    Channels = 3
    BatchSize = 128

    testArray = np.reshape(np.arange(0,resolution*resolution*Channels*BatchSize), (resolution,resolution,Channels,BatchSize)).astype(np.float64)
    result = dwt_5_3(testArray)
