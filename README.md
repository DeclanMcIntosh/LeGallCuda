# LeGall 5/3 Deep Learning Pre-processing

This is an effecient CUDA implementation of the pre-processing step proposed in *Preservation of High Frequency Content for Deep Learning-Based Medical Image Classification*, by Declan McIntosh, Tunai Porto Marques and Alexandra Branzan Albu. The original paper was published in the 18th Conferance on Robotics and Vision. 

This novel method of pre-processing for CNNs can take images at twice the desired height and width and refactor them into 4 representative images at the desired image size. This allows the CNN to effectively "see" at 4x the actual input resolution. This increases accuracy as shown in the original paper. In most modern CNNs increasing the channels of the input by 4x only increases overall FLOPS in inferance by <0.1%. 

Poster can be accessed at https://www.declanmcintosh.com/projects/wavelet-preprocessing-for-cnns


### Requirements
- numba
- cuda
- Python > 3.6.6
- Nvida GPU (Compute > 3.0)

### Cite this work

If you use this work please cite its related publication:


### BibTeX
```
@INPROCEEDINGS{McIntosh_2021_CRV,
title={Preservation of High Frequency Content for Deep Learning-Based Medical Image Classification},
author={McIntosh, Declan and Porto Marques, Tunai and Branzan Albu, Alexandra},
booktitle={2020 17th Conference on Computer and Robot Vision (CRV)},
year={2021}
}
```
