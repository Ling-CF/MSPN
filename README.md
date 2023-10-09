We present a multi-scale predictive coding model for future video frames prediction. Drawing inspiration on the ``Predictive Coding" theories in cognitive science, it is updated by a combination of bottom-up and top-down information flows, which can enhance the interaction between different network levels. However, traditional predictive coding models only predict what is happening hierarchically rather than predicting the future. To address the problem, our model employs a multi-scale approach (Coarse to Fine), where the higher level neurons generate coarser predictions (lower resolution), while the lower level generate finer predictions (higher resolution). In terms of network architecture, we directly incorporate the encoder-decoder network within the LSTM module and share the final encoded high-level semantic information across different network levels. This enables comprehensive interaction between the current input and the historical states of LSTM compared with the traditional Encoder-LSTM-Decoder architecture, thus learning more believable temporal and spatial dependencies. Furthermore, to tackle the instability in adversarial training and mitigate the accumulation of prediction errors in long-term prediction, we propose several improvements to the training strategy. Our approach achieves good performance on datasets such as KTH, Moving MNIST and Caltech Pedestrian. 

The arxiv paper is available [here](https://arxiv.org/abs/2212.11642)

![image](Images/EDLSTM.png) 
 
![image](Images/KTH.png)

### Dependencies
* PyTorch, version 1.12.1 or above
* opencv, version 4.6.0 or above
* numpy, version 1.23.4 or above
* skimage, version 0.19.3 or above
* lpips

Please download and process the relevant datasets first. In order to save time, we process the video sequence into data of size (T, C, H, W) in advance, where T represents the length of the sequence, and C, H, W represent the dimension, height and width of the image respectively. We have provided examples for pre-processing of each dataset in this project. In addition, it is recommended to create new folders named "models" and "metric" in the local project to save the training model and evaluation results. Or, you can save it to other paths, but you need to modify the save path specified in the program.

### Model Implementation
* NetBlock.py, implementation of the basic encoder-decoder network
* ConvLSTM_Module.py, implementation of the encoder-decoder LSTM unit
* MSPN.py, implementation of the complete multi-scale predictive network

### Training and Testing
* pix_train.py, using only pixel-level loss (Euclidene distance) for training
* adv_train.py, using the improved adversarial training method proposed in this paper for training
* test.py, for testing

### Others
* Discriminator.py, the discriminator network constructed with conventional residual block
* utils.py, construction of PyTorch dataset and calculation of loss, etc















