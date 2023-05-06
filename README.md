
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















