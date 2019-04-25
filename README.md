# Ultra-resolving face images by discriminative generative networks (URDGN)  <br><br>
ECCV 2016 <br><br>
Face Hallucination  <br><br>

Run this code with more than 8G of GPU
-------   
The source code is provided by https://github.com/XinYuANU/URDGN. Thanks for their works.

```
@inproceedings{Xin2016Ultra,
  title={Ultra-Resolving Face Images by Discriminative Generative Networks},
  author={Xin, Yu and Porikli, Fatih},
  booktitle={European Conference on Computer Vision},
  year={2016},
}
```
-------  
### 1. move all images (train and test) to "datasets" and creat .h5 file
### python create_YTC_xin.py
### If you want to use your datasets, try to modify the code as follows.
#52 x = create_YTC(pathfolder, 18) the size of low-resolution images <br><br>
#61 x = create_YTC(pathfolder, 144) the size of high-resolution images <br><br>
  
### 2.train URDGN
### th train_ytc_xin_128_D.lua
just change the code of train_ytc_xin_128_D.lua <br><br>
#39 --scale            (default 144) the size of high-resolution images <br><br>
#48 ntrain = 1600 the num of train images <br><br>
#49 nval = 40 the num of test images <br><br>
#112 model_D:add(nn.Reshape(9 *9 *96)) --9 = 144 / (2 ^ 4) <br><br>
#113 model_D:add(nn.Linear(9 *9 *96, 1024)) <br><br>
#198 local noise_inputs = torch.Tensor(N, 3, 18, 18)  <br><br>
#204 noise_inputs[{{i}}] = image.scale(torch.squeeze(noise_input_high[{{idx}}]),18,18)  <br><br>
#224 local to_plot = getSamples(valData_HR, 40) the num of test images <br><br>
#232 local formatted = image.toDisplayTensor({input=to_plot, nrow=10})  <br><br>
#239 IDX = torch.randperm(1600) the num of train images <br><br>

just change the code of adverserial_xin_v1_D.lua <br><br>
#135 local LR_inputs = torch.Tensor(opt.batchSize, 3, 18, 18)  the size of low-resolution images <br><br>
#243 local sample = torch.Tensor(dataBatchSize, 3, 18, 18)  the size of low-resolution images <br><br>
#348 local inputs_lr = torch.Tensor(opt.batchSize, 3, 18, 18)  the size of low-resolution images <br><br>


### 3.This code can split the result and save them as images in "sr".
### run getimg.m





