# CorresAttnTransformer Implementation
The official implementation of [Correspondence Attention Transformer: A Context-sensitive Network for Two-view Correspondence Learning](https://ieeexplore.ieee.org/document/9741369) by Jiayi Ma, Yang Wang, Aoxiang Fan, Guobao Xiao, and Riqing Chen.

Highlights:

1) MLP-based deep architecture and Self-Attention mechanism for Two-view Correspondence Learning;

2) Spatial MultiHead Attention structure to exploit the geometrical context from different aspects; 

3) Second-order Covariance Normalized Channel Attention with less GPU memory.
     
If our work is inspired for you, welcome to follow and cite it!
```bash
@ARTICLE{9741369,
  author={Ma, Jiayi and Wang, Yang and Fan, Aoxiang and Xiao, Guobao and Chen, Riqing},
  journal={IEEE Transactions on Multimedia}, 
  title={Correspondence Attention Transformer: A Context-sensitive Network for Two-view Correspondence Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3162115}}
```


# Requirements
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.

# Run the code
### Test pretrained model

We provide the model trained on YFCC100M and SUN3D described in our paper. Run the test script to get results in our paper.
```bash
bash test.sh
```

### Test model on YFCC100M
```bash
python main.py --use_ransac=False --data_te='/data/yfcc-sift-2000-test.hdf5' --run_mode='test'
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

### Test model on SUN3D
```bash
python main.py --use_ransac=False --data_te='/data/sun3d-sift-2000-test.hdf5' --run_mode='test'
```


### Train model on YFCC100M

After generating dataset for YFCC100M/SUN3D, run the tranining script.
```bash
python main.py --run_mode= 'train'
```

You can train the fundamental estimation model by setting `--use_fundamental=True --geo_loss_margin=0.03` and use side information by setting `--use_ratio=2 --use_mutual=2`

### Acknowledgement
This code is heavily borrowed from [zjhthu/OANet](https://github.com/zjhthu/OANet). If you use the part of code related to data generation, testing and evaluation, you should cite this paper and follow its license.
```bash
@article{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
