# CorresAttnTransformer
The official implementation of "Correspondence Attention Transformer: A Context-sensitive Network for Two-view Correspondence Learning"
### Test pretrained model

We provide the model trained on YFCC100M and SUN3D described in our paper. Run the test script to get results in our paper.
```bash
bash test.sh
```

### Test model on YFCC100M
```bash
cp -r ./log/main.py/train/* ./log/main.py/test/
python main.py --use_ransac=True --data_te='/data/yfcc-sift-2000-val.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/yfcc-sift-2000-val.hdf5' --run_mode='test'
mkdir ./log/main.py/test/known
mv ./log/main.py/test/*txt ./log/main.py/test/known

python main.py --use_ransac=True --data_te='/data/yfcc-sift-2000-test.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/yfcc-sift-2000-test.hdf5' --run_mode='test'
mkdir ./log/main.py/test/unknown
mv ./log/main.py/test/*txt ./log/main.py/test/unknown
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

### Test model on SUN3D
```bash
cp -r ./log/main.py/train/* ./log/main.py/test/
python main.py --use_ransac=True --data_te='/data/sun3d-sift-2000-val.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/sun3d-sift-2000-val.hdf5' --run_mode='test'
mkdir ./log/main.py/test/known
mv ./log/main.py/test/*txt ./log/main.py/test/known

python main.py --use_ransac=True --data_te='/data/sun3d-sift-2000-test.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/sun3d-sift-2000-test.hdf5' --run_mode='test'
mkdir ./log/main.py/test/unknown
mv ./log/main.py/test/*txt ./log/main.py/test/unknown
```


### Train model on YFCC100M

After generating dataset for YFCC100M/SUN3D, run the tranining script.
```bash
python main.py --run_mode= 'train'
```

You can train the fundamental estimation model by setting `--use_fundamental=True --geo_loss_margin=0.03` and use side information by setting `--use_ratio=2 --use_mutual=2`

