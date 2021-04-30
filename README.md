# STN

Heterogeneous Domain Adaptation via Soft Transfer Network, ACM MM 2019

# Codes

You can download the codes of STN in:

Link: https://pan.baidu.com/s/1AHsrP4OwOLYWl1rcjbsQVg

Password: qr91

# Step-by-step Installation

```
conda create -n cwan python=3.6
conda activate cwan

pip install tensorflow-gpu==1.4
conda install cudatoolkit=8.0
conda install cudnn=6.0
conda install scipy
conda install matplotlib
conda install scikit-learn
```

# Running 
1. You can run this code by inputing: python main.py. The results should be close to 93.03 (C->A) and 93.11 (W->A), respectively. Note that different environmental outputs may be different.

2. You can use your datasets by replacing: 

   ```
   source_exp = [ad.SCS, ad.SWS]  % source domain data
   target_exp = [ad.TAD, ad.TAD]  % target domain data
   ```

3. You can tune the parameters, i.e., beta, lr, T, d, tau, for different applications.

4. The default parameters are: beta = 0.001, lr = 0.001, T = 300, d = 256, tau = 0.001.

# Citation
If you find it is helpful, please cite:
```
@InProceedings{Yao-2019,
  author    = {Yuan Yao and Yu Zhang and Xutao Li and Yunming Ye},
  title     = {Heterogeneous Domain Adaptation via Soft Transfer Network},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year      = {2019},
  pages     = {1578--1586},
}
```
