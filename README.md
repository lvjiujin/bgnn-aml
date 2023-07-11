This code is for the paper [Bitcoin Anti-Money Laundering Based on BGNN Graph Neural Network](https://mp.weixin.qq.com/s/6dw7m6DSqsl1C2jVLUVhpQ)

The BGNN Graph Neural Network  algorithm is used in the project  which is from the ICLR 2021 paper: [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://openreview.net/pdf?id=ebS5NUfoMKL)

The elliptic dataset is from the [kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).

This code contains implementation of the following models for graphs based on the bgnn model: 
* **BGNN** (end-to-end {CatBoost + {ARMA, hGANet, DAGNN}})



#####  First,  you should download the repository and install some necessary packages according to the requirements, you can use the following command:

 pip install -r requirements.txt

#####  Second, you can run the code with the following scripts

```
1) first you should process the original dataset, generate the graph dataset, you can execute the following code:

python scripts/preprocessing_elliptic.py


2) train the test the model.
nohup python scripts/run.py datasets/elliptic  bgnn --max_seeds 5 \
--repeat_exp 5 --task classification  \
--metric f1  --save_folder ./results/elliptic_bgnn \
--version_num 1.0 2>&1 > log_test_202109.txt  &
```

