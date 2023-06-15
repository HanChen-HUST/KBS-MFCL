# MFDA 
  Source code for 2023 CIKM under review paper Multi-modal Fake News Detection with Data Augmentation
# Dataset
  The datasets used in the experiments were based on the two publicly available Weibo and PHEME datasets released by Zheng et al. (2023) and Zubiaga et al. (2017).
  Preprocessed datasets are available at: Weibo https://www.dropbox.com/scl/fi/o7mhk0chqeo21pugequh2/Chinese.rar?dl=0&rlkey=zba9vldtuu3np7olct2dbzzb7 and PHEME https://www.dropbox.com/scl/fi/3oh12ur58a8d62l5vm5cb/PHEME.rar?dl=0&rlkey=jwq68ru9l10nbphcg5z8pnf1z
# Dependencies
  Our code runs with the following packages installed:
  ```
    python 3.9.0
    torch 1.13.1
	cuda 11.6.1
	pytorch-cluster  1.6.0             
	pytorch-scatter  2.1.0          
	pytorch-sparse  0.6.15
	torchvision  0.14.1	
	numpy  1.23.5
	yaml  0.2.5
	pandas  1.5.2
    scikit-learn 1.0.2
 ```
 # Run
 Train and test
 ```
 python ./Weibo/test.py -c ./Weibo/configs/weibo.yaml
 ```
