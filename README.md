# DeepMHCII
DeepMHCII: A Novel Binding Core-Aware Deep Interaction Model for Accurate MHC II-peptide Binding Affinity Prediction

## Requirements
* python==3.9.7
* pytorch==1.10.2
* numpy==1.21.2
* scipy==1.7.3
* scikit-learn==1.0.2
* click==8.0.4
* ruamel.yaml==0.16.12
* tqdm==4.62.3
* logzero==1.7.0

## Experiments
```bash
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml # train and evaluation on independent test set.
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode 5cv # 5 cross-validation
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode lomo # leave one molecule out cross-validation
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode binding # binding core prediction (after model training)
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode seq2logo -a <allele> # seq2logo
```

## Declaration
It is free for non-commercial use. For commercial use, please contact 
Mr. Ronghi You and Prof. Shanfeng Zhu (zhusf@fudan.edu.cn).
