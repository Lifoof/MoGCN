# MoGCN
## What is it?
MoGCN, a multi-omics integration method based on graph convolutional network.<br>
![Image text](https://github.com/Lifoof/MoGCN/blob/master/data/Fig1.png)
As shown in figure, inputs to the model are multi-omics expression matrices, including but not limited to genomics, transcriptomics, proteomics, etc. MoGCN exploits the GCN model to incorporate and extend two unsupervised multi-omics integration algorithms: Autoencoder algorithm (AE) based on expression matrix and similarity network fusion algorithm based on patient similarity network. Feature extraction is not necessary before AE and SNF. <br>

## Requirements 
MoGCN is a Python scirpt tool, your Python environment need:<br>
Python 3.6 or above <br>
Pytorch 1.4.0 or above <br>
snfpy 0.2.2 <br>

## Installing
```Python
python setup.py install
```

## Usage
The whole workflow is divided into three steps: <br>
* Use AE to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix <br>
* Use SNF to construct patient similarity network <br>
* Input multi-omics feature matrix  and the patient similarity network to GCN <br>
The sample data is in the data folder, which contains the CNV, mRNA and RPPA data of BRCA. <br>
### Command Line Tool
```Python
python AE_run.py -p1 data/fpkm_data.csv -p2 data/gistic_data.csv -p3 data/rppa_data.csv -m 0 -s 0 -d cpu
python SNF.py -p data/fpkm_data.csv data/gistic_data.csv data/rppa_data.csv -m sqeuclidean
python GCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20
```
The meaning of the parameters can be viewed through -h/--help <br>

### Data Format
*The input type of each omics data must be .csv, the rows represent samples, and the columns represent features (genes). In each expression matrix, the first column must be the samples, and the remaining columns are features. Samples in all omics data must be consistent. AE and SNF are unsupervised models and do not require sample labels.<br>
* GCN is a semi-supervised classification model, it requires sample label files (.csv format) during training. The first column of the label file is the sample name, the second column is the digitized sample label, the remaining columns are not necessary. <br>

## License
MIT License
