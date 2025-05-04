# CS598 Deep Learning for Healthcare Final Project

Final Project repository for Taoran Shen and Yeseong Jeon.

This contains our implementation/replication of the results from the paper: "An Adversarial Approach for the Robust Classification of Pneumonia from Chest Radiographs" (Joseph D. Janizek, Gabriel Erion, Alex J. DeGrave, Su-In Lee, 2020) 

The original repository can be found here: https://github.com/suinleelab/cxr_adv/tree/master 
The paper is published here: arxiv.org/abs/2001.04051

## File Contents

Please refer to the cxrdataset.py, models.py, test.py, and train.py files to see our adoption and implementation of the original paper's code to replicate their results.

We also included the data_label_configuration.ipynb file where we took the data labels that came from the original MIMIC and NIH datasets and transformed them to work with our dataset classes.

The saved .pkl model files and the training logs can be found in the saved_model_files and training_logs folders.

Packages can be found in requirements.txt

## Usage:

## Data Folder Configuration
Before running, you need configure the `./data` directory. You will need to download the [NIH](https://www.kaggle.com/datasets/nih-chest-xrays/data?select=Data_Entry_2017.csv) and [MIMIC](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) datasets and organize the file structure like this:

```
data/
├── MIMIC/
│   ├── p10xxxx/
│   │   └── s504142676.../
│   │       └── image.jpg
│   ├── p10xxx1.../
│   ├── mimic_train.csv
│   └── mimic_valid.csv
├── NIH/
│   ├── images_001/
│   │   ├── 000000001_001.png
│   │   └── 000000001_002.png
│   ├── nih_train.csv
│   └── nih_valid.csv
```

You can refer to the file struture from the [data folder](https://drive.google.com/drive/folders/1lSzCNw1UQcOKfqKB1G3OSVz7dsmmWkPF?usp=sharing) in our Google Drive, where you can download the training/testing data and reference the directory structure.

### Training a model | Command line interface
The instructions to run the command line interface is based on the original paper's implementation except we modified some of the keywords based on our datasets and our extension.

To train the models, run `python train.py dataset training` from terminal. The 'dataset' argume allows user to specify the dataset to use for training should be either ('MIMIC' or 'NIH'). The argument 'training' indicates whether to follow the standard training procedure or to train the adversarial view-invariant model. This argument can be either 'Standard', 'Adversarial', or 'Advanced'. 

### Testing a model | Command line interface

The instructions to run the command line interface is based on the original paper's implementation except we modified some of the keywords based on our datasets and our extension.

To test a model, simply run `python test.py model_path training`, where 'model_path' is the path to the saved model you would like to test, and 'training' specifies whether the model was trained as a 'Standard', 'Adversarial' or 'Advanced' model.
