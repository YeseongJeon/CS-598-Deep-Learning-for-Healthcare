#!/usr/bin/env python
# train.py
import argparse
import sklearn.metrics
import random

from models import CXRClassifier, CXRAdvClassifier
from cxrdataset import MIMICDataset, NIHDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Training Functions for the Models:
This file leverages functions used from the original repository in addition to updates and new methods we implemented.

"""

# Helper function to return the index. From the original paper's code
def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))

# Train the standard Chest X-Ray classification model. From the original paper's code
def _train_standard(datasetclass, checkpoint_path, logpath):
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')

    classifier = CXRClassifier()
    classifier.train(trainds,
                     valds,
                     max_epochs=100,
                     lr=0.01,
                     weight_decay=1e-4,
                     logpath=logpath,
                     checkpoint_path=checkpoint_path,
                     verbose=True)
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:, pneumonia_index]
    true_pneumonia = true[:, pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
        true_pneumonia,
        probs_pneumonia)
    print("area under ROC curve of pneumonia: {:.04f}".format(auroc))

# Train the adversarial Chest X-Ray classification model. From the original paper's code
def _train_adversarial(datasetclass, checkpoint_path, logpath):
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')

    classifier = CXRAdvClassifier()
    classifier.train(trainds,
                     valds,
                     lr=0.01,
                     weight_decay=1e-4,
                     logpath=logpath,
                     checkpoint_path=checkpoint_path,
                     verbose=True)
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:, pneumonia_index]
    true_pneumonia = true[:, pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
        true_pneumonia,
        probs_pneumonia)
    print("area under ROC curve of pneumonia: {:.04f}".format(auroc))

# Train the advanced adversarial model based on our extension of the original paper. 
def _train_advanced_adversarial(datasetclass, checkpoint_path, logpath):
    """
    Extension of standard adversarial model from the paper to use a more robust adversarial model.
    """
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')

    # Initialize CXRAdvClassifier with adv_model='smart'
    classifier = CXRAdvClassifier(adv_model="smart")
    
    classifier.train(trainds,
                     valds,
                     lr=0.01,
                     weight_decay=1e-4,
                     logpath=logpath,
                     checkpoint_path=checkpoint_path,
                     verbose=True)
    
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:, pneumonia_index]
    true_pneumonia = true[:, pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
        true_pneumonia,
        probs_pneumonia)
    
    print("Area under ROC curve of pneumonia (Smart Adversary): {:.04f}".format(auroc))


def main():
    print("running train")
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action="store", default='MIMIC')
    parser.add_argument('training', action="store", default='Standard')

    args = parser.parse_args()

    if args.dataset == 'MIMIC' and args.training == 'Standard':
        _train_standard(MIMICDataset, 'mimic_standard_model.pkl',
                        'mimic_standard.log')
    elif args.dataset == 'NIH' and args.training == 'Standard':
        _train_standard(NIHDataset, 'NIH_standard_model.pkl',
                        'NIH_standard.log')
    elif args.dataset == 'NIH' and args.training == 'Adversarial':
        _train_adversarial(
            NIHDataset, 'NIH_adversarial_model.pkl', 'NIH_adversarial.log')
    elif args.dataset == 'MIMIC' and args.training == 'Adversarial':
        _train_adversarial(
            MIMICDataset, 'mimic_adversarial_model.pkl', 'mimic_adversarial.log')
    # We included parameters to test the Advanced adversarial model based on our extension proposal
    elif args.dataset == 'MIMIC' and args.training == 'Advanced':
        _train_advanced_adversarial(MIMICDataset, 'mimic_advSMART_model.pkl', 'mimic_advSMART.log')
    elif args.dataset == 'NIH' and args.training == 'Advanced':
        _train_advanced_adversarial(NIHDataset, 'NIH_advSMART_model.pkl', 'NIH_advSMART.log')
    else:
        print('arguments not understood')


if __name__ == '__main__':
    main()
