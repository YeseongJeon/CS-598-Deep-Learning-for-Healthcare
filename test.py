#!/usr/bin/env python
# train.py
import argparse
import sklearn.metrics
import random

from models import CXRClassifier, CXRAdvClassifier
from cxrdataset import CheXpertDataset, MIMICDataset, NIHDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# Helper function to calculate area under ROC. From the original paper's code
def calc_pneumonia_auroc(classifier, testds):
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:, pneumonia_index]
    true_pneumonia = true[:, pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
        true_pneumonia,
        probs_pneumonia)
    return auroc

# Test standard Chest X-Ray classification model. From original paper's code.
def test_standard(chkpt_pth):
    testds_MIMIC = MIMICDataset(fold='test')
    # testds_CheXpert = CheXpertDataset(fold='test')
    testds_NIH = NIHDataset(fold='test')
    classifier = CXRClassifier()
    classifier.load_checkpoint(chkpt_pth)
    auroc_MIMIC = calc_pneumonia_auroc(classifier, testds_MIMIC)
    auroc_NIH = calc_pneumonia_auroc(classifier, testds_NIH)
    print("NIH area under ROC curve of pneumonia: {:.04f}".format(auroc_NIH))
    print("MIMIC area under ROC curve of pneumonia: {:.04f}".format(
        auroc_MIMIC))

# Test adversarial Chest X-Ray classification model. From original paper's code.
def test_adversarial(chkpt_pth):
    testds_MIMIC = MIMICDataset(fold='test')
    # testds_CheXpert = CheXpertDataset(fold='test')
    testds_NIH = NIHDataset(fold='test')
    classifier = CXRAdvClassifier()
    classifier.load_checkpoint(chkpt_pth)
    auroc_MIMIC = calc_pneumonia_auroc(classifier, testds_MIMIC)
    auroc_NIH = calc_pneumonia_auroc(classifier, testds_NIH)
    print("NIH area under ROC curve of pneumonia: {:.04f}".format(auroc_NIH))
    print("MIMIC area under ROC curve of pneumonia: {:.04f}".format(
        auroc_MIMIC))

# Test advanced adversarial Chest X-Ray classification model as our extension of the orginal paper.
def test_advanced_adversarial(chkpt_pth):
    """
    Testing function for the advanced adversarial model
    """
    testds_MIMIC = MIMICDataset(fold='test')
    testds_NIH = NIHDataset(fold='test')

    # Load CXRAdvClassifier with smart adversary
    classifier = CXRAdvClassifier(adv_model="smart")
    classifier.load_checkpoint(chkpt_pth)

    auroc_MIMIC = calc_pneumonia_auroc(classifier, testds_MIMIC)
    auroc_NIH = calc_pneumonia_auroc(classifier, testds_NIH)

    print("NIH area under ROC curve of pneumonia (Smart Adversary): {:.04f}".format(auroc_NIH))
    print("MIMIC area under ROC curve of pneumonia (Smart Adversary): {:.04f}".format(auroc_MIMIC))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', action="store")
    parser.add_argument('training', action="store", default='Standard')

    args = parser.parse_args()

    if args.training == 'Standard':
        test_standard(args.model_path)
    elif args.training == 'Adversarial':
        test_adversarial(args.model_path)
    # Included parameter to test advanced adversarial model from our extension.
    elif args.training == 'Advanced':
        test_advanced_adversarial(args.model_path)
    else:
        print('Training argument must be either "Standard" or "Adversarial"')


if __name__ == '__main__':
    main()
