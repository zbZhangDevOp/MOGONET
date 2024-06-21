# MOGONET: Multi-omics Integration via Graph Convolutional Networks for Biomedical Data Classification

Tongxin Wang\*, Wei Shao\*, Zhi Huang, Haixu Tang, Jie Zhang, Zhengming Ding, and Kun Huang

MOGONET (Multi-Omics Graph cOnvolutional NETworks) is a novel multi-omics data integrative analysis framework for classification tasks in biomedical applications.

![MOGONET](https://github.com/txWang/MOGONET/blob/master/MOGONET.png?raw=true 'MOGONET')
Overview of MOGONET. \
<sup>Illustration of MOGONET. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. For clear and concise illustration, an example of one sample is chosen to demonstrate the VCDN component for multi-omics integration. Pre-processing is first performed on each omics data type to remove noise and redundant features. Each omics-specific GCN is trained to perform class prediction using omics features and the corresponding sample similarity network generated from the omics data. The cross-omics discovery tensor is calculated from the initial predictions of omics-specific GCNs and forwarded to VCDN for final prediction. MOGONET is an end-to-end model and all networks are trained jointly.<sup>

## Files

_main_mogonet.py_: Examples of MOGONET for classification tasks\
_main_biomarker.py_: Examples for identifying biomarkers\
_models.py_: MOGONET model\
_train_test.py_: Training and testing functions\
_feat_importance.py_: Feature importance functions\
_utils.py_: Supporting functions

\* Equal contribution

## Detailed Explanation

### Overview

The provided codebase consists of scripts and functions designed for training models to identify biomarkers and work with multi-omics data using machine learning and deep learning approaches.

### Key Components

1. **Feature Importance Calculation**:

   - `feat_importance.py`: Contains functions to calculate and normalize feature importance from a trained model.

2. **Biomarker Model Training**:

   - `main_biomarker.py`: Loads data, trains a RandomForestClassifier model, and calculates feature importance.

3. **Multi-Omics Data Integration**:

   - `main_mogonet.py`: Loads multi-omics data, trains a MoGONet model, and evaluates the model.

4. **Model Definitions**:

   - `models.py`: Defines the MoGONet model, a neural network that integrates two omics datasets for training and prediction.

5. **Data Splitting**:

   - `train_test.py`: Provides a utility function to split data into training and testing sets.

6. **Utility Functions**:
   - `utils.py`: Contains functions for loading multi-omics data and evaluating the model.

### Usage

1. **Training a Biomarker Model**:

   - Run `main_biomarker.py` to train a RandomForest model on the data.
   - This script loads the data, splits it into training and testing sets, trains the model, calculates feature importance, and prints normalized feature importance values.

2. **Training a MoGONet Model**:

   - Run `main_mogonet.py` to train the MoGONet model on multi-omics data.
   - This script loads multi-omics data, splits it into training and testing sets, trains the MoGONet model, and evaluates the model's performance.

3. **Calculating Feature Importance**:
   - The `calculate_feature_importance` function in `feat_importance.py` extracts feature importance from a trained model.
   - The `normalize_importance` function normalizes the importance values to make them comparable.

### Detailed Steps for Training

1. **Data Preparation**:

   - Replace placeholder data loading functions (`load_data`, `load_multiomics_data`) with actual data loading logic to use your specific datasets.

2. **Model Training**:

   - For the biomarker model, the `train_biomarker_model` function in `main_biomarker.py` trains a RandomForestClassifier on the provided data.
   - For the MoGONet model, the `fit` method in the `MoGONet` class trains the model on multi-omics data.

3. **Model Evaluation**:
   - The `evaluate_model` function in `utils.py
