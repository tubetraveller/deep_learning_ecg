# Deep Learning for ECG Analysis - Residual CNNs & Interpretability

## Project Overview
This project focuses on heartbeat classification using deep learning techniques. The approach emphasizes **interpretability** by preprocessing the ECG signals such that individual phases of the cardiac cycle (e.g., P, QRS, T waves) align across signals. This preprocessing step ensures that the averaged signals can be meaningfully compared and interpreted in terms of physiological phases.

Additionally, two distinct preprocessing methods are implemented to shift and align the ECG signals based on specific features of the heart cycle. The resulting models are analyzed using **SHAP (SHapley Additive exPlanations)** to provide insights into the contribution of individual time points to the classification decision.

## Goal of the Project
The main objective is to develop an interpretable heartbeat classification model that not only achieves high classification performance but also provides insights into the decision-making process. By aligning ECG signals to highlight specific cardiac phases, the project aims to bridge the gap between accuracy and interpretability, ultimately supporting medical diagnosis.

## Dataset Information
The project uses the **MIT-BIH Arrhythmia Dataset** and the **PTB Diagnostic ECG Database**, both well etablished datasets in medical research and for ECG signal classification.

### 1. MIT-BIH Arrhythmia Dataset:
- **Number of Samples**: 109,446
- **Number of Classes**: 5
- **Sampling Frequency**: 125Hz
- **Classes**:
  - Class 0: Normal (N)
  - Class 1: Supraventricular ectopic beat (S)
  - Class 2: Ventricular ectopic beat (V)
  - Class 3: Fusion beat (F)
  - Class 4: Unclassifiable (Q)
- **Source**: Physionet's MIT-BIH Arrhythmia Dataset

### 2. PTB Diagnostic ECG Database
- **Number of Samples**: 14,552  
- **Number of Classes**: 2  
- **Sampling Frequency**: 125 Hz  
- **Classes**:  
  - Class 0: Healthy control (N)  
  - Class 1: Myocardial infarction (MI)  
- **Source**: Physionet’s PTB Diagnostic Database  

The datasets are preprocessed to ensure consistent signal length by padding or cropping all ECG signals to a fixed length of 188.

For this specific project, I will mainly focus on the **MIT-BIH Arrhythmia** Dataset.

## Preprocessing Methods
Two different preprocessing methods are introduced. Both align the **R-wave (maximum amplitude)** of the ECG signals to a fixed position across all samples. The signals are shifted either left or right without altering their values. Specific rules ensure that truncated values are appropriately reinserted, maintaining the integrity of the original signal.
Both methods aim to standardize the signal alignment, allowing meaningful interpretation of the averaged ECG signals across classes.

## Model Architecture
The classification model employs a **Residual 1D-CNN** architecture, where stacked residual blocks enable effective extraction of both spatial and temporal features from the preprocessed ECG signals. Hyperparameter optimization was carried out experimentally using Keras Tuner to fine-tune model parameters specifically for our dataset.

### Key Features of the Model:
- **Residual Blocks**: Enhance feature extraction and allow deeper architectures without degradation.
- **Conv1D Layers**: Capture local patterns in the ECG signals.
- **Dropout Regularization**: Mitigates overfitting.
- **Softmax Output**: Provides probabilities for multi-class classification.

## Interpretability with SHAP
To assess the interpretability of the model, **SHAP (SHapley Additive exPlanations)** values are computed for test samples. SHAP values quantify the contribution of each time step to the model's predictions, highlighting which phases of the cardiac cycle are most influential for the classification.

### SHAP Analysis Workflow:
1. **Background Sample Selection**: A subset of training data is used as the background for SHAP computation.
2. **Test Sample Analysis**: SHAP values are computed for a subset of test samples.
3. **Visualization**: Mean SHAP values are overlaid with averaged ECG signals to provide insights into the most critical features for classification.

## Results
The preliminary results indicate that aligning ECG signals to specific cardiac phases improves interpretability without compromising accuracy. Key evaluation metrics include:
- **F1-Score**: High classification performance across all classes.
- **SHAP Analysis**: Highlights the phases of the cardiac cycle most relevant for distinguishing between classes.

## How to Run the Project
To run the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tubetraveller/deep_learning_ekg
   cd deep_learning_ekg
   ```
2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Data**:  
   Download the ECG dataset from Kaggle and place the CSV files into your local `notebooks/data` directory (create it if it doesn’t exist).  
   - Dataset link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data?select=ptbdb_normal.csv  
   - Example (manual):  
     1. Go to the link above and download `ptbdb_normal.csv`, `ptbdb_abnormal.csv` etc.  
     2. Move them into `notebooks/data/`:
        ```bash
        mkdir -p notebooks/data
        mv ~/Downloads/ptbdb_*.csv notebooks/data/
        ```
## Notebooks
The `notebooks` directory contains two Jupyter notebooks:  
- **`01_exploratory_data_analysis.ipynb`**: Performs comprehensive exploratory data analysis on the ECG signals, including visualizations, summary statistics, and initial data insights.  
- **`02_model_and_preprocessing.ipynb`**: Implements the full preprocessing pipeline and the Residual 1D-CNN model training workflow, from data alignment to hyperparameter tuning.

## Conclusion
This project demonstrates the potential of combining data preprocessing techniques with explainable AI methods to enhance the interpretability of deep learning models for medical applications. By aligning ECG signals to cardiac phases and analyzing model predictions with SHAP, the approach provides both high performance and actionable insights for physicians.

## Personal Contribution
This repository presents an extract from a broader team project, highlighting my individual contributions to the overall effort. In particular, I designed and implemented the ECG preprocessing pipeline for phase alignment, developed the Residual 1D-CNN model architecture, and led the interpretability analysis using SHAP to uncover the most influential cardiac-cycle features.

## License
This project is licensed under the MIT License. 