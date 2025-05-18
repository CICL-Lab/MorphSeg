## 1.Project Introduction

MorphSeg is a structure-guided model specifically designed for cerebrovascular segmentation, aiming to identify fine-grained and complex vascular structures.MorphSeg addresses these challenges by leveraging three core modules: Dynamic Feature Extraction, Local Direction-Adaptive Fusion, and Global Multi-Source Aggregation, significantly enhancing the model's ability to capture vascular morphology.

## 2.Table of Contents
- [Project Introduction](#project-introduction)
- [Model Architecture](#model-architecture)
- [Segmentation Effect Display](#segmentation-effect-display)
- [Datasets](#datasets)
- [Installation Guide](#installation-guide)
- [Run](#run)
- [Failure Cases](#failure-cases)
- [Training and Testing Logs](#training-and-testing-logs)

---
## 3.Model Architecture
![Model Structure](Model.png)

## 4.Segmentation Effect Display

To easily demonstrate the segmentation effect of **MorphSeg**, we provide an interactive comparison of slices.  
Slide the slider to view the segmented cerebrovascular structures.
> 🔗 **[Segmentation Demo](https://cicl-lab.github.io/MorphSeg/before_after.html)**

## 5.Datasets

MorphSeg is evaluated on two publicly available datasets:

5.1. **CereVessMRA**:
   - This dataset consists of 271 manually annotated 3D volumes, including both healthy and pathological samples.
   - A five-fold cross-validation strategy was applied, with 216 samples for training and 55 samples for testing.
   - Reference: Guo, B., Chen, Y., Lin, J., Huang, B., Bai, X., Guo, C., Gao, B., Gong, Q., Bai, X. (2024). *Self-supervised learning for accurately modelling hierarchical evolutionary patterns of cerebrovasculature*. Nature Communications, 15(1), 9235.

5.2. **COSTA**:
   - COSTA is a multi-center TOF-MRA dataset with a total of **355 samples**, specifically designed to validate segmentation models across different imaging settings, such as acquisition devices and scanning resolutions.
   - A five-fold cross-validation strategy was applied, with 284 samples for training and 71 samples for testing.
   - Reference: Mou, L., Yan, Q., Lin, J., Zhao, Y., Liu, Y., Ma, S., Zhang, J., Lv, W., Zhou, T., Frangi, A.F., et al. (2024). *COSTA: A multi-center TOF-MRA dataset and a style self-consistency network for cerebrovascular segmentation*. IEEE Transactions on Medical Imaging.


## 6.Installation Guide 

### 6.1. Operating System
We recommend running **MorphSeg** on a **Linux system** for optimal performance and compatibility.

### 6.2. Hardware Requirements
We suggest using **RTX 4090 (24GB)** for training. Of course, higher computing power allows for better configurations, resulting in improved model performance.

### 6.3. Recommended Configuration
- **PyTorch**: 2.1
- **CUDA**: 11.8

### 6.4Installation Steps
#### 6.4.1Clone the MorphSeg repository:
```bash
git clone https://github.com/CICL-Lab/MorphSeg
```

#### 6.4.2Install `dynamic_network_architectures`:
For the acquisition and installation of **dynamic_network_architectures**, please refer to the [dynamic_network_architectures](https://github.com/CICL-Lab/dynamic_network_architectures/tree/main).

#### 6.4.3Install MorphSeg:
```bash
cd path/MorphSeg
pip install -e .
```

### 6.5. Data Preparation
#### 6.5.1Data Structure
Please refer to [Example Data](https://github.com/CICL-Lab/MorphSeg/tree/main/DataSample/Dataset001_Cerebrovascular) to configure the file structure and set up the `dataset.json` file. The folder should be named as `Dataset[Taskid]_Cerebrovascular`.

The dataset should be organized as follows:
```bash
   Dataset001_Cerebrovascular/
   ├── imagesTr
   ├── labelsTr
   ├── imagesTs
   └── dataset.json
```
   
```json
Below is an example of the required dataset.json structure:
{
  "channel_names": {
    "0": "MRA"
  },
  "labels": {
    "background": 0,
    "Cerebrovascular": 1
  },
  "numTraining": 271,
  "file_ending": ".nii.gz"
}
```
#### 6.5.2Environment Variables Setup
Refer to [nnUNetv2 Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) for more details.
```bash
export nnUNet_raw="/root/autodl-tmp/MorphSeg_raw"  
export nnUNet_preprocessed="/root/autodl-tmp/MorphSeg_preprocessed"
export nnUNet_results="/root/autodl-tmp/MorphSeg_results"

# Fix multi-threading issues:
export TORCH_COMPILE_DISABLE=1
```
#### 6.5.3Experiment Configuration and Preprocessing  
To perform experimental preprocessing and configuration:
```bash
MorphSeg_plan_and_preprocess -d [Taskid] --verify_dataset_integrity

# Example:
MorphSeg_plan_and_preprocess -d 1 --verify_dataset_integrity
```
### 6.6. Hyperparameter Settings
We recommend the following hyperparameters:
- **batchsize**: 2 (line in 185 at Path/MorphSeg_preprocessed/Dataxxx_xxx/nnUNetPlans.json) 
- **epoch**: 1500 (line in 152 at Path/MorphSeg/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) 

## 7.Run
To begin model training, execute the following command:

```bash
MorphSeg_train [Taskid] 3d_fullres [fold]
```

Example:
```bash
MorphSeg_train 1 3d_fullres 5
```
## 8.Failure Cases

Although we have demonstrated the effectiveness of **MorphSeg** through extensive experiments, fine-grained cerebrovascular segmentation remains a challenging task.  
Here, we present some **failure cases**, where **blue** and **green** represent **False Negative (FN)** and **False Positive (FP)** regions, respectively.  It can be observed that MorphSeg may produce some False Negative cases during segmentation.
Optimizing these regions will be a major focus of our future work.
<div align="center">
   <img src="https://github.com/CICL-Lab/MorphSeg/blob/main/Failure_case.png" alt="Failure Cases" width="80%"/>
</div>

## 9.Logs and results
The training records and testing results are located in Path/MorphSeg_results/Dataxxx_xxx.
