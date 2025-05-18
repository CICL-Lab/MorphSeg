## Project Introduction

MorphSeg is a structure-guided model specifically designed for cerebrovascular segmentation, aiming to identify fine-grained and complex vascular structures.MorphSeg addresses these challenges by leveraging three core modules: Dynamic Feature Extraction, Local Direction-Adaptive Fusion, and Global Multi-Source Aggregation, significantly enhancing the model's ability to capture vascular morphology.

## Datasets

MorphSeg is evaluated on two publicly available datasets:
1. **CereVessMRA**:
   - This dataset consists of 271 manually annotated 3D volumes , including both healthy and pathological samples.
   - A five-fold cross-validation strategy was applied, with 216 samples for training and 55 samples for testing.
   - Reference: Guo, B., Chen, Y., Lin, J., Huang, B., Bai, X., Guo, C., Gao, B., Gong, Q., Bai, X. (2024). *Self-supervised learning for accurately modelling hierarchical evolutionary patterns of cerebrovasculature*. Nature Communications, 15(1), 9235.

2. **COSTA**:
   - COSTA is a multi-center TOF-MRA dataset  with a total of **355 samples*,specifically designed to validate segmentation models across different imaging settings, such as acquisition devices and scanning resolutions
   - A five-fold cross-validation strategy was applied, with 284 samples for training and 71 samples for testing.
   - Reference: Mou, L., Yan, Q., Lin, J., Zhao, Y., Liu, Y., Ma, S., Zhang, J., Lv, W., Zhou, T., Frangi, A.F., et al. (2024). *COSTA: A multi-center TOF-MRA dataset and a style self-consistency network for cerebrovascular segmentation*. IEEE Transactions on Medical Imaging.
