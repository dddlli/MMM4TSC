# MMM4TSC:Multimodal Mamba Time Series Classification of Fused Images
Multimodal Mamba Time Series Classification of Fused Images
## MMM4TSC architecture
![alt text](vision/archi_mmm4tsc.png)
## Usage of the code
```python
--model: to choose the models from [Encoder, CNN, MMM4TSC](default=MMM4TSC)
--data_path: root path of the data file
--sub_data : to choose the dataset from the UCR Archive (default=Coffee)
```
## Results
Results can be found in the results.csv file for FCN, ResNet, Inception, InceptionTime, ROCKET, LITE, LITETime and MMM4TSC. For non-ensemble methods, results are averaged over five runs, for ensemble methods, we ensemble the five runs.
### Average performance and Params comparison
The following figure shows the comparison between MMM4TSC and state of the art complex deep learners. The comparison consists on the average performance and the number FLOPS.
![alt text](vision/purppl.svg)
### MMM4TSC 1v1 with FCN, InceptionTime and ROCKET
The following compares MMM4TSC with FCN, InceptionTime and ROCKET using the accuracy performance on the test set of the 128 datasts of the UCR archive.

| ![alt text](pic/3M4TSC_FCN.svg) | ![alt text](pic/3M4TSC_InceptionTime.svg) |![alt text](pic/3M4TSC_ROCKET.svg)|
| --- | --- | --- |

### MMM4TSC MCM with SOTA
The following 1v1 and multi-comparison matrix shows the performance of MMM4TSC with respect to the SOTA models for Time Series Classification.
The following compares MMM4TSC with FCN, ResNet and Inception using the accuracy performance on the test set of the 128 datasts of the UCR archive.
![alt text](pic/mcm.png)

### CD Diagram
The following Critical Difference Diagram (CDD) shows the comparison following the average rank between classifiers.
![alt text](pic/cd-diagram.png)

## Requirements
```python
python=3.8
cuda=12.1
pytorch>=2.1.1
mamba_ssm
matplotlib
numpy
pandas
```
