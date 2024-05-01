# Classifying Movie Reviews

## Content
1. Introduction
2. Repo Structure
3. Instructions
4. Transfer Learning Models
5. Comparing Results
6. Conclusions
7. References

## 1. Introduction
This repo uses machine learning and deep learning models to predict the sentiment of movie reviews as being 'Postive' or 'Negative'. The following sections detail how to use the code for new data, as well as showing how the selected models have performed on the code they were trained on.

## 2. Repo Structure
```text
├── code
|  └── front_end.ipynb
|  └── parent_dir.py　　
|  └── model_loader.py　　　
|  └── preprocess_text.py　　　　
|  └── predict_text.py
|  └── display_results.py
|  └── requirements.txt
|  └── README.md
|  └── models
|    └── model_conv.h5
|    └── model_glove.h5
|    └── model_bert.h5
|    └── README.md
├── data
|  └── test_sample.csv
|  └── README.md
├── assignment3_Part1_3_hs3458_kem2231.ipynb
├── assignment3_Part2_3_hs3458_kem2231.ipynb
├── README.md
```
## 3. Instructions

Clone the repository locally. Upload any new text data to predict on to "data" folder. In the "code" folder, open front_end.ipynb. Uncomment and run the first line in order to install required packages. In the indicated space, input the model on which you want to run the inference pipeline on your data. The model choices are as follows;

model_conv: CNN model with Conv1D layers   
model_glove: Transfer learning model with GloVe embeddings   
model_bert: BERT model   

(Model training results are summarized further below.)

## 4. Transfer Learning Models
...

|Model|Summary Plot|
|:-:|:-:|
|ResNet50|![resnet](./visuals/cnn1.png)
|InceptionV3|![inv](./visuals/cnn2.png)

## 5. Comparing Results
|Model|Accuracy|Loss|Precision|Recall|
|:-:|:-:|:-:|:-:|:-:|
|CNN-1 |  |  |   | |
|CNN-2 |  |  |   | |
|CNN-3 |  |  |   | |
|INCEPTIONV3 |  |  |   | |
|RESNET50 |  |  |   | |

...

## 6. Conclusions
...

## 7. References
