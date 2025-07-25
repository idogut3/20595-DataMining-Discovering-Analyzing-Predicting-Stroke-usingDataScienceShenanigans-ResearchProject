# Discovering analyzing and predicting Stroke using DataMining techniques
![bioinformatics](https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images/bioinformatics.jpg)
## A Little about the project ⋆｡˚ ☁︎ ˚｡⋆｡˚☽˚｡⋆ 
This project was completed as part of my university DataMining course.🎓
It is a project regarding stroke prediction; 
It's goal is to find as many "early signs" for stroke, helping doctors / nurses or other health professionals to identify risks, prevent and treat patients with care based on real data.

In a lot of ways its much more of a research project foucusing on the data, diving into medical proven facts and using those facts and known factors to test, verify, and evaluate our models, than a "regular" data-science project
with the ci/cd mindset (plan -> code -> build -> test -> release -> deploy -> operate).

> [!NOTE]
> The Link for the dataset I used: [kaggle dataset link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data).

> [!NOTE]
> The project incorporates AI/ML/Deep Learning techniques used alongside some statistical analysis based models and label classification methods. 
> I encourage you to scroll on to dive deeper into the project.



> [!TIP]
> Explanations about the importance of stroke prediction and comparison of different methods can be read in the following articles published in the journal Nature in 2024 at the following addresses:
>
> [Predictive modelling and identification of key risk factors for stroke using machine learning](https://www.nature.com/articles/s41598-024-61665-4)
> 
> [Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models](https://www.nature.com/articles/s41598-024-82931-5)

---

## The main data mining goals of the project 🧪🗃️🎯
### 1. **Accurate prediction of stroke risk:** 
  - Developing a predictive model that can identify various phenomena and trends that "signal" a chance of experiencing a stroke.
  Maximizing the model's precision, accuracy, F1 score while minimizing false positives and false negatives.

### 2. **Discovering the most influential characteristics and factors:**
  - Discovering which characteristics (e.g: age, hypertension, heart disease, BMI, and glucose levels) are most influential on the chance for stroke in order to interpret model behavior and gain medical insights.

### 3. **Handling imperfect data:**
  - Develop effective strategies or techniques for managing missing values, noisy data and information imbalance
  using SMOTE/ median fill / average fill and more...

### 4. **Model Evaluation:**
  - Goal: To compare several prediction models, including AI-based models
  (such as KNN, SVM, RANDOM FOREST, logistic regression, XGBoost) to determine the best fit.

### 5. **Generalization and prevention of overfitting:**
  - Goal: To ensure that the model can correctly generalize to unseen patient data.

---

### Our dataset:

  - **Id** - is a unique integer identifier assigned to each patient. It ranges from 67 to 72,940 and does not contain missing values.
  - **Gender** - is a categorical attribute indicating the patient's gender. It can take one of three values: "male", "female" or "other", with no missing values.
  - **Age** - is a continuous numeric variable ( float ) representing the patient's age. The values range from 0.08 years to a maximum age of 82 years, and there are no missing values in this field.

### A diagram I drew of the age distribution.

<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images/age-distribution.png" style="width: 50%;" />


  - **hypertension** - is a binary variable where a value of 1 indicates that the patient has hypertension, and 0 indicates that they do not. This attribute contains complete data with no missing values.
  - **disease_heart** - is another binary variable where 1 represents the presence of heart disease and 0 represents its absence. Like the previous field, it does not contain missing data.
  - **married_ever** - is a categorical variable indicating whether the patient has ever been married. It has two possible values: "yes" and "no", and has no missing values.
  - **type_work** - classifies the type of employment of the patient. It can be one of the following: "children", "government job", "never worked", "private", or "self-employed". This field is complete with no missing data.
  - **type_Residence** - identifies whether the patient lives in a "rural" or "urban" area. This categorical variable is fully populated.
  - **avg_glucose_level** - is a float representing the patient's average blood glucose level. It ranges from 55.12 to 271.74 and contains no missing values. bmi - or body mass index, is a continuous numeric attribute ranging from 10.3 to .97.6. However, this field contains 201 missing values, which need to be handled during preprocessing. 

  - **status_smoking** - is a categorical attribute with four possible values: "never smoked", "previously smoked", "smoker", and "unknown". Although it does not contain missing values in the traditional sense, "unknown" is used as a placeholder for data that is not available.
  - **stroke** - is the target binary variable of the dataset. A value of 1 indicates that the patient has had a stroke, and 0 indicates that he has not. There are no missing values in this field.

> [!IMPORTANT]
> #### Main assumptions and simplifications in the project:
> Since we are dealing with medical data that relates to predictions of whether or not a patient will have a stroke (a life or death scenario), the main assumption I use throughout the project is that **_reducing_ false negatives** is the most
> important metric to achieve (since this situation means identifying a person who is going to have a stroke as a person who will not have a stroke, which is a very big risk to their life (false identification).

### Missing Values:
  - **Bmi values** - Approximately 3.9% of records are missing - Which is 201 rows that are missing.
  - **Smoking_status values** - a couple of unknown values.
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images/Comparison%20of%20Valid%20vs%20Corrupted-Unknown%20Data.png" style="width: 80%;" />
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images/Breakdown%20of%20Corrupted%20Data.png" style="width: 80%;" />
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images/bmi-data-availability.png" style="width: 80%;" />

### Data balance issues:
The dataset is unbalanced, which may lead to poor model performance on the minority class. 
Techniques such as SMOTE are needed to balance the data.

### In the project, I used various techniques, and here are some of them:


