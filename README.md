# Discovering analyzing and predicting Stroke using DataMining techniques
![bioinformatics](https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/bioinformatics.jpg)
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

## Goals of the project 🧪🗃️🎯
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

> [!IMPORTANT]
> #### Main assumptions and simplifications in the project:
> Since we are dealing with medical data that relates to predictions of whether or not a patient will have a stroke (a life or death scenario), the main assumption I use throughout the project is that **_reducing_ false negatives** is the most
> important metric to achieve (since this situation means identifying a person who is going to have a stroke as a person who will not have a stroke, which is a very big risk to their life (false identification).

---
## How are we going to achieve those goals?
### Knowledge Discovery in the Database (the KDD mindset):

#### 1. The first step in the KDD process is **SELECTION**, where relevant data is collected and selected for analysis.
  In this project, we use a medical dataset containing 12 patient-related characteristics, such as age, gender, blood glucose levels, BMI, smoking status, and medical history (including hypertension and heart disease).
  
  The main feature we focus on - in analyzing the dataset is the stroke feature, which indicates whether a patient has experienced a stroke or not.
  This target feature is essential for building predictive models aimed at correctly classifying stroke.

```
Description of the data table: 
                 id          age  hypertension  heart_disease  avg_glucose_level          bmi       stroke
count   5110.000000  5110.000000   5110.000000    5110.000000        5110.000000  4909.000000  5110.000000
mean   36517.829354    43.226614      0.097456       0.054012         106.147677    28.893237     0.048728
std    21161.721625    22.612647      0.296607       0.226063          45.283560     7.854067     0.215320
min       67.000000     0.080000      0.000000       0.000000          55.120000    10.300000     0.000000
25%    17741.250000    25.000000      0.000000       0.000000          77.245000    23.500000     0.000000
50%    36932.000000    45.000000      0.000000       0.000000          91.885000    28.100000     0.000000
75%    54682.000000    61.000000      0.000000       0.000000         114.090000    33.100000     0.000000
max    72940.000000    82.000000      1.000000       1.000000         271.740000    97.600000     1.000000

Stroke Case Summary (in data-set):
Number of stroke-free cases (stroke = 0): 4861
Number of stroke cases (stroke = 1): 249
```

#### 2. After selecting the data, the next step is **PREPROCESSING**. This step involves cleaning and organizing the data to ensure its quality and reliability.
  - In the dataset, we encounter missing values in the BMI column and unavailable information marked as "unknown" in the smoking_status attribute.
    
  -  These issues are addressed using various preprocessing techniques such as filling in missing values using median / value average, data filtering, and handling categorical outliers. In addition, all features are checked for                      consistency, duplication, and appropriate formatting to prepare the data for further analysis.

#### 3. The third stage is **TRANSFORMATION**, which involves converting the data into formats suitable for modeling. 
  - In this project, numeric characteristics can be normalized or rescaled, and categorical variables - such as gender, job type, and residence type - can be encoded using techniques such as hot one encoding or categorical label encoding.          Furthermore, the dataset can be balanced using synthetic techniques such as SMOTE to ensure that the minority group (stroke cases) is equally represented as the majority category (non-stroke), thus improving the effectiveness of AI models.

#### 4. Then comes the **DATA MINING** stage, where various artificial intelligence and machine learning algorithms are applied to uncover patterns (or different behaviors) and build predictive models.
  - In the stroke prediction task, several models such as Nearest-K Neighbors (KNN), Support Vector Machines (SVM), Logistic Regression can be implemented.

  - I personally chose to use FOREST RANDOM and XGBOOST (and later on a few more models) which is a gradient state based model. The models are trained to classify whether a patient is at risk of stroke. Each model is evaluated based on metrics     such as accuracy, precision, 1F score and more... to determine its effectiveness. Cross-validation techniques can also be used to ensure that the results are robust and not biased by splitting the data set.

#### 5. At last comes the **INTERPRETATION AND EVALUATION** phase which evaluates the model's results to draw conclusions and insights.
  - This phase includes comparing the model performance, identifying the most influential characteristics that contribute to stroke risk, and interpreting the results in a relevant medical context.
    
  - The goal at the end of this phase is to derive actionable knowledge from the data that can aid in the early detection or prevention of stroke.
    The results are then visually presented in diagrams and the main findings of the data are presented to support further conclusions and further research.

<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/KDD-steps-Knowledge-Discovery-in-Databases.png" style="width: 90%;" />

---

## Our dataset:

  - **Id** - is a unique integer identifier assigned to each patient. It ranges from 67 to 72,940 and does not contain missing values.
  - **Gender** - is a categorical attribute indicating the patient's gender. It can take one of three values: "male", "female" or "other", with no missing values.
  - **Age** - is a continuous numeric variable ( float ) representing the patient's age. The values range from 0.08 years to a maximum age of 82 years, and there are no missing values in this field.
  - **hypertension** - is a binary variable where a value of 1 indicates that the patient has hypertension, and 0 indicates that they do not. This attribute contains complete data with no missing values.
  - **disease_heart** - is another binary variable where 1 represents the presence of heart disease and 0 represents its absence. Like the previous field, it does not contain missing data.
  - **married_ever** - is a categorical variable indicating whether the patient has ever been married. It has two possible values: "yes" and "no", and has no missing values.
  - **type_work** - classifies the type of employment of the patient. It can be one of the following: "children", "government job", "never worked", "private", or "self-employed". This field is complete with no missing data.
  - **type_Residence** - identifies whether the patient lives in a "rural" or "urban" area. This categorical variable is fully populated.
  - **avg_glucose_level** - is a float representing the patient's average blood glucose level. It ranges from 55.12 to 271.74 and contains no missing values. bmi - or body mass index, is a continuous numeric attribute ranging from 10.3 to .97.6. However, this field contains 201 missing values, which need to be handled during preprocessing. 

  - **status_smoking** - is a categorical attribute with four possible values: "never smoked", "previously smoked", "smoker", and "unknown". Although it does not contain missing values in the traditional sense, "unknown" is used as a placeholder for data that is not available.
  - **stroke** - is the target binary variable of the dataset. A value of 1 indicates that the patient has had a stroke, and 0 indicates that he has not. There are no missing values in this field.
    
#### A diagram I drew of the age distribution.

<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/age-distribution.png" style="width: 55%;" />

### Missing Values:
  - **Bmi values** - Approximately 3.9% of records are missing - Which is 201 rows that are missing.
  - **Smoking_status values** - a couple of unknown values.
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/Comparison%20of%20Valid%20vs%20Corrupted-Unknown%20Data.png" style="width: 60%;" />
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/Breakdown%20of%20Corrupted%20Data.png" style="width: 60%;" />
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/bmi-data-availability.png" style="width: 60%;" />

### Data balance issues:
The dataset is unbalanced, which may lead to poor model performance on the minority class. 
Techniques such as SMOTE are needed to balance the data.

> [!IMPORTANT]
> #### **What is SMOTE? Why do we need it?**
>  **SMOTE stands for Synthetic Minority Over-sampling Technique.**
> 
> It's a way to balance imbalanced datasets which helps increase performance & accuracy (alongside helping precision, recall etc... more on that later);
> when one data class (or many) has/have way fewer examples than the others (like 90% healthy patients, 10% stroke patients), AI / Machine learning / Deep learning models tend to favor the majority class.
>
> **Specifically in classification tasks** (such as this project) we see this "favoritism" very clearly (classification tasks = predicting which category or class an input belongs to, like spam vs not spam or stroke vs no stroke).
>
> **Why does it happen?**
>
> Because when we are training a model we give it rewards & punishments (negative rewards) - by the way this way of training is called reinforcement learning.
> _When training a model that way,_ it may just predict "healthy" all the time and still be 90% accurate (because 90% of the time we get healthy patients) the model is trained to get as few punishments as it can so it does that.
> **but that’s useless** if you care about identifying strokes.
>
> Since we are dealing with a greatly imbalanced dataset **_we need to do something about it!_**  
> There are a couple ways to do that, like:
>
> Duplicating data from the minority class (over-sampling), removing some samples from the majority class (undersampling), adjusting class weights during training to penalize misclassifying, splitting 
> the data into k balanced subsets (**_k-fold sampling_** - more on that later) and just getting more data overall...
> 
> **BUT** there is another way which is soooo smart and just brilliant that helps deal with that problem amazingly.
> Like you probably already guessed it is called SMOTE! SMOTE creates a new, fake-but-realistic examples of the minority class.
>
> **How does it do it?**
> 1. It takes a minority sample (e.g. a stroke case).
> 2. It finds it's nearest neighbors in the minority class.
> 3. It Picks one neighbor at random.
> 4. &Generates a new sample somewhere in between the two samples (interpolation).
> _It's very simillar to another ai algorithm that we will talk about later (knn)._
> #### Here is a gif showing SMOTE in action:
> <p align="center">
>  <img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/raw/main/images%26gifs/smote_animation.gif" width="350">
> </p>
> As you can see blue squares are the majority class while the orange circles are the minority, SMOTE deals with it by generating new synthetic minority samples.

> [!CAUTION]
> Using SMOTE too much is **bad**,
> Generating too many synthetic samples can make the model overfit to the artificial data and miss real-world variation. It can also blur class boundaries and reduce model performance if used without care.
>
>> So as always,
>> **_with great power comes great responsibility ~uncle Ben_** 🕷

### In the project, I used various techniques of dealing with missing or unbalanced data, moreover we need a more general, proper preprocessing of the data before we can start working on it
#### here are some of the preprocessing aspects:

#### SMOTE for unbalanced data
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-using-DataMining-ResearchProject/blob/main/images&gifs/stroke-class-distribution-before&after-smote.png" style="width: 100%;" />

#### Median filling for missing values
```python
# Fill missing values in numeric columns
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
...
# Just in case: Fill remaining missing values
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))
```
#### Ignoring the problem (missing values) 😌🙃
```python
# Replace 'Unknown' with NaN in smoking_status
df['smoking_status'] = df['smoking_status'].replace('Unknown', pd.NA)
...
```
#### Raw categorical data is not compatible with some of our algorithms, it needs encoding:
```python
# Map binary columns
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
...
df = pd.get_dummies(df, columns=['gender', 'work_type', 'smoking_status'], drop_first=True)
```

---

## A comparative review of different AI models for performing data mining

- In our task, the main goal is stroke prediction using the given medical dataset, and data mining techniques that can draw new conclusions and distinguish different patterns. Several DATA MINING techniques can be applied to achieve              significant results. 
- Each approach has its own advantages but disadvantages, and the choice of algorithm can significantly affect performance.
- Here is the comparison I made between 4 well-known data mining prediction models with their advantages and disadvantages:

### Logistic Regression
1. **Logistic Regression** is a statistical model that is used mainly in binary classification problems (such as stroke prediction 0 – no stroke, 1 has a stroke/will have).
  - The method is simple, fast and easy to interpret, which gives us the ability to understand the impact of each characteristic on the likelihood of stroke.
  - It's coefficients provide clear insights into the strength of the relationships between risk factors for stroke. However, Logistic Regression assumes a linear relationship between characteristics and the logistic odds, which may fail to        identify complex, non-linear data in the dataset.
  - This limitation can limit the predictive power of the method, especially when the relationship between characteristics and stroke is nonlinear. Absolutely.

<p align="center">
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/LogisticRegressionGif.gif" width="500">
</p>

### KNN (K Nearest Neighbors)
2. **KNN (K Nearest Neighbors)** is a non-parametric method that classifies instances based on the majority label of the nearest neighbors in the feature space. In other words, it classifies the object according to the majority category of its K closest neighbors.
- It is easy to implement and makes no assumptions about the underlying data distribution, which is useful for datasets with complex patterns.

#### Disadvantages:
- It can be sensitive to irrelevant or graded features, and it's performance tends to deteriorate with high-dimensional data.
- KNN is very computationally expensive (especially when the data has high dimensionality) with the computational cost increasing exponentially for each additional dimension during prediction, since it requires calculating distances from all     training samples.

<p align="center">
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/KNNgif.gif" width="400">
</p>

### SVM (Support Vector Machines)
3. **SVM (Support Vector Machines)** is a good model for classifying points even in high-dimensional spaces,
  - It has the ability to model nonlinear decision boundaries using kernel functions.
  - SVMs are particularly effective for datasets where classes are separable and the gap between them is important.
  - In the context of prediction, SVMs can help address the problem of class imbalance through the use of class weights.
    
  #### Disadvantages:
  - They can be sensitive to the choice of hyperparameters and kernel functions, and often require extensive tuning.
  - SVMs are less interpretable (i.e., require a deep understanding of the features) than logistic regression and can be computationally expensive for large datasets.

<p align="center">
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/SVMgif.gif" width="400">
</p>

### Random forests & XGBoost
4. **Random forests** are learning and prediction methods that build multiple decision trees and pool their predictions.
  - The models are known for being noise-resistant, and able to handle missing data relatively well.
  - They can model complex predictions without overfitting, such as Single decision trees.
  - Random forests are very effective for this type of health data (in this project) because they can handle mixed types of variables and are less sensitive to data imbalance when used with appropriate techniques.

    #### Disadvantages:
  - Their main disadvantage is their interpretability - while the importance of features can be extracted, the model itself functions as a black box, which is a significant disadvantage if we want to know the trends and problematic                 features ourselves, especially in this medical application.

#### XGBoost (Extreme Gradient Boosting)
- **XGBoost (Extreme Gradient Boosting)** builds decision trees sequentially, where each tree tries to fix the errors made by the previous ones. It minimizes a loss function (like log loss or mean
  squared error) and includes regularization to avoid overfitting. It's known for its speed and performance

<p align="center">
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Random%20Forest%20gif.gif" width="500">
<br/>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/XGboostgif.gif" width="500">
</p>

### Summary
  In summary, each method offers unique advantages and disadvantages. Logistic regression is the easiest to interpret, KNN offers simplicity of the model itself but can be computationally expensive, especially with large data and will be         sensitive to imbalances, SVM provides strong performance with careful tuning but can be computationally expensive when dealing with a lot of data. And random forests balance their accuracy at the expense of unclear interpretation.

--- 

## Results
### XGBoost:
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/XGBoost-ConfusionMatrix.png" width="400">


#### XGBoostConfusionMatrix-Threshold-0.4
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/XGBoost-ConfusionMatrixThreshold-0.4.png" width="400">


#### XGBoostConfusionMatrix-BestThreshold-0.03
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/XGBoost-ConfusionMatrixBestThreshold-0.03.png" width="400">

### XGBoost – Results Analysis:
When using XGBoost with a threshold of 0.03, the model achieves an Accuracy of about 75%, which means that it correctly predicts stroke or non-stroke in 3 out of 4 cases.
- While Precision is relatively low (around ~12.3%), Recall is quite high at 66%, which means that the model is good at identifying true stroke cases.
- The F1 score – which is the balance relationship between Accuracy and Recall – is around 0.21, indicating that although the model detects many positive cases (stroke), it also misclassifies a significant (quite large) number of non-stroke cases as stroke.

In general, the model's results are quite good. Considering all the data, it does meet the task of the problem and reduces the amount of false negatives, which is a very, very significant indicator in medical problems (where a person is identified as not going to have a stroke and he actually will have a stroke). It has problems identifying a person who is not sick with a disease, but as an initial test and an indication measure, you can be "calm" if you are diagnosed as not sick with the help of the model, and if so, reduce the number of additional tests that are not necessary for people. Therefore, I would say that it meets its goal. However, there are significant improvements that can be made to the model, and I will detail them in later sections.

### RandomForest:

<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/RandomForest-ConfusionMatrix.png" width="400">

#### RandomForestConfusionMatrix-Threshold-0.4
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/RandomForest-ConfusionMatrixThreshold-0.4.png" width="400">

#### RandomForestConfusionMatrix-BestThreshold-0.26
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/RandomForest-ConfusionMatrixBestThreshold-0.26.png" width="400">

### RANDOM FOREST – Results Analysis:
The RANDOM FOREST model, with a threshold of 0.26, gives a slightly better overall accuracy of 78.7%, and a slightly higher F1 score (0.21) compared to XGBoost. 
- The accuracy is only slightly higher (12.9%), and the recall is lower than XGBoost (58%), which means that it detects fewer strokes but is more confident in its predictions when it does detect.
- Like XGBoost, the model performs well in non-stroke cases because the recall is lower for strokes compared to XGBoost. The model may be less superior if capturing as many strokes as possible is the priority.

Like the previous model, the results are quite good, considering all the data it does meet the problem tasks and reduces the number of false negatives. The model performs better in terms of identifying a person who does not have a disease (fewer such cases compared to the previous model) but is a little more sensitive to false positive diagnosis problems. However, there are significant improvements that can be made to the model, and I will detail them in later sections.

### Suggestions for improvements and final comparisons:

In terms of my suggestions for improving the model, I think one of the most significant would be to assign a different "penalty", for each wrong identification.
As the models (both) operate, I saw that there were a lot of false positives, with a higher penalty for such errors we could probably reduce the amount of these errors, but there is a significant need to keep the amount of false negatives as low as possible (since false postivies are dangerous and valuable to identify early to save lives even).


I think that in general it would help if we combined 2 different models that would run one after the other (and maybe even other types of models), one that would be a more "heavy filter" of the information, and would be able to robustly filter out false positives and then run a more "sensitive" model for qualitative clarification / a second test on the information so that we could keep the amount of false negatives very low but we wouldn't have to run too many "unnecessary" tests for a large number of people and thus we could find the "balance".
- This method would be similar to Ensemble learning (more on this later).

- In addition, I think it is possible to use, for example, XGBoost with the method of
  K-Fold Cross-Validation. To split the information into K equal samples that will be used for both the control and test groups.

#### Other suggestions for improvement include:
- **Fine tuning:** Continue experiments with different classification thresholds (thresholds). In this case, a low threshold (0.03 for XGBoost) yielded quite good Recall and this is a promising direction.

- **Advanced sampling techniques:** Beyond SMOTE, consider using SMOTE- (ENN undersampling) ENN or ADASYN, which combine oversampling with cleaning of noisy or ambiguous samples.

- **Ensemble learning models:** Combine predictions from XGBoost and Random Forest in a voting or stacking ensemble to extract the benefits of both recall and precision.

- **Integrate existing and known medical knowledge:** You can always consult and use clinical guidelines or expert knowledge (a doctor / research or clinical doctor) to create more reliable models that rely on solid and known science.

- **Explanation tool:** It may be a good idea to use tools like SHAP or LIME to interpret the model predictions. In medical applications like this project, understanding why a model predicts stroke is sometimes as important as the prediction itself to help identify more cases in the future.

#### In terms of inferences on the data:**
XGBoost may be better suited to this medical scenario because its recall is higher, which ensures that fewer strokes are missed. In healthcare, false negatives (missing strokes) are often more dangerous than false positives.

Random Forest, with slightly better accuracy and precision, may be suitable in environments where "false alarms" are costly (e.g. limited resources for follow-up testing).









