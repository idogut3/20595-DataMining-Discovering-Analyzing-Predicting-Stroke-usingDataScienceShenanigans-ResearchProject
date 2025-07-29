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

> [!NOTE]
> The project was made in 2 parts (and 3 different files) so the placement of the readme contents might be a bit "off" apologies in advance.

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

#### In terms of inferences on the data:
XGBoost may be better suited to this medical scenario because its recall is higher, which ensures that fewer strokes are missed. In healthcare, false negatives (missing strokes) are often more dangerous than false positives.

Random Forest, with slightly better accuracy and precision, may be suitable in environments where "false alarms" are costly (e.g. limited resources for follow-up testing).

---

## Association rule learning:

> [!IMPORTANT]
> ### What is Association rule learning?
> Association rule learning is a way for computers to find interesting relationships or patterns in large sets of data.
> 
> Imagine you're looking at customer shopping habits in a supermarket.
> If the data shows that people who buy bread often also buy butter, the computer can learn that pattern and create a rule like: "If someone buys bread, they’re likely to buy butter too."
> 
> **These rules help businesses make better decisions**, like placing related items close together or suggesting products online.
> 
> **_In the context of stroke prediction_** it is helpful for us to uncover the association rules that lead to stroke to gain useful insight into the factors which lead up to stroke.
>
>
> <p align="center">
> <img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/association-rule-learning-visual-example.png" width="450">
>
> <br/>
> 
> <img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Association%20rule%20mining%20applications.png" width="500">
> </p>


#### _I chose the **FP-Growth algorithm** to do such task. This algorithm is a popular method used to discover patterns and associations in large datasets._


> [!NOTE]
> #### how it works? why is it efficient?
> Instead of testing every possible combination of features in the data (which is slow and wasteful for large datasets), FP-Growth works by building a special data structure called an “FP tree” that compresses the data and allows the algorithm > to quickly find groups of items that frequently appear together.
> 
> Other algorithms, such as the Pope algorithm, are less computationally expensive/unsuitable for our data because they repeatedly scan the dataset and generate a large number of of candidate combinations, which can be very slow. In contrast, > FP-Growth avoids this by using a special data structure of a “compact” tree, which makes it fast and scalable and helps it find meaningful relationships in our stroke prediction dataset efficiently.

> [!NOTE]
> #### Why is FP-Growth useful in our case?
> FP-Growth is useful because it can uncover hidden associations between patient characteristics—such as age, job type, or smoking status—and the likelihood of having a stroke.
> Such insights can be valuable for understanding risk factors, guiding medical decisions about a patient, or conducting medical research.

### Before running FP-Growth, I had to balance the Data:

Since stroke cases are rare in the original dataset, most of the patterns the algorithm would find would only reflect the majority who did not have a stroke. To fix this, I split the Data into 2 groups, a “stroke” group and a “non-stroke” group.

- I chose a random sample from both groups and joined it together to a new group having equal representation of both stroke cases and none stroke cases.
- [x] This allowed my algorithm to focus more on learning the differences between the two groups, and on data that might be related to stroke. This is a common way to make association rule mining more useful when working with rare events like stroke.

```python
# ---------------------------
# Balance the Dataset (for Association Rule Mining)
# ---------------------------
df_majority = df[df['stroke'] == 0]
df_minority = df[df['stroke'] == 1]
df_balanced = pd.concat([
    df_majority.sample(n=len(df_minority), random_state=42),
    df_minority
]).reset_index(drop=True)
...
```

In addition to balancing the data, I needed to categorize numeric data into "bins" instead of continuous numeric functions:
So I binned continuous numbers like age, glucose, and BMI into labeled categories like
"old" or "high glucose". 

- [x] This is a necessary process to make the data usable for the FP-Growth algorithm, which needs data that works more like checkboxes (yes/no), rather than numeric                (continuous) numbers without a step This, the algorithm will not be able to find patterns in these important features.

```python
# Bin continuous variables
df_trans['age'] = pd.cut(df_trans['age'], bins=[0, 30, 50, 100], labels=['young', 'middle-aged', 'old'])
# BMI Binning
df_trans['bmi'] = pd.cut(df_balanced['bmi'],
                         bins=[0, 18.5, 25, 30, 35, df_balanced['bmi'].max()+1],
                         labels=['underweight', 'normal', 'overweight', 'obese', 'extremely obese'])

# Glucose Level Binning
df_trans['avg_glucose_level'] = pd.cut(df_balanced['avg_glucose_level'],
                                       bins=[0, 70, 100,  df_balanced['avg_glucose_level'].max()+1],
                                       labels=['hypoglycemic', 'normal', 'diabetic/hyperglycemic'])
...
```

**_Regarding blood glucose level & BMI I validated the levels with online medical sources in order to accurately bin the values together._**
_see sources at the end._

> [!WARNING]
> I simplified it for me and binned the values according to the correct ranges for an adult (male). 
> In a real study, these levels should be considered according to the age of the patient with his other data (sex, gender, diseases / medical history, etc...).

### BMI Range Classification

| BMI Range     | Label            |
|---------------|------------------|
| 0 – 18.5      | underweight      |
| 18.5 – 25     | normal           |
| 25 – 30       | overweight       |
| 30 – 35       | obese            |
| 35 – max+1    | extremely obese  |

### Glucose Range Classification (mg/dL)

| Glucose Range | Label               |
|----------------|---------------------|
| 0 – 70         | hypoglycemic        |
| 70 – 100       | normal              |
| 100 – max+1    | diabetic/hyperglycemic |


## Results running FP-Growth
> [!IMPORTANT]
> Disclaimer: Our goal was to find a connection between data and associations related to stroke, so from all the information the algorithm brought, we filtered out only the conclusions relevant to the stroke that appear with high frequency (within the balanced database > we created).

> [!NOTE]
> ### Let's briefly recall what support/confidence is:
> **Support** tells us how common a particular "pattern" or rule is in the data set.
> For example, if a rule has a support of 45%, this means that it applies to 45% of all patients in your data.
> 
> Confidence tells us how likely an outcome (such as a stroke) is, given that the condition in the rule is met.
> 
> For example, if a rule has a confidence of 68.9%, this means that when the conditions are met (such as being old), there is a 68.9% chance that the person has had / will have a stroke.

### Results for MIN_SUPPORT = 40%, MIN_CONFIDENCE = 60%

```python
Top Frequent Itemsets:
     support                                           itemsets
0   1.000000                               (gender_Other_False)
1   0.997992                     (work_type_Never_worked_False)
20  0.997992  (gender_Other_False, work_type_Never_worked_Fa...
21  0.935743     (gender_Other_False, work_type_children_False)
2   0.935743                         (work_type_children_False)

Top Stroke-Related Rules (Readable):
rules:                                                                                                                        support    confidence      lift
If [age is old] → Then [stroke = Yes, work type children is False]                                                          45.38%     68.90%          1.39
If [age is old, gender Other is False] → Then [stroke = Yes, work type children is False]                                   45.38%     68.90%          1.39
If [age is old] → Then [gender Other is False, stroke = Yes, work type children is False]                                   45.38%     68.90%          1.39
If [age is old, work type Never worked is False] → Then [stroke = Yes, work type children is False]                         45.38%     68.90%          1.39
If [age is old] → Then [stroke = Yes, work type Never worked is False, work type children is False]                         45.38%     68.90%          1.39
If [age is old, gender Other is False, work type Never worked is False] → Then [stroke = Yes, work type children is False]  45.38%     68.90%          1.39
If [age is old, work type Never worked is False] → Then [work type children is False, gender Other is False, stroke = Yes]  45.38%     68.90%          1.39
If [age is old, gender Other is False] → Then [stroke = Yes, work type Never worked is False, work type children is False]  45.38%     68.90%          1.39
If [age is old] → Then [work type children is False, gender Other is False, work type Never worked is False, stroke = Yes]  45.38%     68.90%          1.39
If [age is old] → Then [stroke = Yes]                                                                                       45.38%     68.90%          1.38
```

### In our results we mainly got the following:

#### "If age is older -> then stroke = yes"
  - Receives 45.38% support and 68.9% confidence level.


#### "If age is older and gender is female and she worked at least once -> stroke = yes"
  - Receives 45.38% support and 68.90% confidence level.
...


In general the results are pretty much the same and all refer to the fact that there is a high chance of stroke when the age is older.

#### In relation to the first claim:

This means that 45% of all people in the (balanced) dataset were older people who also had a stroke and did not work as children - and among those who were older, 68.9% had a stroke and matched the rest of the group. 
This is a pretty strong and interpretable relationship, especially in healthcare data. The lift value of 1.39 (Lift) shows how strong and significant the rule is - it adapts itself to how common the outcome is in general. In the above case, it shows that this is the most common outcome in 39% of cases.

#### **These results suggest that age is a dominant factor in predicting stroke** – **_which is not surprising_**, but it is important to see this clearly reflected in the data.


- It also shows that certain types of work (such as "childhood work = TRUE" or "never worked = FALSE") tend to occur less frequently in stroke cases in older people which could indicate that active people, who have worked since the beginning of their lives / had at least one job are at lower risk of stroke.

#### This is an insight that is less surprising today, since it is known that the more your brain is stimulated throughout your life (and the more you are active), the lower your chance of experiencing a stroke.


> [!NOTE]
> **Although these types of rules do not replace a predictive model**, they are useful for identifying high-risk groups, building hypotheses, or explaining patterns to stakeholders such as doctors or health care social workers in a simple "if-then" format.
> 
> _We should also recall and note that in this question we only addressed the conclusions relevant to STROKE, but during the data collection we found more general insights about the data (which are not relevant to the project, so I will not mention most of them)._

---

## Predicting using Clustering, AI models

### Our clustring model:
I chose to use **_k-means clustering_** approach. It is a fairly simple method to implement and our database is not "particularly large" (about 5000 records) and so despite the algorithm's inefficiency it is good enough for now. 

> [!NOTE]
> Later on I used another clustering algorithm to hopefully get better results, so keep scrolling to see the **Agglomerative Clustering approach**.


> [!IMPORTANT]
> ### What is K-Means Clustering
> K-Means clustering is a simple and intuitive algorithm that groups similar data points together into k clusters, making it easy to spot patterns in complex data.
> 
> #### How does it work?
> Imagine you have a bunch of dots scattered on a page, and you want to divide them into groups. 
> 1. First, you choose how many groups you want (this is the "k"). 
> 2. Then, the algorithm randomly picks k points to start as the "centers" of the groups. Each dot is assigned to the nearest center.
> 3. After that, it moves the centers to the middle of their group, and reassigns the dots again. This keeps happening - updating the group centers and reassigning points, until the groups don’t change much anymore.
> 4. The result is clear, separate clusters that are easy to analyze and understand.
>
> #### Strengths and weaknesses:
> ##### Strengths:
> - Easy to understand: Results are simple to interpret, which can help doctors, therapists, and other professionals identify common risk factors among different groups (e.g., stroke patients).
> - Flexible: You can easily adjust the number of clusters (k) to explore different groupings.
> - Evaluatable: You can use tools like the silhouette index (see later) to compare and choose the best clustering.
>
> ##### Weaknesses:
> - Sometimes the algorithm can get stuck in a local optimum — meaning it may not find the best possible grouping.
> - It works best when clusters are clearly separated and roughly equal in size — it can struggle with more complex shapes or overlapping data.
> - Choosing the right number of clusters (k) isn’t always obvious and may require trial and error.
> <p align = 'center'>
> <img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/K-Means-Clustring-gif.gif" width="400">
> </p>


### Clustering quality metrics:

There are several clustering quality metrics that are suitable for our problem, here are a few:

#### 1. Silhouette Score:
One of the most popular ways to check whether clustering makes sense. The metric measures how close each data point is to its assigned cluster, compared to other clusters. Values closer to 1 indicate that the sample is well-matched to its cluster and poorly-matched to other clusters - which is exactly what we want.

#### 2. Calisinski-Harabaz Score:
This metric measures how "dense" each cluster is and how "far apart" the clusters are. A high score means that internally consistent groups (e.g., similar age, glucose level, BMI, etc.) have been found and are well separated from each other. This helps to assess whether clusters may represent true stroke risk patterns.

#### 3. Davies-Bouldin Index:
This is an internal evaluation index, where the validation of the clustering performance is done using quantities and features inherent in the dataset. A disadvantage of this is that a good value reported by this method does not imply the best information retrieval.
These indices give a clear picture of whether the clustering results may be useful for identifying patterns related to stroke.

#### We used **_Silhouette Score_** because it is popular and easy to implement. 

### The cluster analysis steps are detailed below:
  1. We loaded the data and imported libraries.

  2. We ran the General Preprocessing that we did in the previous models, which includes filling in missing values in various ways (see previous post for explanations – a picture of the code is attached).

  3. We dropped unnecessary parameters in the methods (for example, ID, which does not have a value that can predict a cluster – a random number, and we also dropped stroke = 1/0 because we hope to find groups that divide themselves without "knowing" in advance whether   this group has experienced a stroke or not – all in order to know how to search for the given groups that are related to stroke).

  4. We adjusted all the data parameters (the Attributes) to a range between [0,1] since the algorithm can tend to prioritize "large" areas since they "indicate" a sharper difference in the data, which is not true if we want to treat each data in a way that its impact      probability is equal to each other.

  5. We ran on a number of possible K-s (to find an optimal number of clusters).

  6. We ran the algorithm (we calculated what each point was expected to be and what its Actual value was).

  7. We calculated the **Silhouette Score** of the algorithm after the results stage.

  8. Plotted the results according to each graph.

### _< General Preprocessing >_
```python

# Replace 'Unknown' with NaN in smoking_status
stroke_data_table['smoking_status'] = stroke_data_table['smoking_status'].replace('Unknown', pd.NA)

# Fill missing values
stroke_data_table['bmi'] = stroke_data_table['bmi'].fillna(stroke_data_table['bmi'].median())
stroke_data_table['ever_married'] = stroke_data_table['ever_married'].fillna(stroke_data_table['ever_married'].mode()[0])
stroke_data_table['Residence_type'] = stroke_data_table['Residence_type'].fillna(stroke_data_table['Residence_type'].mode()[0])

# Convert binary categoricals
stroke_data_table['ever_married'] = stroke_data_table['ever_married'].map({'Yes': 1, 'No': 0})
stroke_data_table['Residence_type'] = stroke_data_table['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Restore 'Unknown' for smoking_status and one-hot encode categorical columns
stroke_data_table['smoking_status'] = stroke_data_table['smoking_status'].fillna('Unknown')
df = pd.get_dummies(stroke_data_table, columns=['gender', 'work_type', 'smoking_status'], drop_first=True)

```

## Results

#### At first I tried testing a small range of clusters, between 2-15 maximum (to see the effect of the catalog into groups.

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Silhouette%20Score%20vs%20Number%20of%20Clusters.png">
</p>

**As you can see, the process "was not very successful"** - there was some kind of PEAK for K=8 clusters, but it is certainly much below what was expected with a Silhouette Score that is closer to 0 than to 1 (a little over 0.25) meaning a not very high-quality catalog for different clusters and a great inability to diagnose any cluster.


Therefore, I said that **_I would try to increase the maximum K (and I ran it up to K = 100)_**, yes it is of course not practical to extract medical information from it that can help us predict a stroke and of course the more I define the K as larger and larger, the
Silhouette Score will increase since at some point it will simply define every point in space as a cluster and therefore reach 100% success. But I wanted to try to see if there would be a significant jump because there are still 5000 records and reaching 100 clusters of ~50 points in each of them with a very high Silhouette Score could really help us perhaps draw a few conclusions about image cataloging.

#### So, here are the results up to K = 100:

<br/>
<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Silhouette%20Score%20vs%20Number%20of%20Clusters%20100K.png">
</p>

**Unfortunately, although there is indeed an increase in the Silhouette Score**, this is a negligible and insignificant increase, which is expected from the number of clusters we have. And in fact we reach a situation where the highest Silhouette Score we received is 0.34, a very low number and even disappointing for dividing K = such a large number (98 clusters).


Therefore, the most significant information that can perhaps be deduced is for K=8, which we saw in the previous graph, but certainly the conclusions/results that we will receive are not unambiguous and unambiguous for diagnosing stroke, since the Silhouette Score is very low, which indicates that "the information is quite overlapping."


### Below are results for K = 8:

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Average%20medical%20features%20per%20cluster.png">
</p>

```python

Overall stroke rate in full dataset: 4.87% (249 out of 5110)

Stroke=1 distribution across clusters (adds up to 100%) with counts:
Cluster 0: 24.90% of all stroke=1 (overall cluster patients=1999) => stroke rate in cluster: 3.10%
Cluster 1: 0.80% of all stroke=1 (overall cluster patients=690) => stroke rate in cluster: 0.29%
Cluster 2: 24.50% of all stroke=1 (overall cluster patients=650) => stroke rate in cluster: 9.38%
Cluster 3: 10.04% of all stroke=1 (overall cluster patients=712) => stroke rate in cluster: 3.51%
Cluster 4: 20.88% of all stroke=1 (overall cluster patients=761) => stroke rate in cluster: 6.83%
Cluster 5: 0.00% of all stroke=1 (overall cluster patients=1) => stroke rate in cluster: 0.00%
Cluster 6: 18.88% of all stroke=1 (overall cluster patients=275) => stroke rate in cluster: 17.09%
Cluster 7: 0.00% of all stroke=1 (overall cluster patients=22) => stroke rate in cluster: 0.00%
```
```python
Number of patients per cluster:
cluster
0          1999
1           690
2           650
3           712
4           761
5             1
6           275
7            22
Name: count, dtype: int64
```

| Cluster | Age   | Avg Glucose Level | BMI   | Hypertension | Heart Disease |
|---------|-------|-------------------|-------|--------------|----------------|
| 0       | 42.44 | 101.71            | 29.79 | 0.01         | 0.0            |
| 1       | 6.87  | 94.29             | 20.23 | 0.0          | 0.0            |
| 2       | 59.88 | 116.58            | 30.91 | 0.43         | 0.0            |
| 3       | 45.36 | 106.0             | 30.45 | 0.09         | 0.0            |
| 4       | 53.75 | 108.85            | 30.71 | 0.09         | 0.0            |
| 5       | 26.0  | 143.33            | 22.4  | 0.0          | 0.0            |
| 6       | 68.43 | 137.09            | 30.05 | 0.23         | 0.0            |
| 7       | 16.18 | 96.04             | 25.55 | 0.0          | 0.0            |



| Cluster | Male | Other | Never Worked | Private | Self-employed | Children | Formerly Smoked | Never Smoked | Smokes |
|---------|------|--------|---------------|---------|----------------|----------|------------------|----------------|--------|
| 0       | 0.37 | 0.0    | 0.0           | 0.83    | 0.0            | 0.0      | 0.0              | 0.64           | 0.0    |
| 1       | 0.53 | 0.0    | 0.0           | 0.0     | 0.0            | 1.0      | 0.02             | 0.08           | 0.0    |
| 2       | 0.34 | 0.0    | 0.0           | 0.18    | 0.76           | 0.0      | 0.05             | 0.69           | 0.02   |
| 3       | 0.41 | 0.0    | 0.0           | 0.71    | 0.13           | 0.0      | 0.0              | 0.0            | 1.0    |
| 4       | 0.44 | 0.0    | 0.0           | 0.64    | 0.2            | 0.0      | 1.0              | 0.0            | 0.0    |
| 5       | 0.0  | 1.0    | 0.0           | 1.0     | 0.0            | 0.0      | 1.0              | 0.0            | 0.0    |
| 6       | 0.59 | 0.0    | 0.0           | 0.57    | 0.29           | 0.0      | 0.28             | 0.33           | 0.22   |
| 7       | 0.5  | 0.0    | 1.0           | 0.0     | 0.0            | 0.0      | 0.0              | 0.64           | 0.0    |

```python


Cluster categorical traits per cluster:

Cluster 0:
               age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level          bmi
count  1999.000000   1999.000000         1999.0   1999.000000     1999.000000        1999.000000  1999.000000
mean     42.437719      0.011006            0.0      0.680340        0.495248         101.706738    29.785293
std      17.742062      0.104354            0.0      0.466461        0.500103          39.990852     7.487578
min       8.000000      0.000000            0.0      0.000000        0.000000          55.120000    11.500000
25%      27.000000      0.000000            0.0      0.000000        0.000000          76.955000    24.600000
50%      41.000000      0.000000            0.0      1.000000        0.000000          90.600000    28.300000
75%      56.000000      0.000000            0.0      1.000000        1.000000         111.090000    33.100000
max      82.000000      1.000000            0.0      1.000000        1.000000         266.590000    97.600000

Cluster 1:
              age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level         bmi
count  690.000000         690.0     690.000000         690.0      690.000000         690.000000  690.000000
mean     6.866667           0.0       0.001449           0.0        0.504348          94.287304   20.226087
std      4.548126           0.0       0.038069           0.0        0.500344          26.656742    4.641006
min      0.080000           0.0       0.000000           0.0        0.000000          55.340000   10.300000
25%      2.000000           0.0       0.000000           0.0        0.000000          76.120000   17.200000
50%      7.000000           0.0       0.000000           0.0        1.000000          89.785000   19.000000
75%     11.000000           0.0       0.000000           0.0        1.000000         108.630000   22.100000
max     17.000000           0.0       1.000000           0.0        1.000000         219.810000   41.700000

Cluster 2:
              age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level         bmi
count  650.000000    650.000000          650.0    650.000000      650.000000         650.000000  650.000000
mean    59.880000      0.432308            0.0      0.875385        0.504615         116.578185   30.906462
std     16.362451      0.495778            0.0      0.330537        0.500364          54.396682    7.419361
min     14.000000      0.000000            0.0      0.000000        0.000000          55.230000   11.300000
25%     49.000000      0.000000            0.0      1.000000        0.000000          77.075000   26.200000
50%     62.000000      0.000000            0.0      1.000000        1.000000          93.975000   29.500000
75%     74.000000      1.000000            0.0      1.000000        1.000000         147.162500   34.400000
max     82.000000      1.000000            0.0      1.000000        1.000000         267.600000   92.000000

Cluster 3:
              age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level         bmi
count  712.000000    712.000000          712.0    712.000000      712.000000         712.000000  712.000000
mean    45.358146      0.091292            0.0      0.762640        0.540730         106.004157   30.448596
std     15.659154      0.288227            0.0      0.425764        0.498689          45.161884    7.140995
min     13.000000      0.000000            0.0      0.000000        0.000000          55.320000   15.700000
25%     32.000000      0.000000            0.0      1.000000        0.000000          76.647500   25.675000
50%     45.000000      0.000000            0.0      1.000000        1.000000          93.200000   28.700000
75%     56.000000      0.000000            0.0      1.000000        1.000000         113.457500   34.100000
max     82.000000      1.000000            0.0      1.000000        1.000000         267.610000   78.000000

Cluster 4:
              age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level         bmi
count  761.000000    761.000000          761.0    761.000000      761.000000         761.000000  761.000000
mean    53.745072      0.086728            0.0      0.837057        0.511170         108.854770   30.710118
std     16.862000      0.281621            0.0      0.369557        0.500204          49.260456    6.787753
min     13.000000      0.000000            0.0      0.000000        0.000000          55.270000   15.000000
25%     41.000000      0.000000            0.0      1.000000        0.000000          77.520000   26.200000
50%     55.000000      0.000000            0.0      1.000000        1.000000          90.960000   29.400000
75%     67.000000      0.000000            0.0      1.000000        1.000000         113.570000   34.300000
max     82.000000      1.000000            0.0      1.000000        1.000000         267.760000   56.100000

Cluster 5:
        age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level   bmi
count   1.0           1.0            1.0           1.0             1.0               1.00   1.0
mean   26.0           0.0            0.0           0.0             0.0             143.33  22.4
std     NaN           NaN            NaN           NaN             NaN                NaN   NaN
min    26.0           0.0            0.0           0.0             0.0             143.33  22.4
25%    26.0           0.0            0.0           0.0             0.0             143.33  22.4
50%    26.0           0.0            0.0           0.0             0.0             143.33  22.4
75%    26.0           0.0            0.0           0.0             0.0             143.33  22.4
max    26.0           0.0            0.0           0.0             0.0             143.33  22.4

Cluster 6:
              age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level         bmi
count  275.000000    275.000000          275.0    275.000000      275.000000         275.000000  275.000000
mean    68.429091      0.232727            1.0      0.887273        0.512727         137.087600   30.053818
std     10.923317      0.423340            0.0      0.316836        0.500749          62.778395    5.023668
min     28.000000      0.000000            1.0      0.000000        0.000000          56.310000   19.100000
25%     61.000000      0.000000            1.0      1.000000        0.000000          83.745000   27.300000
50%     71.000000      0.000000            1.0      1.000000        1.000000         106.680000   28.900000
75%     78.000000      0.000000            1.0      1.000000        1.000000         201.985000   32.500000
max     82.000000      1.000000            1.0      1.000000        1.000000         271.740000   54.700000

Cluster 7:
             age  hypertension  heart_disease  ever_married  Residence_type  avg_glucose_level        bmi
count  22.000000          22.0           22.0          22.0       22.000000          22.000000  22.000000
mean   16.181818           0.0            0.0           0.0        0.681818          96.042727  25.545455
std     2.342899           0.0            0.0           0.0        0.476731          28.697132   7.441757
min    13.000000           0.0            0.0           0.0        0.000000          59.990000  14.600000
25%    14.250000           0.0            0.0           0.0        0.000000          78.457500  20.975000
50%    16.000000           0.0            0.0           0.0        1.000000          86.020000  23.150000
75%    17.000000           0.0            0.0           0.0        1.000000         112.807500  28.350000
max    23.000000           0.0            0.0           0.0        1.000000         161.280000  44.900000

```

---

### Result analysis 

#### Cluster 0 - "Adults, Generally Healthy" - 1999 people - 3.10% of this cluster had a stroke
(16.87% of all stroke patients in the dataset)
- Average age: 42
- Very low blood pressure (1.1%)
- No heart disease
- 68% married
- Live in - Balanced urban/rural (50/50)
- BMI: 29.7 (moderately overweight)
- Average glucose level: 101 (near normal but above normal)

#### Cluster 1 - Children - 690 people - 0.29% of this cluster had a stroke
- Average age: ~7
- No high blood pressure or heart disease
- All single
- BMI (~20) healthy
- Glucose level ~94 Great.

#### Cluster 2 – Adults, Some Hypertension – 650 people – 9.38% of this cluster experienced a stroke
- Age: ~60
- 43% have hypertension
- No heart disease
- 87% married
- BMI ~30.9 (critically overweight)
- Glucose: High (116) – High risk of type 2 diabetes.

#### Cluster 3 – Adults, Not so healthy – 712 people – 3.51% of this cluster experienced a stroke
- Age: 45
- 9% have hypertension
- No heart disease
- 76% married
- BMI: 30.4 (critically overweight)
- Glucose: 106- High risk of type 2 diabetes.

#### Cluster 4 – Adults, Menopause – 761 - 6.83% of this cluster experienced a stroke
- Age: 54
- 8% with hypertension
- No heart disease
- 83% married
- BMI ~30.7 (critically overweight)
- , glucose ~108 High risk of type 2 diabetes.

#### Cluster 5 – "Exception (single point)" Only one person can be ignored.

#### Cluster 6 – “Elderly with Heart Disease” – 275 people – 17.09% of this cluster experienced a stroke
- Average age: 68
- 100% have heart disease
- 88% married
- 23% have high blood pressure
- BMI 30~ (critically overweight)
- Glucose ~137 definitely diabetic

#### Cluster 7 – “Teenagers” – 22 people – 0% experienced a stroke
- Average age: 16
- Healthy (no high blood pressure/heart disease)
- Single
- BMI (~25.5) – (moderately overweight)


The groups show the clusters that the algorithm found. It is easy to see that as you get older, there is indeed a sharp increase in the chance of having a stroke. According to the distribution of the residential area, in almost every cluster there are 50% who live in an urban area and 50% who live in a rural area.

- According to the data, there may also be a connection between heart disease and diabetes and a higher risk of stroke.
According to Appendix 3, approximately 3.1% of the adult population experiences a stroke, and if so, if I were a doctor, I would "keep an eye" on clusters 2,3,4,6 (less on 3), since they show the strongest symptoms of stroke above the general population average.


### I took this exercise a step further and said that I would try different other ways to correctly catalog clusters:

For example, remove insignificant data, use a different method for grouping clusters (Agglomerative Clustering).
Even trying to include different scales to test the quality of the clusters, but in the end no new and significant result was given.

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Agglomerative%20Clustering%20(k%3D5)%20-%20PCA%20Projection.png" width="550">
</p>

**This indicates that this approach (clustering) of data mining on medical data "does not work very well"** meaning that the clusters are not able to be separated clearly enough (according to Silhouette Score), it is worth trying to add data from other ages and healthier people (normal BMI/normal blood pressure/normal glucose level) because currently most of the data is not very "positive" in terms of people's health level.


However, we were able to achieve quite impressive results regarding medical insights from clustering into different clusters, but since the clusters overlap quite a bit, I would consider adding more data to the study that are essentially different from all the other clusters (for example, healthy adults and elderly people and people who are underweight/normal weight). And I would consider turning to other models (statisticians or AI, ML, Deep Learning).

---

## Deep learning nerual networks

### I wanted to see how the deep nerual network model would work against this dataset soooo...

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/neural%20network%20gif.gif" width="550">
</p>

### Explanation on the network architecture:
**My network architecture was determined dynamically (we will detail it later here):**

This is a fully connected neural network, meaning that each neuron in a layer is connected to all neurons in the next layer, with information flowing in one direction (Feedforward network) - from the input layer, through one or more hidden layers, and to the final output. (At the end of the explanation we will say what the final parameters that were chosen are).
The network starts with an input layer that receives the features (columns) of the data set. Each feature enters its own input neuron.
The data is then passed through 1 to 3 hidden layers (the exact number is automatically selected by the hyperparameter mechanism that I implemented - we will see later what the number was chosen).

In each hidden layer, the network takes a weighted sum of the inputs, adds a bias, and applies a transformation (i.e., activates the activation function – I chose to use ReLU) before passing it forward.
Finally, the data reaches the output layer, which has only one neuron – this neuron gives a number between 0 and 1, representing the probability of a stroke.

What are the parameters I am trying to dynamically optimize during the learning process?
- The number of hidden layers (between 1 and 3).
- The number of neurons (also called "units") in each hidden layer – between 32 and 128 per layer.
- The optimizer used to update weights during training (Adam, RMSprop, or SGD).
- The learning rate (lr – learning rate), which controls the speed of the model's update while it is learning (how much each gradient calculation affects).

Further explanation about the activation function:
Within each hidden layer, the network uses the ReLU activation function and in the input layer I use the Sigmoid function.

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/Sigmoid%26Relu.png" width="550">
</p>

#### On the right is ReLU, on the left is sigmoid.
- Both are well-known activation functions that turn the final result into a number between 0 and 1.
  Therefore, they are suitable for binary classification, in our case predicting whether someone has had a stroke is a question of yes or no (1 or 0 - respectively).

- In ReLU, its advantage is that neurons will "fire" only if the input is positive, which helps the network learn complex patterns efficiently - the neurons actually fire only if they are "necessary" in the calculation, and do not remove "small weights" from             calculations that are not needed - which increases the difficulty of the calculation.

- Sigmoid The goal in the end is to estimate our "chance" of having a stroke in the end on a scale between 0 and 1 so that we can calculate the probability of a stroke as the model estimates it and give a percentage of the model's confidence.



### Output and Learning:

The model is trained to minimize the loss function, which in this case I defined as binary_crossentropy.

Since we are predicting whether a patient has had a stroke (1) or not (0) –
a binary classification problem – this is one of the cost functions that fits the problem.

It works quite simply, after the model predicts a probability (e.g. 0.9 for stroke or 0.1 for no stroke) the function compares it to the actual answer (1 or 0).
If the "guess" is very far off, the penalty (error) will be large. If it is close, the penalty will be small.
The model tries to minimize this error by adjusting its internal weights (and the other hyperparameters I mentioned).
During training, the model also tracks Accuracy, Precision, Recall and F1 score to help measure its performance, especially in the minority class (stroke cases).

In terms of Batch size we use the default Tensor flow 32 patient records at a time (per batch 32 patients) and make predictions for them,
it compares these predictions to the correct answers, calculates the average error, updates the model weights.

And this process repeats again and again, for all the data as a quantity The EPOCS – which I defined. (A full cycle of data transfer is called an epoch) – I defined 32 such cycles.

### Best hyperparameters for an example run:

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/HyperParameters.png" width="550">
</p>

#### In the end, here are the hyperparameters I decided on (according to optimization algorithms tested).
2 hidden layers, 96 parameters in the first and second layers, "ADAM" type optimization algorithm and a very low LR of about 0.0006.

### Results

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/EvaluationNN.png">
</p>

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/NN-ConfuisionMatrix-Threshold-0.5.png" width="450">
</p>

<p align = 'center'>
<img src="https://github.com/idogut3/20595-DataMining-Discovering-Analyzing-Predicting-Stroke-usingDataScienceShenanigans-ResearchProject/blob/main/images%26gifs/NetworkEvaluationSum.png" width="450">
</p>


#### As you can see, the error was very large at first, but as more and more epochs passed, the error decreased as expected.

### Model errors:
#### The model made several mistakes – including false positives
False Positives (stroke expected, but no stroke actually occurred):
In all five examples, the model predicted that the person would experience a stroke, but in reality, they did not experience a stroke. These people had stroke-like characteristics such as:
- Higher glucose levels (e.g., 208.17 mg/dL – diabetes in one case).
- Married and older (e.g., 67 years old).
- Past smoking.

These are all factors that increase the risk of stroke, so the model took the precaution of classifying them as strokes. But in these specific cases, no actual strokes occurred, which shows that the model can sometimes overreact to risk patterns, likely due to a bias to avoid missing actual strokes (i.e., it was optimized for High Recall - which is a good thing, especially in medical cases. "Better safe than sorry in medical cases is the exact phrase to say).

Within the top error results, no false negatives were reported:
- That is, in this part of the data, the model successfully identified all real strokes. This is a very good sign! And especially important in healthcare, where missing a real stroke is much worse than triggering a false alarm.


### Summary on the results for the deep neural network

The network has proven itself in its ability to learn the risks of stroke in the above dataset, things that are important to pay attention to in the future for further work on the Shabbat identification study:
- More data is needed! 5000 samples is quite low (even more so considering that the amount of strokes in the dataset is even lower) especially for life-threatening diseases. In addition, I would add data related to the patients' brain activity (clinical / non-behavioral information of the patient) since stroke is still a disease that is closely related to the brain and its function. Information about this data can even help us try to understand what the biological factors are under microscopic observation that cause stroke.

- I would also test the network on data outside the dataset (to check that there is no overfitting to this dataset (especially since it is also a dataset with a lot of obese / diabetic / etc. patients and there is not enough information about healthy people).

- In general, I am very satisfied with the results of the network, it reached impressive results of good recall

The model It successfully detected 72% of actual strokes, which is very important in medical settings - this means that it catches most patients who are really at risk.
It has problems such as very low accuracy for strokes:
When it does predict strokes - it is wrong 87% of the time. That is, most of the "stroke" predictions are false alarms. This can lead to worry or unnecessary tests in real life.
The model prefers to catch as many strokes as possible - a kind of compromise, even at the cost of accuracy. And this makes sense for stroke prediction - it is better to over-alert than to miss a real case (this is how I trained it), but the accuracy still needs to improve to be practical in a real environment.

#### The model is promising in terms of recall (it almost never misses real strokes), but its accuracy is too low to be used on its own for diagnosis. It could be useful as an early warning tool, but would need support from additional tests or a review Experts.

---


## Final summary on all of the project

- We reviewed various methods for predicting/predicting/and statistical analyses of the chance of stroke:

In the framework of the work, we investigated the subject of stroke using a multi-stage approach that combines data mining techniques, machine learning, and model optimization.

The data were analyzed from an existing database (Kaggle Stroke Prediction Dataset), with the main goal being to identify patterns, characterize risk groups, and finally predict future stroke cases as accurately as possible – to support future clinical decisions.

### We worked in stages, sometimes in parallel and sometimes in order:

#### 1. The first stage was the initial cleaning and analysis of the data:
The first stage involved data processing: replacing missing values and categories such as "Unknown", handling outliers, and converting columns to numeric representation (such as One-Hot Encoding). Distributions of various characteristics were examined among patients with and without stroke – for example, age, smoking, marital status, underlying diseases, and more – to understand which of them may be associated with stroke risk.

#### 2. The second stage – discovering insights with data mining techniques
Using the Association Rules (FP Growth) algorithm, recurring rules were identified among the patients, such as "most smokers do not suffer from stroke" or "patients with certain underlying diseases are at high risk". Next, we performed clustering using K-Means and divided the population into similar groups. For example, we saw that stroke tends to concentrate in a few specific clusters, while in others it almost does not appear – which indicated a potential For early triage of patients.


#### 3. Step Three – Building Stroke Prediction Models
We went through prediction stages using machine learning models: such as
Random Forest and XGBoost
And finally a deep neural network. (Deep Neural Network)


**In order to improve performance, SMOTE was used** to balance the distribution of groups (due to the small number of stroke cases). Sometimes we also used splitting into subgroups to feed the model with a K random samples approach and more...
The neural network was tuned using Keras Tuner (Hyperband): where parameters such as the number of layers, number of neurons in each layer, learning rate and type of optimizer were selected.

#### 4. Step Four – Performance Analysis and Error Evaluation
The models were evaluated using metrics such as Precision, Recall, Accuracy and F1 Score, with an emphasis on the model's ability not to miss stroke cases (i.e. high sensitivity to stroke cases Real
The model was able to detect 72% of stroke cases (deep learning), but sometimes gave false positives with relatively low precision.

**We analyzed model errors:** there were several cases in which the model predicted stroke for healthy patients (False Positives), usually at the statistical limit of the decision – for example, patients with advanced age or high sugar levels, but without a corresponding medical background.

#### General summary and insights
The work indicates that a combination of statistical analyses, clustering, various ML algorithms and deep learning is able to yield useful clinical insights and even predict stroke with a good level of accuracy. However, the balance of the model must be improved, the decision threshold calibrated and the quality of the information improved to prevent the burden of false positives that doctors rely on. It is equally important to adapt the models to the reality of clinical medical knowledge, so that they can actually help doctors and patients alike in the real world.


---


## Appendices and sources:

### Appendix 1:
Linoy, O. (2022, October 30). Normal blood sugar level - What is a normal blood sugar level? All information - Olenik Law Firm. Olenik Linoy - National Insurance Specialist Law Firm. https://www.oleniklaw.co.il/%D7%A8%D7%9E%D7%AA-%D7%A1%D7%95%D7%9B%D7%A8-%D7%AA%D7%A7%D7%99%D7%A0%D7%94/

BMI for men: Body mass index: Ramsay Health Care UK. Body Mass Index | Ramsay Health Care UK. (n.d.). https://www.ramsayhealth.co.uk/treatments/weight-loss-surgery/bmi/bmi-for-men

### Appendix 2:
BMI Chart for kids in PDF - download. Template.net. (n.d.). https://www.template.net/editable/94377/bmi-chart-for-kids

### Appendix 3:
Walker, M. (2024, July 5). Stroke demographics: Who is most at risk and why?. Franciscan Missionaries of Our Lady Health System.
https://health.fmolhs.org/body/neuroscience/stroke-demographics-who-is-most-at-risk-and-why/




