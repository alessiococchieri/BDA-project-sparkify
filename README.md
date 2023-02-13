# BDA project - Sparkify

- **Student:** Alessio Cocchieri
- **Matricola:** 0001046067
- **Email:** alessio.cocchieri2@studio.unibo.it

# Apache Spark Cluster on Docker
This project is executed in Apache Spark cluster mode with a JupyterLab interface built on top of Docker.

## Cluster overview 
| Application              | URL | Description  |
|-------------------|----|----|
| JuyterLab | localhost:8888| Cluster interface with built-in Jupyter notebooks | 
| Spark Driver | localhost:4040| Spark Driver web ui | 
| Spark Master | localhost:8080| Spark Master node | 
| Spark Worker | localhost:8081| Spark Worker node with 5 core and 5GB of memory (default)| 

## Cluster execution
1. Install *Docker* and *Docker Compose*
2. Download the docker-compose file
3. Edit the file with the preferred setup
4. Run the cluster: <code> docker-compose up </code>

# Dataset
The dataset consists of **286500 patterns** and **18 features** related to typical user activity on Sparkify platform.
Sparkify is a music streaming service. Just like other famous streaming services (e.g. Spotify), users can choose free tier subscription with ads or paid tier without ads. They are free to upgrade, downgrade, or cancel their subscription plan.
 

# Task
The tasks consist of a binary classification problem where in this case the objective is to identify clients more likely to churn. If we can identify which users are at-risk to churn, then the business can take action and potentially make them stay.

# Data overview 
The main issue of the current task is related to the high imbalanced class ratio. From the data it is possible to identify 225 **distinct users**. As shown in the figure below, the positive class (churn) is only about 23% of the total data points. 

<p align="center">
  <img src="img/first.png" height = 300 width = 350px>
</p>

# Evaluation metric
Because of the high imbalanced class ratio in the data, *accuracy* is inappropriate as evaluation metric. The main reason is that the overwhelming number of examples from the majority class will overwhelm the number of examples in the minority class, meaning that even unskillful models can achieve accuracy scores of 90 percent, or 99 percent, depending on how severe the class imbalance happens to be.

An alternative to using classification accuracy is to use precision and recall metrics:
- Precision quantifies the number of positive class predictions that actually belong to the positive class.
- Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.
- F1-score provides a single score that balances both the concerns of precision and recall in one number.

Since our goal is to predict the users more likely to churn (postive class), **F1-score of positive class** has been considerend as main reference to select the best model for the current problem. 

At the same time also AUC-PR turned out to be good indicator to evaluate the performances of the models. When working with imbalanced datasets in binary classification problems, it's often more appropriate to use the AUC-PR as a performance metric rather than the AUC-ROC. It provides a more accurate representation of the model's ability to correctly classify the minority class.

# Feature analysis
The 18 features can be divided into 3 different levels:

1. User-level information
    - `userId (string)`: user’s id
    - `firstName (string)`: user’s first name
    - `lastName (string)`: user’s last name
    - `gender (string)`: user’s gender, 2 categories (M and F)
    - `location (string)`: user’s location
    - `userAgent (string)`: agent (browser) used by the user
    - `registration (int)`: user’s registration timestamp
    - `level (string)`: subscription level, 2 categories (free and paid)

2. Log-specific information
    - `ts (int)`: timestamp of the log
    - `page (string)`: type of interaction associated with the page (NextSong, Home, Login, Cancellation Confirmation, etc.)
    - `auth (string)`: authentication level, 4 categories (Logged In, Logged Out, Cancelled, Guest)
    - `sessionId (int)`: a session id
    - `itemInSession (int)`: log count in the session
    - `method (string)`: HTTP request method, 2 categories (GET and PUT)
    - `status (int)`: HTTP status code, 3 categories (200, 307 and 404)

3. Song-level information
    - `song (string)`: song name
    - `artist (string)`: artist name
    - `length (double)`: song’s length in seconds

# Data preprocessing
The first step of our analysis consists of preprocessing the data: 
- Map the target class: not churn --> 0 and churn --> 1
- Map the gender: Male --> 0, Female --> 1
- Convert time stamp format to date time format
- Check for nulls

# Exploratory Data Analysis (EDA)
EDA was used to analyze and investigate the data and summarize its main characteristics. It helps determine how best to manipulate data sources to get the answers we need. The most interesting insights discovered are the following: 
- In addition to playing music, *Thumbs Up* and *Add to playlist* turned out to be the activities most performed by the users
- The are more male users than female users
- Most common location are Los Angeles and New York
- Premium users are more prone to churn
- The date range in the dataset is 2018-09-30 to 2018-12-02. We can see paid tier and free tier has similar numbers of user sessions and distinct user at the beginning, then both metrics increased over time for the paid tier as the metrics decreased for free users.
- Most churns happens within 100 days after registration.

# Features engineering
It is the most crucial step in the current task. The initial 18 features provided by the dataset need to be transformed to be actually useful for our task. Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. 
Based on the analysis carried out still now thanks to EDA, we are ready to extract the features that actually will allow us to understand clients more prone to churn.

In this regards, I decided to extract the following 11 features:
- `gender`: male or female
- `days_registered`: number of days the user is registered
- `paid_user`: free account or premium account
- `downgraded`: has the user ever downgraded from premium to free?
- `artists`: average number of artits listened per session by the user
- `songs`: average number of songs listened per session by the user 
- `length`: average second listened of songs per session by the user
- `interactions`: average proactive operations performed by the user per session 
- `thumbs_down`: average thumbs down released by the user per session
- `total_session`: total number of session 
- `session_gap`: average time between each session and the pevious one

# Features distribution
The distibution of the extracted features is shown in the figure below:
<p align="center">
  <img src="img/features-distr.png" height = 500 width = 500px>
</p>

# Correlation between features (Feature selection)
Another relevant point which could affect the classification performances is the correlation among features: the presence of strongly correlated features may lead to a decline in the performances of some classification algorithms which assume that the predictors are all independent. Another benefit from spotting correlation among features is that the same information may be encoded with less attributes, and this could lead to simpler final models. The reduction of the dimensions of the feature vectors can make the model more trustable and stable, if the discarded dimensions don’t affect significantly the total original information.

In this case, Pearson correlation has been considered. It ranges from -1 to 1, with -1 indicating a strong negative linear relationship, 0 indicating no linear relationship, and 1 indicating a strong positive linear relationship.
The following representation shows the correlation matrix between features: since the way the matrix is constructed make it symmetric and because of the large number of features, only the heatmap of the lower diagonal matrix (the diagonal itself is excluded) is reported for a better visualization.

<p align="center">
  <img src="img/corr-matrix.png" height = 500 width = 500px>
</p>


There is no obvious strong predictor for `cancelled` except for `user_age`

`songs`, `interactions`, `thumbs_up`, `length`, `artists` are very similar according to the histograms. Although they all show high correlation with each other, this is possibly caused by the small dataset (225 users). If we have more data, we might see more variance in user behaviors. Therefore, we will **only exclude songs and artists** as they will always be similar to length.

To give a further proof of the linear dependence among `length`, `artist` and `songs`, their interactions plots are shown:
<p align="center">
  <img src="img/direct-corr.png">
</p>

For completness, also charts of non-strongly correlated features are reported. In particular:
- ***interactions*** and ***total_session*** $p$ = 0.07
- ***interactions*** and ***session_gap*** $p$ = -0.17
- ***days_registered*** and ***total_session*** $p$ = 0.20

<p align="center">
  <img src="img/no-corr.png">
</p>


# Outliers removal
There is a particular category of users to consider before making inference: the users whose total session is equal to 1. They are probably new users just entered in the application and their session_gap is indeed equal to NaN.  However, here we are trying to predict people who used to be an active user but decides to leave. For this reason, they are out of the scope of our analysis and we will exclude them from the prediction.

# Modeling (features preparation)
Features must be prepared in the right way in order to be fed to the model. In this regards, a couple of futher steps are needed:

- **Assembling**: for each reacord, its features must be assembled in a unique array. Pyspark provides the function VectorAssembler() which takes as input the features selected and the name of the column which will contain their assembling. Indeed, the function creates an array which aggregates all the features toghether.

- **Scaling**: After being assembled, the features must be scaled by exploiting the function StandardScaler(). The purpose of Standard Scaler is to transform the features so that they have a Gaussian distribution with zero mean and unit variance. This can help the model to achieve better performance by normalizing the scale of the features

# Split train and test set
```
# Split train/test data and set seed for reproducibility
train, test = data.randomSplit([0.75, 0.25], seed=42)
```
The split has been randomly generated but kept the same for all the tested approaches, in order to make evaluations statistically meaningful and provide consistent results among the different classifiers. This has been obtained by setting the random seed = 42. This specific seed allowed also to maintain the proportion of churn/non-churn clients in each of the final sets approximately the same as in the initial data. This approach is desirable every time we have a highly unbalanced dataset as in our case.

# Model fine-tuning
To find the best model and parameters, we use CrossValidator to evaluate the model performance and validate the robustness of the models. With numFolds = 3 , the CrossValidator generates 3 sets of training/testing pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular model/param selection, CrossValidator computes the average evaluation metrics for the 3 models fitted on the 3 train/test pairs.

Since our objective is to maximize the ability of the model to predict users more likely to churn, I decided to define a custom evaluator which consider as reference metric only the F1 score of the positive class 1(churn). To do so, it is necessary to extend the MulticlassClassificationEvaluator class and override the evaluate method.

```
class F1PositiveEvaluator(MulticlassClassificationEvaluator):

    def __init__(self, predictionCol="prediction", labelCol="label"):
        super(F1PositiveEvaluator, self).__init__(predictionCol=predictionCol, labelCol=labelCol)

    def evaluate(self, dataset):
        predictionAndLabels = dataset.select(self.getPredictionCol(), self.getLabelCol()).rdd.map(lambda lp: (float(lp[0]), float(lp[1])))
        metrics = MulticlassMetrics(predictionAndLabels)
        return metrics.fMeasure(1.0)
```

# Models selection
I decided to compare the performances of 4 different models:

- Logistic Regression
- Random Forest Classifier
- SVM (linear kernel)
- GBT Classifier

## Logistic regression 
Logistic Regression is a statistical method that is commonly used for binary classification. It is a simple and easy to interpret model, which outputs a probability between 0 and 1 that represents the likelihood of the positive class. The coefficients of the model represent the contribution of each feature to the predicted probability.

These are the results obtained by the model:
```
TEST SET results:
Precision positive class: 1.0
Recall positive class: 0.54
F1 positive class: 0.71
F1 macro: 0.82
AUC-PR: 0.83
```

### Coefficients
In logistic regression for binary classification, the coefficients represent the change in the log odds of the dependent variable (the binary outcome) for a one unit change in the independent variable, while holding all other variables constant.

- session_gap, total_sessiom and interactions are inversely correlated with the probability of churn, so the higher their value, the less the probability of the client to churn
- On the other hand, the length and the typology of the user turned out to be features directly correlated with the probability of being a churn.

