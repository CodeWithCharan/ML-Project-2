## ML-Project-2

### Software and Tools Requirements

1. [GitHub Account](https://github.com/)

2. [Python](https://www.python.org/downloads/)

3. Jupyter Notebook

	```
	pip install jupyter notebook
	```

4. Modules
	- pandas
	```
	pip install pandas
	```

	- SKlearn
	```
	pip install scikit-learn
	```
# Documentation

## Titanic - Machine Learning from Disaster
### Project Overview:
  The goal of this project is to create a machine learning model that predicts which passengers survived the Titanic shipwreck. The dataset provided by Kaggle includes passenger information such as their age, gender, class, and ticket details. The objective is to use this data to train a model that can accurately predict whether a passenger survived or not.

### Dataset:
The dataset provided by the competition includes two CSV files: train.csv and test.csv. The train.csv file is used to train the machine learning model, while the test.csv file is used to test the model's predictions. The dataset includes passenger information such as PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

### Evaluation Metric:
The competition uses accuracy as the evaluation metric to determine the performance of the machine learning model. The accuracy score is calculated as the percentage of correctly predicted passenger survival outcomes.

### Solution Approach: 
The code provided uses the following models to predict passenger survival outcomes:
- a. Logistic Regression: The code uses a logistic regression model to predict passenger survival outcomes. Logistic regression is a statistical method that uses a logistic function to model a binary dependent variable, such as passenger survival outcomes in this case. The Scikit-learn library is used to implement the logistic regression model.
- b. Label Encoding: The code uses label encoding to convert categorical variables to numerical variables. Label encoding is a method of assigning numerical labels to categorical variables in such a way that each label has a unique numerical value. The Scikit-learn library is used to implement the label encoding method.

### Cleaning the Data: 
The clean() function is defined to clean the data by dropping unnecessary columns, filling missing values, and converting categorical variables to numerical.

### Model Training and Testing:
The Scikit-learn library is used for data preprocessing, model training, and testing. The data is split into training and testing sets using the train_test_split() function. The accuracy of the model is calculated using the accuracy_score() function.

### Submission:
The final step of the competition involves submitting the predicted passenger survival outcomes in a CSV file format, which should include the PassengerId and Survived columns. The submission file should be named "gender_submission.csv".

### Conclusion:
Our analysis of the Titanic dataset and the machine learning model we developed suggest that a passenger's gender, age, and class were the most important factors in determining their likelihood of survival. We also identified several limitations of our model, such as the small sample size and potential biases in the data. To improve our model's performance, we could consider using more advanced machine learning techniques.

### References:
We used the following resources and references during the project:
- Kaggle Titanic: Machine Learning from Disaster competition page (https://www.kaggle.com/competitions/titanic/overview)
