import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, log_loss
import pandas as pd
import mlflow
from mlflow import set_experiment, set_tag, log_artifact,log_param,log_metric, log_params
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
iris_data=load_iris()
X=pd.DataFrame(iris_data.data, columns=iris_data.feature_names) 
y=iris_data.target

# Split the dataset into training and testing sets
x_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Set up MLflow experiment
experiment_name=set_experiment('Classification on iris dataset using logistic regression')
run_name='iris_classifier_v1.0'
with mlflow.start_run(run_name=run_name) as run:
    # Define the model
    model=LogisticRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred,average='micro')
    #roc_auc=roc_auc_score(y_test,y_pred,average='micro')
    precision=precision_score(y_test,y_pred,average='micro')
    log_param('Learning rate',0.001)
    log_metric('accuracy',accuracy)
    log_metric('recall',recall)
    #log_metric('ROC_AUC',roc_auc)
    log_metric('precision_score',precision)
    coef=model.coef_
    intercept=model.intercept_
    model_params={"coef":coef, "intercept":intercept}
    log_params(model_params)
    # plot scatter plots:
    sns.scatterplot(x=X['sepal length (cm)'], y=X['sepal width (cm)'], hue=y, palette="deep")
    plt.savefig('scatter.png')
    log_artifact('scatter.png')
    
    mlflow.sklearn.log_model(model,'classification model')
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))



