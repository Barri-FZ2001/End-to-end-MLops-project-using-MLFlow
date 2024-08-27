import mlflow 
from mlflow.tracking import MlflowClient
import requests
"""
# Initialize the MLflow client
client=MlflowClient()

# Get the latest run ID
run_id='454d5ff819c14e15b7deb3478930adb7'
print(client.list_artifacts(run_id=run_id))

logged_model = 'runs:/'+str(run_id)+'/classification model' # specify model from run id
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)
#prediction = loaded_model.predict('example of input to predict using the loaded model')
#print(prediction)
"""

#Model serving 
# mlflow models serve --model-uri 'runs:/454d5ff819c14e15b7deb3478930adb7/classification model' --no-conda --port 4000

# Send request to the model served at port number 4000
"""
request_data={
    'dataframe_records':[[3,4,5,6],[1,2,1,2]]
}
endpoint='http://localhost:4000/invocations'
response = requests.post(endpoint, json=request_data)

print(response.json())
"""


client=MlflowClient()
client.transition_model_version_stage(
    name="iris_classifier_v1.0",
    version=1,
    stage="Production"
)
