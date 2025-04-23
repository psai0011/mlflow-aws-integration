import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.models.signature import infer_signature

import logging

logging.basicConfig(level=logging.WARN)
logging = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


if __name__ == "__main__":

    ## first part is data ingetsion Reading the dataset - wine quality dataset

    

    # Setup basic logger (optional)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        data = pd.read_csv("winequality-red.csv")
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error while loading dataset: {e}")

    ## spilt the data in train and test 

    train, test = train_test_split(data)

    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train.drop(['quality'])
    test_y= test.drop(['quality'])

    alpha = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    ll_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x,train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)


        print("Elastic model(alpha = {:f}), l1_ratio{:f}".format(alpha,l1_ratio))

        print("RMSE: %s"%rmse)
        print("MAE:%s"%mae)
        print("R2: %s" %r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("rmse", rmse)
        mlflow.log_param("r2", r2)
        mlflow.log_param("mae", mae)
        

        ## for the remote sever AWS we need to do the setup

        remote_server_uri = ""
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(
                lr, 'model', registerd_model_name = "ElasticnetWineModel"

            )
        else:
            mlflow.sklearn.log_model(lr, "model")