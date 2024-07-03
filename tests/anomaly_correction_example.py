import pandas as pd
import os
from src.anomaly_correction import AnomalyCorrection


def main(data_path='..\\datasets\\sample_data_preprocessed.csv',
         model_path='..\\models\\anomaly_correction_model.pkl'):
    """ Author: Omar
    Example of the AnomalyCorrection class
    """
    """ 1/2)
    Your colleague has given you a clean dataset with examples
    of normal data-points and anomalies. He wants you to train 
    a model that is able to classify the anomalies. 
    """
    # get data
    df_x = pd.read_csv(data_path)
    df_x = df_x.sample(frac=1).reset_index(drop=True)
    df_x['Annual_Premium'] = df_x['Annual_Premium'].apply(lambda x: round(x))
    df_x['Age'] = df_x['Age'].apply(lambda x: round(x, -1))
    df_y = df_x.copy()['Response_1']
    del df_x['Response_1']

    # train/load the model

    if os.path.exists(model_path):
        ac = AnomalyCorrection(model_name=DEFAULT_MODEL_NAME)

    """ 2/2)
    Then, your colleague passes you examples of data-points that
    a system classified as  anomalies. You want to correct these 
    anomalies, so that they look more like normal data-points. 
    This will help your colleague in understanding why the system
    classified them as anomalies.  
    """
    ac = AnomalyCorrection()

