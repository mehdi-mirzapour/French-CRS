import json
import pandas as pd
import numpy as np
import random
import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn import metrics



"""

This class is responsible for splitting the dataset based on the file name property
and NOT the rows in the dataset. 

"""

class dataset_splitter():

    # initiator for getting the main, training and test dataset file names
    def __init__(self, input_file, train_file, test_file, split_config_json="./split_config.json"):
        self.input_dataframe = pd.read_excel(input_file, index_col=0)
        self.train_file = train_file
        self.test_file = test_file
        self.split_config_json = split_config_json


    #  The splitter divides the dataset based on the file name property in the dataset.
    def dataset_splitter_by_file(self, lower_rate=0.2,upper_rate=0.21, files_num=60):

        rate = 0

        # while not(rate >= lower_rate and rate < upper_rate):
        sub_corpus_files = self.input_dataframe["FILE_NAME"].unique().tolist()
        sub_corpus_files_test = random.sample(sub_corpus_files, files_num)
        sub_corpus_files_train = [
            file for file in sub_corpus_files if file not in sub_corpus_files_test]

        self.train_dataframe = self.input_dataframe[self.input_dataframe["FILE_NAME"].isin(
            sub_corpus_files_train)]
        self.test_dataframe = self.input_dataframe[self.input_dataframe["FILE_NAME"].isin(
            sub_corpus_files_test)]

        rate = self.test_dataframe.shape[0]/self.train_dataframe.shape[0]

        print (rate)

        self.train_dataframe.to_excel(self.train_file)
        self.test_dataframe.to_excel(self.test_file)

        self.sub_corpus_files_test = sub_corpus_files_test
        self.sub_corpus_files_train = sub_corpus_files_train

        print("Created with", rate,"rate")

        with open(self.split_config_json, 'w') as outfile:
            json.dump({
                "sub_corpus_files_train": self.sub_corpus_files_train,
                "sub_corpus_files_test": self.sub_corpus_files_test
            },
                outfile)

    def dataset_splitter_by_json_config(self):

        with open(self.split_config_json) as json_file:
            config_json = json.load(json_file)

        sub_corpus_files_train = config_json["sub_corpus_files_train"]
        sub_corpus_files_test = config_json["sub_corpus_files_test"]
        
        self.train_dataframe = self.input_dataframe[self.input_dataframe["FILE_NAME"].isin(
            sub_corpus_files_train)]
        self.test_dataframe = self.input_dataframe[self.input_dataframe["FILE_NAME"].isin(
            sub_corpus_files_test)]

        rate = self.test_dataframe.shape[0]/self.train_dataframe.shape[0]


        self.train_dataframe.to_excel(self.train_file)
        self.test_dataframe.to_excel(self.test_file)

        self.sub_corpus_files_test = sub_corpus_files_test
        self.sub_corpus_files_train = sub_corpus_files_train

        print("Created with", rate, "rate")




"""

This class is responsible for training the models. It includes different methods
for performing. 

"""


class model_trainer():

    # initiator for getting the main, training and test dataset file names

    def __init__(self, train_file, test_file, output_file, train_column, test_column):
        self.train_df = pd.read_excel(train_file, index_col=0)
        self.test_df = pd.read_excel(test_file, index_col=0)
        self.output_file = output_file
        self.train_column = train_column
        self.test_column = test_column
        self.values_list = [
                            'AMOUNT', 'DEF_DEM', 'DEF_SPLE', 'EVENT',
                            'EXPL', 'F', 'FONC', 'INDEF', 'LOC', 'M', 'N',
                            'NO', 'ORG', 'PERS', 'PL', 'PR', 'PROD', 'SG',
                            'TIME', 'UNK', 'VOID', 'YES'
                            ]
        self.values_list_for_replace = range(1, 23)
        self.columns_drop_list = ['m1_DEF', 'm2_DEF', 'ID_DEF']
        self.threshold=0.5

    def convert_columns_to_numeric(self):

        self.train_df_converted = self.train_df.drop(
            self.columns_drop_list, axis=1)
        self.train_df_converted.fillna("VOID", inplace=True)
        self.train_df_converted = self.train_df_converted.replace(
                                     self.values_list,self.values_list_for_replace)

        self.test_df_converted = self.test_df.drop(
            self.columns_drop_list, axis=1)
        self.test_df_converted.fillna("VOID", inplace=True)
        self.test_df_converted = self.test_df_converted.replace(
                                     self.values_list, self.values_list_for_replace)
    


    def train_model_random_forest(self, model_name ,**kwargs):

        #Create a random forest Classifier
        clf = RandomForestRegressor(**kwargs)

        #Train the model using the training sets
        clf.fit(self.train_df_converted.iloc[:, 0:-1],
                self.train_df_converted.iloc[:, -1])

        y_pred = clf.predict(self.test_df_converted.iloc[:, 0:-1])

        binary_y_pred = []

        for x in y_pred:
            if x >= self.threshold:
                binary_y_pred.append(1)
            else:
                binary_y_pred.append(0)

        self.test_df["Prediction"] = y_pred
        self.test_df.to_excel(self.output_file)

        self.precision_score_pos= metrics.precision_score(
            self.test_df[self.test_column], binary_y_pred)
        self.recall_score_pos = metrics.recall_score(
            self.test_df[self.test_column], binary_y_pred)
        self.f1_score_pos = metrics.f1_score(
            self.test_df[self.test_column], binary_y_pred)

        self.precision_score_neg = metrics.precision_score(
            self.test_df[self.test_column], binary_y_pred, pos_label=False)
        self.recall_score_neg = metrics.recall_score(
            self.test_df[self.test_column], binary_y_pred, pos_label=False)
        self.f1_score_neg = metrics.f1_score(
            self.test_df[self.test_column], binary_y_pred, pos_label=False)

        joblib.dump(clf, model_name)

        return(
            {
            "precision_score_pos": self.precision_score_pos,
             "recall_score_pos": self.recall_score_pos,
             "f1_score_pos": self.f1_score_pos,
             "precision_score_neg": self.precision_score_neg,
             "recall_score_neg": self.recall_score_neg,
             "f1_score_neg": self.f1_score_neg
            }
        )


class model_tester():

    # initiator for getting the main, training and test dataset file names

    def __init__(self, model_name, column_gold, column_outcome, columns_drop_list,threshold):
        self.model_name = model_name
        self.column_gold = column_gold
        self.column_outcome = column_outcome
        self.values_list = [
            'AMOUNT', 'DEF_DEM', 'DEF_SPLE', 'EVENT',
            'EXPL', 'F', 'FONC', 'INDEF', 'LOC', 'M', 'N',
            'NO', 'ORG', 'PERS', 'PL', 'PR', 'PROD', 'SG',
            'TIME', 'UNK', 'VOID', 'YES'
        ]
        self.values_list_for_replace = range(1, 23)
        self.columns_drop_list = columns_drop_list
        self.threshold = threshold


    def apply_model_to_row(self, coref_features):

        model = joblib.load(self.model_name)
        
        df_coref_relations = pd.DataFrame.from_dict(coref_features, orient='index')

        self.input_dataframe = df_coref_relations.drop(self.columns_drop_list, axis=0)

        self.input_dataframe.replace(to_replace={"NULL": "VOID"}, inplace=True)
        self.input_dataframe.fillna("VOID", inplace=True)
        self.input_dataframe = self.input_dataframe.replace(
                                self.values_list, self.values_list_for_replace)

        y_pred = model.predict(np.array(self.input_dataframe[0:-1]).reshape(1, -1))

        return(y_pred[0])










