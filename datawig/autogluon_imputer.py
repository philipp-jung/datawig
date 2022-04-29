# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""

AutoGluon Imputer:
Imputes missing values in tables based on autogluon-tabular.

Originally, the Imputer contained sophisticated outlier detection. Philipp
stripped that completely and turned the module into a convenience wrapper
around AutoGluon.

"""
import os
import pickle
import inspect
from autogluon.tabular import TabularPredictor

from typing import List, Any

import pandas as pd

class TargetColumnException(Exception):
    """Raised when a target column cannot be used as label for a supervised learning model"""
    pass


class AutoGluonImputer():

    """

    AutoGluonImputer

    :param model_name: name of the AutoGluonImputer (as tring)
    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param verbosity: verbosity level from 0 to 2
    :param output_path: path to which the AutoGluonImputer is saved to
    """

    def __init__(self,
                 columns: List[str],
                 model_name: str = 'AutoGluonImputer',
                 input_columns: List[str] = None,
                 output_column: str = None,
                 verbosity: int = 0,
                 output_path: str = ''
                 ) -> None:

        self.model_name = model_name
        self.columns = columns  # accessed by the imputer feature generator
        self.input_columns = input_columns
        self.output_column = output_column
        self.verbosity = verbosity
        self.predictor = None
        self.output_path = '.' if output_path == '' else output_path

    @property
    def datawig_model_path(self):
        return os.path.join(self.output_path, 'datawigModels', f'{self.model_name}.pickle')

    @property
    def ag_model_path(self):
        return os.path.join(self.output_path, 'agModels', self.model_name)

    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            time_limit: int = 30) -> Any:
        """

        Trains AutoGluonImputer model for a single column

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        :param time_limit: time limit for AutoGluon in seconds
        """

        if self.input_columns is None:
            self.input_columns = [c for c in train_df.columns if c is not self.output_column]

        if train_df[self.output_column].value_counts().max() < 10:
            raise TargetColumnException("Maximum class count below 10, "
                                        "cannot train imputation model")

        self.predictor = TabularPredictor(label=self.output_column,
                                      path=self.ag_model_path,
                                      verbosity=self.verbosity).\
        fit(train_data=train_df,
            tuning_data=test_df,
            time_limit=time_limit,
            verbosity=self.verbosity)

        return self

    def predict_proba(self, data_frame: pd.DataFrame):
        """
        Run the imputer on a data_frame and return class probabilities.
        """
        if not self.predictor:
            raise ValueError("No predictor has been trained. Run .fit() first,"
                             " then continue with .predict().")

        probas = self.predictor.predict_proba(data_frame)
        return probas

    def save(self):
        """

        Saves model to disk. Requires the directory
        `{self.output_path}/datawigModels` to exist.

        """
        params = {k: v for k, v in self.__dict__.items() if k != 'module'}
        pickle.dump(params, open(self.datawig_model_path, "wb"))

    @staticmethod
    def load(output_path: str, model_name: str) -> Any:
        """

        Loads model from output path. Expects
        - a folder `{output_path}/datawigModels` to exist and contain
          `{model_name}.pickle`, itself containing a serialized
          AutoGluonImputer.
        - a folder `{output_path}/agModels` to exist and contain a folder
          named `model_name`, itself containing AutoGluon serialized models.

        :param model_name: string name of the AutoGluonImputer model
        :param output_path: path containing agModels/ and datawigModels/ folders
        :return: AutoGluonImputer model

        """
        params = pickle.load(open(os.path.join(output_path,
                                               "datawigModels",
                                               f"{model_name}.pickle"), "rb"))
        imputer_signature = inspect.getfullargspec(
            AutoGluonImputer.__init__)[0]

        constructor_args = {p: params[p]
                            for p in imputer_signature if p != 'self'}
        non_constructor_args = {p: params[p] for p in params.keys() if
                                p not in ['self'] + list(constructor_args.keys())}

        # use all relevant fields to instantiate AutoGluonImputer
        imputer = AutoGluonImputer(**constructor_args)
        # then set all other args
        for arg, value in non_constructor_args.items():
            setattr(imputer, arg, value)

        # lastly, load AG Model
        imputer.predictor = TabularPredictor.load(os.path.join(output_path,
                                                               "agModels",
                                                               model_name))
        return imputer
