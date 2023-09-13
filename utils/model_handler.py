import os
import sys

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import psycopg2

from .arima_model import get_arima_model
from .corr import get_portfolio
from .indicators import get_indicators
import gc


class ModelEvaluator:
    """
    A class for evaluating machine learning models on financial data.

    Attributes:
        window_size (int): The size of the input window for sequence data.
        window_size_of_indicators (int): The size of the window for calculating indicators.
        batch_size (int): The batch size for training the model.
        epochs (int): The number of training epochs.
        time_estimated (int): Estimated time for model training.
        full_dataframe (pd.DataFrame): The full dataset.
        dataframe (pd.DataFrame): The dataset for training and testing.
        df_columns_to_train (list): Columns used for training the model.
        use_indicators (bool): Whether to use financial indicators in the model.
        training_part (float): The proportion of data used for training.
        validation_part (float): The proportion of data used for validation.
        use_arima (bool): Whether to use ARIMA model.
        item_ids_column_name (str): Name of the column containing item IDs.
        cycle_size (int): Number of different item IDs in the dataset.
        cycle_items_ids (list): List of unique item IDs.
        dropped_short_items_ids (list): List of item IDs that were too short for modeling.
        current_item_num (int): Current item number being processed.cha
        time_of_last_cycle (datetime.datetime): Time of the last cycle.
        time_to_cycle (datetime.datetime): Time until the next cycle.
        data_name (str): Name of the dataset.
        image_output_path (str): Path to save output images.
        model_output_path (str): Path to save trained models.
        model_name (str): Name of the model.
        model_file_name (str): Name of the saved model file.
        scaler (MinMaxScaler): Scaler for normalizing data.
        train_sequences (np.array): Sequences of training data.
        train_targets (np.array): Targets for training data.
        test_sequences (np.array): Sequences of test data.
        test_targets (np.array): Targets for test data.
        test_item_id (int): Item ID for testing.
        test_dataframe (pd.DataFrame): DataFrame for testing.
        column_format (dict): Dictionary of column names and their data types.
        connection: Database connection.
        cursor: Database cursor.
        search_query (str): SQL query to retrieve data.
        item_count_query (str): SQL query to count items in the database.
        target_column_name (str): Name of the target column.
        timeframe_column_name (str): Name of the time column.
        callbacks (list): List of Keras callbacks.
    """

    def __init__(self):
        """
            Initializes a ModelEvaluator object with default parameters and settings.
        """
        self.window_size = 30
        self.window_size_of_indicators = 30
        self.batch_size = 1
        self.epochs = 100
        self.time_estimated = 0

        self.full_dataframe: pd.DataFrame = None
        self.dataframe: pd.DataFrame = None
        self.df_columns_to_train = []
        self.training_part: float = .6
        self.validation_part: float = .2
        self.use_arima: bool = False
        self.use_indicators: bool = False

        self.item_ids_column_name = None
        self.cycle_size: int = 0
        self.cycle_items_ids: list = []
        self.dropped_short_items_ids: list = []
        self.current_item_num: int = 0

        self.time_of_last_cycle = datetime.datetime.now().replace(microsecond=0)
        self.time_to_cycle = self.time_of_last_cycle

        self.data_name = ''
        self.image_output_path = 'output_pics'
        self.model_output_path = 'output_models'
        self.model_name = 'UnknownModel'
        self.model_file_name = 'UnknownModel.h5'
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.train_sequences = None
        self.train_targets = None
        self.test_sequences = None
        self.test_targets = None
        self.test_item_id = 1
        self.test_dataframe: pd.DataFrame = None

        self.column_format: dict = {}
        self.connection = None
        self.cursor = None
        self.search_query = ''
        self.item_count_query = ''

        self.target_column_name = ''
        self.timeframe_column_name = ''

        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_output_path, self.model_file_name),
                save_best_only=True,
                monitor="val_loss",
            ),
            keras.callbacks.ReduceLROnPlateau(
                # monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                monitor="val_loss", factor=0.1, patience=6, min_lr=0.00005
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0),
            keras.callbacks.LambdaCallback(on_epoch_end=self._print_training_log)
        ]

        # Set random seeds for reproducibility
        np.random.seed(0)
        tf.random.set_seed(0)

    def _df_to_sequences_targets(self, dataframe: pd.DataFrame, columns_for_sequences: list, test_only: bool = False):
        """
        Convert a DataFrame into sequences and targets for training and testing.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns_for_sequences (list): List of column names used for creating sequences.
            test_only (bool): If True, only create sequences for testing data.

        Returns:
            tuple: A tuple containing train sequences, train targets, test sequences, and test targets.
                If test_only is True, train sequences and train targets will be None.

        Note:
            This method converts the input DataFrame into sequences of data and their corresponding targets.
            If test_only is False, it splits the data into training and testing sets based on the configured
            training and validation proportions. The sequences are created with a sliding window approach,
            where each sequence contains window_size data points and the target is the value at the next time step.

        """
        dataframe_len = dataframe.shape[0]
        # If we need to cut test zone, otherwise scip calculations
        if test_only is False:
            self.train_size = int(dataframe_len * (self.training_part + self.validation_part))
            self.test_size = int(dataframe_len - self.train_size)
            # If test zone available by settings
            if self.test_size > 0:
                train, test = (dataframe[columns_for_sequences].iloc[:self.train_size].copy(),
                               dataframe[columns_for_sequences].iloc[self.train_size:].copy())
                # reshape into X=t and Y=t+window_size
                train_sequences, train_targets = self._create_sequences(train)
                test_sequences, test_targets = self._create_sequences(test)
                self.test_dataframe = test.reset_index()
                self.test_dataframe.set_index(self.timeframe_column_name, inplace=True)
                return train_sequences, train_targets, test_sequences, test_targets
        train_sequences, train_targets = self._create_sequences(dataframe[columns_for_sequences])
        return train_sequences, train_targets, None, None

    def _df_scale_and_clean(self, dataframe, columns_for_sequences):
        """
        Scale and clean the specified columns of a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to be scaled and cleaned.
            columns_for_sequences (list): List of column names to be scaled and cleaned.

        Returns:
            pd.DataFrame: The DataFrame with specified columns scaled and cleaned.

        Note:
            This method performs two main data preprocessing steps:
            1. It fills missing values (NaNs) in the specified columns using the mean value of the column.
            2. It normalizes the data in the specified columns using the Min-Max scaling technique, which scales
               the data to the range [0, 1].

        """
        # fill the NaNs
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataframe[columns_for_sequences] = imputer.fit_transform(dataframe[columns_for_sequences])
        # normalize the dataset
        dataframe[columns_for_sequences] = self.scaler.fit_transform(dataframe[columns_for_sequences].bfill())
        return dataframe

    def csv_load(
            self,
            file_path: str,
            date_time_column_name: str,
            target_column_name: str,
            features_column_names: list = None,
            item_ids_column_name: int = None,
    ):
        """
        Load data from a CSV file and prepare it for model evaluation.

        Args:
            file_path (str): The path to the CSV file containing the data.
            date_time_column_name (str): The name of the column containing date-time information.
            target_column_name (str): The name of the target column to predict.
            features_column_names (list, optional): A list of column names representing features.
            item_ids_column_name (int, optional): The name or index of the column containing item IDs.

        Returns:
            None

        Note:
            This method loads data from a CSV file, performs preprocessing steps, and prepares the data for model evaluation.
            It handles date-time parsing, duplicate removal, and sorting of item IDs if item_ids_column_name is provided.

        """
        all_column_names_to_work_with = [date_time_column_name, target_column_name]
        if features_column_names:
            all_column_names_to_work_with.extend(features_column_names)
        if item_ids_column_name:
            all_column_names_to_work_with.append(item_ids_column_name)

        csv_dataframe = pd.read_csv(
            file_path,
            parse_dates=True,
            usecols=all_column_names_to_work_with,
            engine='python',
            # nrows=50000
        )
        csv_dataframe = csv_dataframe.drop_duplicates().copy()
        # Is there cycles for different item_id's in the dataframe
        if item_ids_column_name:
            # Get sorted list of id's
            self.cycle_items_ids = sorted(self.dataframe.iloc[:, item_ids_column_name]
                                          .drop_duplicates().values.tolist())
            if not self.cycle_size:
                self.cycle_size = len(self.cycle_items_ids)
            elif len(self.cycle_items_ids) < self.cycle_size:
                print(f'Item limit exceeded')
                return

        csv_dataframe[date_time_column_name] = pd.to_datetime(csv_dataframe[date_time_column_name])
        csv_dataframe = csv_dataframe.set_index(date_time_column_name).copy()

        self.full_dataframe = csv_dataframe.astype('float32').copy()
        del csv_dataframe
        self.item_ids_column_name = item_ids_column_name
        self.target_column_name = target_column_name
        self.df_columns_to_train = [target_column_name]
        self.timeframe_column_name = date_time_column_name
        if features_column_names:
            self.df_columns_to_train.append(features_column_names)

        # for output data naming
        self.data_name = os.path.basename(file_path)

    def db_connect(self,
                   host: str,
                   port: str,
                   db_name: str,
                   user_name: str,
                   password: str,
                   ):
        """
        Establish a connection to a PostgreSQL database.

        Args:
            host (str): The host address of the PostgreSQL server.
            port (str): The port number for the PostgreSQL server.
            db_name (str): The name of the database to connect to.
            user_name (str): The username for authentication.
            password (str): The password for authentication.

        Returns:
            None

        Note:
            This method establishes a connection to a PostgreSQL database using the provided connection details
            such as host, port, database name, username, and password. It also initializes a database cursor for executing
            SQL queries.

        """
        # Connecting
        self.connection = psycopg2.connect(host=host, port=port, dbname=db_name, user=user_name, password=password)
        self.cursor = self.connection.cursor()

    def _db_get_item_count_by_query(self, query):
        """
        Retrieve the count of database items using a SQL query.

        Args:
            query (str): The SQL query to execute for counting items.

        Returns:
            None

        Note:
            This method executes a SQL query to count items in the database and updates the `cycle_items_ids` and `cycle_size`
            attributes based on the query result.

        """
        print(f'Wait a little. Counting database items...')
        self.cursor.execute(query)
        self.cycle_items_ids = np.array(self.cursor.fetchall())[:, 0].tolist()
        self.cycle_size = len(self.cycle_items_ids)

    def _db_get_item_by_month(self, item_id: int, date_start, date_end):
        """
        Retrieve data for a specific item within a date range from the database.

        Args:
            item_id (int): The ID of the item for which to retrieve data.
            date_start: The start date of the date range.
            date_end: The end date of the date range.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved data.

        Note:
            This method executes a SQL query to retrieve data for a specific item within the specified date range
            from the database. The query is constructed using the provided item ID, start date, and end date. The retrieved
            data is then converted into a DataFrame with column names based on `column_format`.

        """
        # Receiving data for a few month
        query = self.search_query % {
            'item_id': item_id,
            'date_start': date_start,
            'date_end': date_end,
        }
        self.cursor.execute(query)
        table_data = self.cursor.fetchall()

        # Create a DataFrame from the data and column names
        dataframe = pd.DataFrame(table_data, columns=list(self.column_format.keys()))
        del table_data
        return dataframe

    def _db_transform_column_data_types(self, dataframe):
        """
        Transform the data types of columns in the DataFrame to match the specified data types.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to be transformed.

        Returns:
            pd.DataFrame: A DataFrame with transformed data types.

        Note:
            This method iterates through the columns of the DataFrame and converts the data types of specific columns
            based on the `column_format` attribute. It ensures that date columns are converted to datetime objects
            and other columns are converted to the specified data types. The resulting DataFrame is set to have
            a datetime index and is cast to 'float32' data type.

        """
        for column_name, column_type in self.column_format.items():
            # if 'date' in column_name or 'time' in column_name:
            if column_name == self.timeframe_column_name:
                dataframe[column_name] = pd.to_datetime(dataframe[column_name])
            else:
                dataframe[column_name].astype(column_type)
        dataframe.set_index(self.timeframe_column_name, inplace=True)
        return dataframe.astype('float32')

    def _add_indicators(self, dataframe):
        """
        Add indicator columns to the DataFrame based on the target column.

        Args:
            dataframe (pd.DataFrame): The DataFrame to which indicators will be added.

        Returns:
            pd.DataFrame: The DataFrame with added indicator columns.
            list: A list of column names, including the target and indicator columns.

        Note:
            This method calculates and adds indicator columns to the DataFrame based on the target column's data.
            It uses the `get_indicators` function to generate the indicators, appends them to the DataFrame, and
            returns both the updated DataFrame and a list of column names, including the target and indicator columns.

        """
        indicators_df = get_indicators(dataframe[self.target_column_name])
        df_columns_to_train = indicators_df.columns.tolist()
        dataframe[df_columns_to_train] = indicators_df[df_columns_to_train].copy()
        del indicators_df
        df_columns_to_train.append(self.target_column_name)
        return dataframe, df_columns_to_train

    def _year_month_day_plus_months_to_date_times(self, year: int, month: int, months: int = 1):
        """
        Calculate the start and end dates based on a given year and month.

        Args:
            year (int): The starting year.
            month (int): The starting month (1-12).
            months (int, optional): The number of months to add to the start date (default is 1).

        Returns:
            tuple: A tuple containing the start date and end date as datetime.date objects.

        Raises:
            ValueError: If the `months` argument is less than 1.

        Note:
            This method takes a starting year and month, and optionally a number of months to add.
            It calculates the start date as the first day of the specified month and the end date by adding the
            specified number of months to the start date. The result is returned as a tuple of datetime.date objects.
            If `months` is less than 1, a ValueError is raised.

        """
        if months < 1:
            self.prepare_and_stop(f'Error: Relative month must be positive: {months}')
        date_start = datetime.date(year, month, 1)
        date_end = date_start + relativedelta(months=months)
        return date_start, date_end

    @staticmethod
    def _date_parser(string):
        """
        Parse a string into a date using Pandas Timestamp.

        Args:
            string (str): The input string representing a date.

        Returns:
            datetime.date: A date object extracted from the input string.

        Note:
            This static method takes an input string representing a date and parses it using Pandas Timestamp.
            It returns the parsed date as a `datetime.date` object.

        """
        return pd.Timestamp(string).date()

    def _create_sequences(self, data):
        """
        Create sequences and corresponding targets from input data.

        Args:
            data (pd.DataFrame): The input data frame.

        Returns:
            tuple: A tuple containing sequences and targets as NumPy arrays.

        Note:
            This method takes input data and creates sequences and corresponding targets based on the window size
            and window size of indicators. It iterates through the data frame and extracts sequences of data
            and their corresponding target values. The sequences and targets are returned as NumPy arrays.

        """
        sequences, targets = [], []
        for i in range(self.window_size_of_indicators, len(data) - self.window_size - 1):
            sequences.append(data[i:i + self.window_size])
            targets.append(data.iloc[i + self.window_size + 1][self.target_column_name])
        return np.array(sequences), np.array(targets).astype('float32')

    def _get_minimum_frametime_length(self):
        """
        Calculate the minimum frame time length for validation.

        Returns:
            int: The minimum frame time length.

        Note:
            This method calculates the minimum frame time length based on the window size, window size of indicators,
            and the validation part. The result is an integer representing the minimum frame time length.

        """
        return int((self.window_size + self.window_size_of_indicators) // self.validation_part)

    def get_risk_coefficient(self,
                             item_ids: list = None,
                             column_name: str = None,
                             start_year: int = None,
                             start_month: int = 1,
                             months: int = 1,
                             start_date: datetime.date = None,
                             end_date: datetime.date = None,
                             ):
        """
        Calculate the risk coefficient based on item IDs and a specified column.

        Args:
            item_ids (list): A list of item IDs for which to calculate the risk coefficient.
            column_name (str): The name of the column used for risk calculation.
            start_year (int): The starting year for data retrieval.
            start_month (int): The starting month for data retrieval.
            months (int): The number of months for data retrieval.
            start_date (datetime.date): The specific starting date for data retrieval.
            end_date (datetime.date): The specific ending date for data retrieval.

        Note:
            This method calculates the risk coefficient for a list of item IDs using a specified column of data.
            It allows for various options for specifying the time period for data retrieval, including start_year and
            months, or start_date and end_date.

        """
        if item_ids is None or len(item_ids) < 2 or column_name is None:
            print('Items IDs and column name is a must to evaluate correlations')
            return

        if start_year:
            _start_date, _end_date = (self._year_month_day_plus_months_to_date_times(start_year, start_month, months))
        else:
            _start_date, _end_date = start_date, end_date
        dataframe = pd.DataFrame()

        # Fetching raw data from tables
        for item_id in item_ids:
            dataframe[str(item_id)] \
                = (self._db_get_item_by_month(item_id, _start_date, _end_date)).copy()[column_name]

        get_portfolio(dataframe, dataframe.columns)

    def start_training(self,
                       model: tf.keras.Model,
                       epochs: int = 10,
                       item_ids: list = None,
                       start_year: int = None,
                       start_month: int = 1,
                       months: int = 1,
                       start_date: datetime.date = None,
                       end_date: datetime.date = None,
                       use_arima: bool = False,
                       use_indicators: bool = False,
                       ):
        """
        Start the training process for a machine learning model.

        Args:
            model (tf.keras.Model): The machine learning model to train.
            epochs (int): The number of training epochs.
            item_ids (list): A list of item IDs to train on. If provided, training is performed on these items only.
            start_year (int): The starting year for data retrieval.
            start_month (int): The starting month for data retrieval.
            months (int): The number of months for data retrieval.
            start_date (datetime.date): The specific starting date for data retrieval.
            end_date (datetime.date): The specific ending date for data retrieval.
            use_arima (bool): If True, use ARIMA modeling instead of machine learning model training.
            use_indicators (bool): If True, use Indicators for machine learning model training. Not work with ARIMA.

        Note:
            This method initiates the training process for a machine learning model, allowing for various options
            such as specifying the time period for data retrieval, using ARIMA modeling, and training on specific item IDs.
            If item_ids are provided, training is performed only on those items. If use_arima is True, ARIMA modeling is
            used instead of machine learning model training.
        """
        if use_indicators:
            self.use_indicators = True

        if use_arima:
            self.use_arima = True
            self.model_name = 'ARIMA'

        if start_year:
            _start_date, _end_date = (self._year_month_day_plus_months_to_date_times(start_year, start_month, months))
        else:
            _start_date, _end_date = start_date, end_date

        self.epochs = epochs

        if item_ids:
            self.cycle_items_ids = item_ids
            self.cycle_size = len(item_ids)
            # Training change 0.6>0.8 will remove test part. Validation default value is 0.2
            self.training_part: float = .8
        else:
            # If we work with database
            if self.connection:
                # for output data (files) naming
                self.data_name = 'db'
                if self.item_count_query:
                    self._db_get_item_count_by_query(self.item_count_query)
                else:
                    self.prepare_and_stop(f"No item id's provided and no count query. Exit")

        print(f'Epochs: {self.epochs} Items: {self.cycle_size}')

        minimum_frametime_length = self._get_minimum_frametime_length()

        # Main cycle
        if self.item_ids_column_name or self.cycle_size > 0:
            for self.current_item_num, current_item_id in enumerate(self.cycle_items_ids):

                if self.full_dataframe is not None:
                    dataframe = self.full_dataframe[
                        (self.full_dataframe.index >= _start_date)
                        & (self.full_dataframe.index <= _end_date)
                        & (self.full_dataframe[self.item_ids_column_name] == current_item_id)
                        ].copy()
                else:
                    # Fetching raw data from tables
                    dataframe = (self._db_get_item_by_month(current_item_id, _start_date, _end_date)).copy()

                if use_arima:
                    self._prepare_and_remember_dataframe_arima(dataframe)
                    model = get_arima_model(self.dataframe, self.target_column_name)
                else:
                    # if dataframe is too short to predict, skip it
                    if dataframe.shape[0] < minimum_frametime_length:
                        self.dropped_short_items_ids.append(current_item_id)
                        continue
                    self._prepare_and_remember_dataframe(dataframe)
                    model = self._start_training(model=model)
                gc.collect()
            self.model_file_name = f'{self.model_name}-{self.cycle_size}' \
                                   f'[{datetime.datetime.now().isoformat(sep="_", timespec="seconds")}].h5'
            self._save_model(model, self.model_file_name)
        else:
            if self.full_dataframe is not None:
                if _start_date and _end_date:
                    dataframe = self.full_dataframe[(self.full_dataframe.index >= _start_date)
                                                    & (self.full_dataframe.index <= _end_date)]
                else:
                    dataframe = self.full_dataframe
                if use_arima:
                    self._prepare_and_remember_dataframe_arima(dataframe)
                    model = get_arima_model(self.dataframe, self.target_column_name)
                else:
                    self._prepare_and_remember_dataframe(dataframe)
                    model = self._start_training(model=model)
                self.model_file_name = f'{self.model_name}-{self.cycle_size}' \
                                       f'[{datetime.datetime.now().isoformat(sep="_", timespec="seconds")}].h5'
                self._save_model(model, self.model_file_name)

        if self.dropped_short_items_ids:
            print(f'Next items are too small and was dropped '
                  f'(expected at least {minimum_frametime_length}):\n'
                  f'{self.dropped_short_items_ids}')

    def _prepare_and_remember_dataframe_arima(self, dataframe):
        """
        Prepare and remember the DataFrame for ARIMA modeling.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the data for modeling.

        This method applies data preprocessing steps such as normalization and handling missing values to prepare
        the DataFrame for use with ARIMA modeling. It focuses on a single target column specified by 'self.target_column_name'.
        After preprocessing, the DataFrame is stored in 'self.dataframe' for use in modeling.

        Note:
            The 'self.target_column_name' attribute should be set to specify the target column for ARIMA modeling.
        """
        # Apply normalizer, antiNaN, clean correlations
        self.dataframe = self._df_scale_and_clean(dataframe, [self.target_column_name]).copy()
        self.df_columns_to_train = [self.target_column_name]
        del dataframe

    def _prepare_and_remember_dataframe(self, dataframe):
        """
        Prepare and remember the DataFrame for machine learning model training.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the data for training.

        This method performs the following steps to prepare the DataFrame for training with a machine learning model:
        1. Builds indicators based on the current dataframe and adds them to the dataframe.
        2. Applies data preprocessing steps such as normalization, handling missing values, and cleaning correlations.
        3. Extracts sequences of features and targets from the dataframe.

        After preprocessing and feature extraction, the resulting sequences and targets are stored in instance variables
        for use in model training. The prepared dataframe is also stored in 'self.dataframe', and the list of columns
        used for training is stored in 'self.df_columns_to_train'.

        Note:
            The target column for training should be specified in the 'self.target_column_name' attribute.
        """
        if self.use_indicators:
            # Building indicators based on current dataframe and add it to df
            dataframe, df_columns_to_train = self._add_indicators(dataframe)
        else:
            df_columns_to_train = [self.target_column_name]

        # Apply normalizer, antiNaN, clean correlations
        dataframe = self._df_scale_and_clean(dataframe, df_columns_to_train)

        # Extract sequences of features and targets from dataframe.
        # If validation (0.2) + training parts (0.8) = 1, then test sequences will be None
        self.train_sequences, self.train_targets, self.test_sequences, self.test_targets = (
            self._df_to_sequences_targets(dataframe, df_columns_to_train))

        self.dataframe = dataframe.copy()
        self.df_columns_to_train = df_columns_to_train
        del dataframe, df_columns_to_train

    def _get_estimate_and_update_timer(self):
        """
        Calculate the estimated time remaining for the current task and update the timer.

        Returns:
            datetime.timedelta: The estimated time remaining for the current task.

        This method calculates the estimated time remaining for processing items in a cycle and updates the timer
        for tracking task progress. It does so by comparing the current time with the time of the last cycle update.
        The estimated time remaining is based on the time taken for processing previous items in the cycle and
        extrapolates it for the remaining items.

        The calculated estimated time remaining is returned as a `datetime.timedelta` object.

        Note:
            This method is typically used to provide progress updates during a long-running task.
        """
        current_time = datetime.datetime.now().replace(microsecond=0)
        self.time_to_cycle = current_time - self.time_of_last_cycle
        self.time_of_last_cycle = current_time
        return self.time_to_cycle * (self.cycle_size - self.current_item_num)

    def db_test_model_on_item(self, item_id, start_year, start_month: int = 1, months: int = 1,
                              start_date: datetime.date = None,
                              end_date: datetime.date = None, ):
        """
        Test a machine learning model on a specific item's data from a database.

        Args:
            item_id (int): The ID of the item to test the model on.
            start_year (int): The start year for fetching data.
            start_month (int): The start month (default: 1).
            months (int): The number of months to fetch data (default: 1).
            start_date (datetime.date): The specific start date for data fetching (optional).
            end_date (datetime.date): The specific end date for data fetching (optional).

        This method retrieves data for a specific item from a database, prepares it for testing, and evaluates the model's accuracy
        on this data. It performs the following steps:

        1. Fetches raw data for the specified item from the database based on the provided date range.
        2. Checks if the data is long enough for testing; if not, it skips testing for this item.
        3. Converts the data types of the dataframe for compatibility with TensorFlow.
        4. Builds indicators based on the current dataframe and adds them to the dataframe.
        5. Applies data preprocessing steps such as normalization, handling missing values, and cleaning correlations.
        6. Creates sequences and targets for testing using the prepared dataframe.
        7. Loads the machine learning model previously trained.
        8. Evaluates the accuracy of the model on the test data.

        Note:
            If the data for the specified item is too short to test, it will be skipped, and a message will be printed.
        """
        print(f'Getting dataset for test..')
        if item_id is not None:
            if start_year:
                _start_date, _end_date = (
                    self._year_month_day_plus_months_to_date_times(start_year, start_month, months))
            else:
                _start_date, _end_date = start_date, end_date
            # Fetching raw data from tables
            test_dataframe = self._db_get_item_by_month(item_id, _start_date, _end_date)
            # if dataframe is too short to predict, skip it
            if test_dataframe.shape[0] < self._get_minimum_frametime_length():
                self.dropped_short_items_ids.append(item_id)
                print('Test item skipped, it is too short')
                return
            # Converting data types for tensorflow
            test_dataframe = self._db_transform_column_data_types(test_dataframe)

            if self.use_indicators:
                # Building indicators based on current dataframe and add it to df
                test_dataframe, df_columns_to_train = self._add_indicators(test_dataframe)
            else:
                df_columns_to_train = [self.target_column_name]
            # test_dataframe, df_columns_to_train = self._add_indicators(test_dataframe)

            # Apply normalizer, antiNaN, clean correlations
            self.test_dataframe = self._df_scale_and_clean(test_dataframe, df_columns_to_train)
            self.df_columns_to_train = df_columns_to_train

            self.test_sequences, self.test_targets, _, _ = (
                self._df_to_sequences_targets(self.test_dataframe, self.df_columns_to_train, test_only=True))
            model = self._load_model(self.model_file_name)

    @staticmethod
    def _text_lvl(lvl_max, lvl_cur):
        """
        Returns text formatted as a process level line.

        Args:
            lvl_max (int): The maximum level of the process.
            lvl_cur (int): The current level of the process.

        Returns:
            str: A text line representing the process level, with '=' characters indicating the current level
                 and '-' characters indicating the remaining level.

        This static method generates a text line to visually represent the progress or level of a process.
        It creates a line of characters where '=' characters represent the current level, and '-' characters represent
        the remaining level up to the maximum level.

        Example:
            If lvl_max=5 and lvl_cur=3, the method will return '===--', indicating that the process is at level 3
            out of a maximum of 5 levels.
        """
        """Returns text formatted as process level line"""
        return f"{'=' * lvl_cur}{'-' * (lvl_max - lvl_cur)}"

    def _start_training(self, model: tf.keras.Model):
        """
        Train a TensorFlow Keras model.

        Args:
            model (tf.keras.Model): The Keras model to be trained.

        Returns:
            tf.keras.Model: The trained Keras model.

        This method compiles and trains the provided Keras model. It compiles the model with a specified loss function
        (Mean Squared Error), optimizer (Adam optimizer with a learning rate of 0.0005), and metrics (Mean Absolute Error
        and Accuracy). The training process includes multiple epochs with specified training parameters, and the training
        history is stored.

        The model file name for saving is determined based on whether there are item IDs or a cycle size. Metrics from
        multiple training runs can be concatenated for analysis. Memory management is performed after each training run.

        Example:
            To train a model, you can call this method with the desired Keras model:
            model = MyKerasModel()
            trained_model = self._start_training(model=model)
        """
        model.compile(
            loss=tf.losses.MeanSquaredError(),
            # optimizer=tf.optimizers.Adam(learning_rate=0.001),
            optimizer=tf.optimizers.Adam(learning_rate=0.0005),
            metrics=[tf.metrics.MeanAbsoluteError()]
        )

        # self.time_estimated = self._get_estimate_and_update_timer()

        combo_mae = combo_val_mae = None
        if self.item_ids_column_name or self.cycle_size:
            self.model_file_name = f'{self.model_name}-{self.cycle_size}[temp].h5'
        else:
            self.model_file_name = (f'{self.model_name}'
                                    f'[{datetime.datetime.now().isoformat(sep="_", timespec="seconds")}].h5')

        history = model.fit(
            self.train_sequences,
            self.train_targets,
            epochs=self.epochs,
            validation_split=1 / ((self.training_part + self.validation_part) / self.validation_part),
            verbose=0,
            callbacks=self.callbacks,
            batch_size=self.batch_size,
        )
        new_mean_absolute_error = history.history['mean_absolute_error']
        new_val_mean_absolute_error = history.history['val_mean_absolute_error']

        if combo_mae is None:
            combo_mae = new_mean_absolute_error
            combo_val_mae = new_val_mean_absolute_error
        else:
            combo_mae = np.concatenate((combo_mae, new_mean_absolute_error), axis=0)
            combo_val_mae = np.concatenate((combo_val_mae, new_val_mean_absolute_error), axis=0)
        self.combo_acc, self.combo_val_acc = combo_mae, combo_val_mae
        del history, combo_mae, combo_val_mae, new_mean_absolute_error, new_val_mean_absolute_error
        gc.collect()
        return model

    def _print_training_log(self, epoch, logs):
        """
        Print training progress log during model training.

        Args:
            epoch (int): The current training epoch.
            logs (dict): A dictionary containing training metrics.

        This method prints a training log to the console, providing information about the current training progress. It
        displays the current item number, the total number of items, the current epoch, the total number of epochs, the
        estimated time remaining for the training, the Mean Absolute Error (MAE), and the loss.

        Example:
            This method is typically called during model training to provide real-time training progress updates.
            model = MyKerasModel()
            history = model.fit(...)
            for epoch in range(epochs):
                self._print_training_log(epoch, history.history)
        """
        self.time_estimated = self._get_estimate_and_update_timer()
        print(f'\rTraining. '
              f'Item: {self.current_item_num + 1}/{self.cycle_size} '
              f'Epoch: {epoch + 1}/{self.epochs} '
              f'Left:[{self.time_estimated}] '
              f'mae: {np.round(logs["mean_absolute_error"], 7)} '
              f'loss: {np.round(logs["loss"], 7)} ',
              # f'acc: {np.round(logs["accuracy"], 7)}',
              end='')

    def plot_accuracy(self, save_file: bool = True):
        """
        Plot the training and validation Mean Absolute Error (MAE) over epochs.

        Args:
            save_file (bool, optional): Whether to save the generated plot as an image file. Default is True.

        This method plots the Mean Absolute Error (MAE) for both the training and validation datasets over epochs during
        model training. It can help visualize the model's training progress and identify potential overfitting or
        underfitting.

        Example:
            After training a model, you can use this method to plot and visualize the training and validation MAE trends.
            trained_model = MyTrainedModel()
            trained_model.load_weights('my_model_weights.h5')
            trained_model.compile(...)
            trained_model.plot_accuracy(save_file=True)
        """
        plt.plot(self.combo_acc, 'orange', label='Training mae')
        plt.plot(self.combo_val_acc, 'blue', label='Validation mae')
        plt.ylabel("sparse_categorical_mae", fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend()
        if save_file:
            # Saving figure by changing parameter values
            self._save_image('acc')
        plt.show()

    def _predict(self, horizon, model: keras.Model):
        """
        Generate predictions for future time steps using the trained model.

        Parameters:
        - horizon (int): The number of future time steps to forecast.
        - model (keras.Model): The trained Keras model for making predictions.

        Returns:
        - prediction_list (np.ndarray): An array containing the forecasted values for the specified horizon.
        """
        seq_list = self.test_sequences.reshape((-1))
        last_proven = seq_list[-1]

        if self.use_indicators:
            last_seq = self.test_sequences[-1, :, :]
            prediction_list = last_seq[-(self.window_size + self.window_size_of_indicators):]
            x = prediction_list[-self.window_size:]
            x = x.reshape((1, self.window_size, len(self.df_columns_to_train)))
            out = model.predict(x, verbose=0)[0][0]
            prediction_list = np.full(horizon + self.window_size + self.window_size_of_indicators, np.nan)
            if hasattr(out, '__iter__'):
                for i, price in enumerate(out, 1):
                    prediction_list[self.window_size + self.window_size_of_indicators + i] = price
            else:
                prediction_list[self.window_size + self.window_size_of_indicators + 1] = out

        else:
            seq_list = self.test_sequences.reshape((-1))
            prediction_list = seq_list[-(self.window_size + self.window_size_of_indicators):]

            for _ in range(horizon):
                x = prediction_list[-self.window_size:]
                x = x.reshape((1, self.window_size, 1))
                out = model.predict(x, verbose=0)[0][0]
                prediction_list = np.append(prediction_list, out)

        initial_diff = last_proven - prediction_list[self.window_size + self.window_size_of_indicators]
        prediction_list = [x + initial_diff for x in prediction_list]

        return prediction_list

    def _predict_dates(self, num_prediction):
        """
        Generate dates for future predictions based on the last date in the test data.

        Parameters:
        - num_prediction (int): The number of future time steps to forecast.

        Returns:
        - prediction_dates (list of datetime.date): A list of datetime dates representing the forecasted dates.
        """
        last_date = self.test_dataframe.index[-1]
        forecasting_dates = pd.date_range(last_date + datetime.timedelta(days=1), periods=num_prediction)
        left_dates = self.test_dataframe.tail(self.window_size + self.window_size_of_indicators).index
        right_dates = forecasting_dates
        prediction_dates = pd.to_datetime(np.concatenate((left_dates, right_dates), axis=None)).tolist()
        return prediction_dates

    def forecast(self, model, horizon: int = 30):
        """
        Generate forecasts for future data points using the trained model.

        Parameters:
        - model: A trained machine learning model.
        - horizon (int, optional): The number of future time steps to forecast. Defaults to 30.

        Returns:
        - forecasted_dataframe (pd.DataFrame): A DataFrame containing the forecasted data with dates as the index.
        """
        forecasted_targets = self._predict(horizon, model)
        forecasted_dates = self._predict_dates(horizon)
        # Make dataframe with some from left and right from the end
        forecasted_dataframe = pd.DataFrame({
            self.timeframe_column_name: forecasted_dates,
            self.target_column_name: forecasted_targets,
        })
        forecasted_dataframe.set_index(self.timeframe_column_name, inplace=True)
        forecasted_dataframe = forecasted_dataframe.astype('float32')
        forecasted_dataframe = pd.concat([self.test_dataframe, forecasted_dataframe[-horizon:]]).ffill()
        forecasted_dataframe[self.df_columns_to_train] = (
            self.scaler.inverse_transform(forecasted_dataframe[self.df_columns_to_train])
        )
        output_dict = {}
        for index, row in forecasted_dataframe[-horizon:].iterrows():
            output_dict[str(index)] = row[self.target_column_name]
        print(f'{output_dict=}')

        return forecasted_dataframe

    def plot_prediction(self, save_file: bool = True):
        """
        Plot the model's predictions for future data points.

        Parameters:
        - save_file (bool, optional): Whether to save the plot as an image file. Defaults to True.

        Returns:
        None
        """
        horizon = 30
        model = self._load_model(self.model_file_name)
        # Generate forecasts for future data points
        forecasted_dataframe = self.forecast(model, horizon)
        # Remove duplicate price from indicators
        self.df_columns_to_train.remove(self.target_column_name)

        x = forecasted_dataframe.index
        if self.test_dataframe is not None:
            plt.plot(x[:-horizon], forecasted_dataframe[:-horizon][self.target_column_name],
                     label=self.target_column_name, color='darkblue')
            plt.plot(x[-horizon:], forecasted_dataframe[-horizon:][self.target_column_name],
                     label='forecasted', color='red', linestyle='dotted')
            for parameter in self.df_columns_to_train:
                plt.plot(x[:-horizon], forecasted_dataframe[:-horizon][parameter], label=parameter, alpha=0.2)

        plt.legend()
        plt.xlabel(self.timeframe_column_name)
        plt.ylabel(self.target_column_name)
        plt.grid(color='lightblue', linestyle='-', linewidth=.5, alpha=0.2)

        if save_file:
            # Saving figure
            self._save_image('prd')
        plt.show()

    def _save_image(self, prefix: str = 'nan'):
        """
        Save the current matplotlib plot as an image file.

        Args:
            prefix (str, optional): A prefix to include in the image file name. Default is 'nan'.

        This method saves the current matplotlib plot as an image file (JPEG format) in the specified
        output directory. It also includes a timestamp in the file name to make it unique.

        Example:
            To save the current plot as an image file:
            my_instance = MyInstance()
            my_instance._save_image(prefix='plot')
        """
        # Check if the directory exists
        if not os.path.exists(self.image_output_path):
            # Create the directory if it does not exist
            os.makedirs(self.image_output_path)

        plt.savefig(os.path.join(self.image_output_path,
                                 f'{self.data_name}[{prefix}][{str(datetime.datetime.now())}].jpg'), dpi=300)

    def _save_model(self, model, file_name):
        """
        Save a TensorFlow/Keras model to a file.

        Args:
            model (tf.keras.Model): The TensorFlow/Keras model to be saved.
            file_name (str): The name of the file to save the model to.

        This method saves a trained TensorFlow/Keras model to the specified file path. The model can be loaded later
        for reuse or further evaluation.

        Example:
            To save a trained model to a file:
            my_instance = MyInstance()
            my_model = tf.keras.Sequential([...])  # Define and train the model
            my_instance._save_model(my_model, file_name='my_model.h5')
        """
        # Check if the directory exists
        if not os.path.exists(self.model_output_path):
            # Create the directory if it does not exist
            os.makedirs(self.model_output_path)
        if model is None:
            print(f'Model was not created. Look for possible errors')
            return
        model.save(os.path.join(self.model_output_path, file_name))
        print(f'Model saved as: {file_name}')

    def _load_model(self, file_name):
        """
        Load a TensorFlow/Keras model from a file.

        Args:
            file_name (str): The name of the file containing the saved model.

        Returns:
            tf.keras.Model or None: The loaded TensorFlow/Keras model if successful, or None if loading failed.

        This method attempts to load a saved TensorFlow/Keras model from the specified file path. If the model is successfully
        loaded, it is returned. If loading fails, None is returned.

        Example:
            To load a previously saved model from a file:
            my_instance = MyInstance()
            loaded_model = my_instance._load_model(file_name='my_model.h5')
            if loaded_model:
                # Use the loaded model for inference or further training
            else:
                print("Model loading failed.")
        """
        # Check if the directory exists
        if not os.path.exists(self.model_output_path):
            print(f'Cannot find path to load Model: {self.model_output_path}')
            return
        model = keras.models.load_model(os.path.join(self.model_output_path, file_name))
        return model

    def prepare_and_stop(self, message):
        """
        Close database connection and exit the program with an error message.

        Args:
            message (str): The error message to display before exiting.

        This method closes the database connection (if open) and then exits the program with an error message. It's typically
        used to gracefully handle errors and terminate the program when necessary.

        Example:
            To close the database connection and exit the program with an error message:
            my_instance = MyInstance()
            my_instance.prepare_and_stop("An error occurred. Exiting.")
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print(message)
        sys.exit(1)
