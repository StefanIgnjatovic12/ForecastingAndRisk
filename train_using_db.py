import os
from utils.model_handler import ModelEvaluator
from utils.models import *
from dotenv import load_dotenv

load_dotenv()

DB_IP = os.getenv('DB_IP')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_TABLE_1 = os.getenv('DB_TABLE_1')

model_trainer = ModelEvaluator()
model_trainer.db_connect(host=DB_IP, port=DB_PORT, db_name=DB_NAME, user_name=DB_USERNAME, password=DB_PASSWORD)
# Use date or time (x axis) as first parameter
model_trainer.column_format = {'date': 'datetime64', 'id': 'int32', 'price': 'float32', 'item_id': 'int32'}
model_trainer.timeframe_column_name = 'date'
model_trainer.target_column_name = 'price'
model_trainer.search_query = (f"SELECT DISTINCT {', '.join(model_trainer.column_format.keys())} "
                              f"FROM {DB_TABLE_1} "
                              "WHERE marketplace_id = 1 "
                              "AND item_id = %(item_id)i "
                              "AND date >= '%(date_start)s' "
                              "AND date < '%(date_end)s' "
                              "ORDER BY date ASC "
                              "LIMIT 1000")

model_trainer.item_count_query = ("SELECT DISTINCT item_id "
                                  f"FROM {DB_TABLE_1} "
                                  "WHERE marketplace_id = 1")

# Set number of values to predict next one
# Must be at least like biggest "timescale" in utils.indicators (30 by default)
window_size = 36
model_trainer.window_size = window_size

# Choose your model here
model = ModelVad1
# Create model and get class name to name the model file
# For some models number of neurons are correlate with window size.
model_trainer.model_name = model.__name__
model = model(window_size)

# model_trainer._db_get_item_by_month(1, 2022, 9, 3)
epochs = 10
# Test on that item after training a list or an all items in db
model_trainer.test_item_id = 126
model_trainer.batch_size = 1
# You can appoint exact id's or leave the list empty to use all items in table
# item_ids = [i for i in range(1, 3)]
item_ids = [126]
start_year = 2022
start_month = 9
months = 13
use_arima = False
use_indicators = True

model_trainer.start_training(
    model=model,
    epochs=epochs,
    item_ids=item_ids,
    start_year=start_year,
    start_month=start_month,
    months=months,
    use_arima=use_arima,
    use_indicators=use_indicators,
)
if not use_arima:
    model_trainer.db_test_model_on_item(
        item_id=10,
        start_year=start_year,
        start_month=start_month,
        months=months,
    )

    # Plot and save accuracy image
    model_trainer.plot_accuracy()
    # Plot and save prediction image
    model_trainer.plot_prediction()
