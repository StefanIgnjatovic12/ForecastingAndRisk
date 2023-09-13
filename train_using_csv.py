import os
from utils.model_handler import ModelEvaluator
from utils.models import *

file_1 = {'name': 'EURUSD_Candlestick_1_D_ASK_05.05.2003-30.06.2021.csv', 'dtime': 'Local time', 'target': 'close'}
file_2 = {'name': 'aapl.us.txt', 'dtime': 'Date', 'target': 'Close'}
file_3 = {'name': 'amd.us.txt', 'dtime': 'Date', 'target': 'Close'}
file_4 = {'name': 'nvda.us.txt', 'dtime': 'Date', 'target': 'Close'}
file_5 = {'name': 'oxfd.us.txt', 'dtime': 'Date', 'target': 'Close'}
# file_csgodb = {'name': 'csgodb.csv', 'dtime': 'date', 'target': 'price', 'item_ids_col': 2, 'item_ids': [1, 2]}

# Choose your file here
file = file_5

model_trainer = ModelEvaluator()
model_trainer.csv_load(
    file_path=os.path.join('inputs', file["name"]),
    date_time_column_name=file['dtime'],
    target_column_name=file['target'],
    features_column_names=file.get('features', None),
    item_ids_column_name=file.get('item_ids_col', None),  # Column number, with item ids
)

# Set number of values to predict next one
window_size = 36
model_trainer.window_size = window_size

# Choose your model here
model = ModelVad1
# Create model and get class name to name the model file
# For some models number of neurons are correlate with window size.
model_trainer.model_name = model.__name__
model = model(window_size)
model_trainer.batch_size = 1
# Training start. There are no difference 5x10 or 10x5 epochs x aeons,
# but aeons visualised on progress bar.
epochs = 10
use_arima = False
use_indicators = True

model_trainer.start_training(
    model=model,
    epochs=epochs,
    item_ids=file.get('item_ids', None),
    use_arima=use_arima,
    use_indicators=use_indicators,
)
if not use_arima:
    # Plot and save accuracy image
    model_trainer.plot_accuracy()
    # Plot and save prediction image
    model_trainer.plot_prediction()
