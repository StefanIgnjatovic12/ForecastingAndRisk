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

risk_evaluator = ModelEvaluator()
risk_evaluator.db_connect(host=DB_IP, port=DB_PORT, db_name=DB_NAME, user_name=DB_USERNAME, password=DB_PASSWORD)
# Use date or time (x axis) as first parameter
risk_evaluator.column_format = {'date': 'datetime64', 'id': 'int32', 'price': 'float32', 'item_id': 'int32'}
risk_evaluator.timeframe_column_name = 'date'
# risk_evaluator.target_column_name = 'price'
risk_evaluator.search_query = (f"SELECT DISTINCT {', '.join(risk_evaluator.column_format.keys())} "
                               f"FROM {DB_TABLE_1} "
                               "WHERE marketplace_id = 1 "
                               "AND item_id = %(item_id)i "
                               "AND date >= '%(date_start)s' "
                               "AND date < '%(date_end)s' "
                               "ORDER BY date ASC "
                               "LIMIT 1000")

# risk_evaluator.item_count_query = ("SELECT DISTINCT item_id "
#                                   f"FROM {DB_TABLE_1} "
#                                   "WHERE marketplace_id = 1")

# Column name in table to evaluate risks
column_name = 'price'
# Id's of items to evaluate
item_ids = [i for i in range(1, 5)]

risk_evaluator.get_risk_coefficient(
    item_ids=item_ids,
    column_name=column_name,
    start_year=2022,
    start_month=6,
    months=20
)
