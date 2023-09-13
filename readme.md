#### Prerequisites
- Python 3.10 or higher
- PIP

#### Installation:
`git clone "repo_link" or unpack the archive`  
+ Create and activate virtual environment:
  + Linux / macOS  
  `python3 -m venv /path_to_virtual_environment/venv`  
  `source /path_to_virtual_environment/venv/bin/activate`  
  + Windows:  
  `python -m venv \path_to_virtual_environment\venv`  
  `\path_to_virtual_environment\venv\bin\activate.bat`
+ Install requirements:  
  `python -m pip install -r requirements.txt`
+ Install precompiled ta-lib (indicators):
  + Linux - Download cp310 x64 version lib from:  
  https://www.wheelodex.org/projects/ta-lib-precompiled/  :  
  `python -m pip install TA_Lib_Precompiled-0.4.25-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`  
  + Windows - Download cp310 x64 version lib from:  
  https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib  :  
  `python -m pip install TA_Lib‑0.4.24‑cp310‑cp310‑win_amd64.whl`

#### Configuration:

+ Make a copy of .env_example and name it as .env
+ Fill all the variables in .env to connect to your PostgreSQL database
+ Make configurations in files you want to use:
  + correlation_using_db.py
  + train_using_csv.py
  + train_using_db.py
+ Keep window_size at least as 36-40 and month not less than 12 to get result
+ Keep batch_size=1 for more detailed prediction, higher values will do training faster

#### Project Structure

- `main.py`: The main entry point of the application.
- `config/`: Directory containing configuration files.
- `modules/`: Directory containing reusable modules.
- `logs/`: Directory for storing log files.
- `plots/`: Directory for storing data files.
+ Start any of the file:
+ `correlation_using_db.py`: File to configurate and start risks evaluation (work only with database)
+ `train_using_csv.py`: File to configurate and start model training using csv file
+ `train_using_db.py`: File to configurate and start model training using database
+ `utils/`: Directory containing reusable modules
+ `inputs/`: Directory containing csv file for testing and training
+ `output_models/`: Directory to save trained model file
+ `output_pics/`: Directory to save rendered pictures

#### Training:

+ `python train_using_csv.py` - To use database tables start this script
+ `train_using_csv.py` - To use csv files start this script

#### Calculate risks:
+ `correlation_using_db.py` - To evaluate item risks using database tables start this script

