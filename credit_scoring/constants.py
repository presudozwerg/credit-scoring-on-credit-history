import torch
from pathlib import Path

# Data describe
ID_COLUMN_NAME = 'id'
TECH_COLS = ['rn']

# Data paths for prod
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = 'data_balanced'
DATA_ROOT = ROOT / DATA_FOLDER
# DATA_ROOT = Path('/home/przwerg99/Документы/MIPT/sem2/MLOps/credit-scoring-on-credit-history/data_raw')

TRAIN_FILES_FOLDER = 'train_data'
TRAIN_TARGET_FILE = 'train_target.csv'

TEST_FILES_FOLDER = 'test_data'

# Preprocessing
ROWS_THRESHOLD = 15
VAL_SIZE = 0.2
BATCH_SIZE = 128

# Training process
N_EPOCHS = 10
CHECKPOINTS_PATH = ROOT / 'model_checkpoints'

# Model params
EMBED_DIM = 64
HIDDEN_DIM = 32
N_LAYERS = 1
RNN_TYPE = 'rnn'
AGG_TYPE = 'last'
IS_BIDIR = True
DROPOUT_P = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
