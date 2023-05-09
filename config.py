import time
from keras.layers import LSTM

# Window size/Sequence length
N_STEPS = 50
# Number of days we want to predict after
LOOKUP_STEP = 15
# Scale is set to True for our application
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# Shuffle is also set to True for our application
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# Split by date is set to False for our application
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# Test size: 20%
TEST_SIZE = 0.2
# Columns to use from api data
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# Date now
date_now = time.strftime("%Y-%m-%d")
# Model parameters
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# Bidirectional will be set to False
BIDIRECTIONAL = False
# Training parameters
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 5

