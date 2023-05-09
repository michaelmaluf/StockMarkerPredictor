import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard
from config import *
from modules.model import create_model
from modules.data_preparation import load_data
import seaborn as sns

# Set stock ticker and data file name
ticker = "GOOGL"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

# Create necessary directories
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

# Load data and save dataframe
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                 shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                 feature_columns=FEATURE_COLUMNS)
data["df"].to_csv(ticker_data_filename)

# Create and train model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True,
                               verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
def plot_graph(test_df):
    plt.figure(figsize=(12, 6))
    sns.set_style('whitegrid')
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.title(f"${ticker} Actual vs Predicted Price", fontsize=14, fontweight='bold')
    plt.xlabel("Days", fontsize=12, fontweight='bold')
    plt.ylabel("Price", fontsize=12, fontweight='bold')
    plt.legend(["Actual Price", "Predicted Price"], loc='upper left', fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.show()


# Function to create final dataframe
def get_final_df(model, data):
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    test_df.sort_index(inplace=True)
    final_df = test_df
    final_df["buy_profit"] = list(map(buy_profit,
                                      final_df["adjclose"],
                                      final_df[f"adjclose_{LOOKUP_STEP}"],
                                      final_df[f"true_adjclose_{LOOKUP_STEP}"]))
    final_df["sell_profit"] = list(map(sell_profit,
                                       final_df["adjclose"],
                                       final_df[f"adjclose_{LOOKUP_STEP}"],
                                       final_df[f"true_adjclose_{LOOKUP_STEP}"]))

    return final_df


# Function to predict future price
def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# Load optimal model weights
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# Evaluate model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# Get final dataframe and predict future price
final_df = get_final_df(model, data)
future_price = predict(model, data)

# Calculate metrics:
# Number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(
    final_df)
# Total buy and sell profits
total_buy_profit = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# Total profit by adding buy and sell profit together
total_profit = total_buy_profit + total_sell_profit
# Profit per trade
profit_per_trade = total_profit / len(final_df)

# Print metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)

plot_graph(final_df)
