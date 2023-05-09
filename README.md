# StockMarkerPredictor
Application that downloads real time stock prices from Yahoo Finance and builds a model via TensorFlow. A Recurrent Neural Network model is built utilizing LSTM cells, dropout for regularization, and a dense output layer which produces the final output.

The data from the API is standardized and training and test sets are built. The model is then trained with its respective sets. The model predicts a multitude of metrics such as future price, accuracy score, total profit, and potential profit per trade. 

Concepts excelled/learned: 
1) ML with tensor flow
2) More API practice
3) More pandas, scikit-learn, and matplot practice. 
4) The ability to take a real life problem and code a solution (ex. if you were a day trader, this is a model that you can use to confirm your bias).
