# previsao_demanda
## Previsão de receita para seis meses futuros
Este projeto foi elaborado no Python. A lista de bibliotecas utilizadas foi:

1. import pandas as pd
2. import scipy.stats as stats
3. from scipy.stats import shapiro
4. from plotnine import *
5. import seaborn as sns
6. import numpy as np
7. import matplotlib.pyplot as plt
8. import warnings
9. from sklearn.linear_model import LinearRegression
10. from sklearn.metrics import mean_absolute_error
11. from sklearn.model_selection import cross_validate
12. from sklearn.model_selection import cross_val_score
13. from datetime import datetime
14. from tensorflow.keras.layers import LSTM
15. from tensorflow.keras.models import Sequential
16. from tensorflow.keras import layers
17. from tensorflow.keras import callbacks
18. from sklearn.ensemble import RandomForestRegressor
19. from xgboost import XGBRegressor
20. from prophet import Prophet
21. from prophet.diagnostics import cross_validation
22. from prophet.diagnostics import performance_metrics
23. from prophet.plot import plot_cross_validation_metric
24. from prophet.plot import plot_plotly, plot_components_plotly
25. from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, mean_absolute_percentage_error
26. from statsmodels.graphics.tsaplots import plot_acf
27. from statsmodels.graphics.tsaplots import plot_predict
28. import plotnine
29. from pmdarima.arima.utils import ndiffs
30. from sklearn.preprocessing import StandardScaler
31. import pmdarima as pm
32. import tensorflow as tf
33. import math
34. from matplotlib import rcParams
                                                                        
[Página pessoal no Github](https://robertomoniz.github.io/)
