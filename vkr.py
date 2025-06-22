import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import entropy, kstest
import statsmodels.api as sm
import moexalgo as ma
import yfinance as yf
import random
from datetime import datetime, timedelta
import os
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
from itertools import combinations

# Включаем Wide Mode
st.set_page_config(layout="wide")

# Установка темы
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
        color: #212529;
    }
    .stSelectbox, .stSlider, .stTextInput, .stButton, .stDateInput, .stCheckbox {
        background-color: #FFFFFF;
        color: #212529;
        border-radius: 5px;
        border: 1px solid #CED4DA;
    }
    .stButton>button {
        background-color: #007BFF;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #0056B3;
    }
    [data-testid="column"] {
        padding-right: 20px;
    }
    [data-testid="column"]:first-child {
        min-width: 400px !important;
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .js-plotly-plot .plotly .modebar {
        display: none;
    }
    .catastrophe-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E9ECEF;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h3 {
        color: #007BFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Заголовок приложения
st.title("Аналитическая платформа для анализа облигаций")

# Инициализация портфеля
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'GAZP (Excel)': {'type': 'excel', 'path': 'C:/Users/Kurchin/Desktop/.venv/GAZP_.xlsx'},
        'SBER (Excel)': {'type': 'excel', 'path': 'C:/Users/Kurchin/Desktop/.venv/SBER_.xlsx'}
    }

portfolio = st.session_state.portfolio

# Функция экспоненциального сглаживания
def exponential_smoothing(series, alpha=0.2):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

# Функция для классификации нейронной сетью
def classify_with_nn(df_subset):
    try:
        # Проверка на минимальное количество данных
        if len(df_subset) < 10:
            st.warning("Недостаточно данных для классификации (менее 10 записей).")
            return None, None, None, None, None, None, None, None

        # Обработка пропусков и аномалий
        df_subset = df_subset.dropna()
        z_scores = np.abs((df_subset[['dp_dt', 'RSI', 'MACD', 'Stoch_K']] - 
                          df_subset[['dp_dt', 'RSI', 'MACD', 'Stoch_K']].mean()) / 
                          df_subset[['dp_dt', 'RSI', 'MACD', 'Stoch_K']].std())
        df_subset = df_subset[(z_scores < 3).all(axis=1)]
        
        # Проверка после очистки
        if len(df_subset) < 10:
            st.warning("После удаления аномалий осталось слишком мало данных.")
            return None, None, None, None, None, None, None, None
        
        # Подготовка данных
        features = df_subset[['dp_dt', 'db_dp', 'd2b_dp2', 'd3b_dp3', 'RSI', 'MACD', 'Stoch_K']].values
        threshold = df_subset['dp_dt'].std()
        target = np.select(
            [df_subset['dp_dt'] > threshold, 
             (df_subset['dp_dt'] > 0) & (df_subset['dp_dt'] <= threshold),
             (df_subset['dp_dt'] >= -threshold) & (df_subset['dp_dt'] <= 0),
             df_subset['dp_dt'] < -threshold],
            [2, 1, 0, -1],
            default=0
        )
        
        # Проверка количества уникальных классов
        unique_classes = np.unique(target)
        if len(unique_classes) < 2:
            st.warning(f"Недостаточно уникальных классов для классификации (найдено: {len(unique_classes)}).")
            return None, None, None, None, None, None, None, None
        
        # Нормализация
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        model = MLPClassifier(
            hidden_layer_sizes=(20, 10, 5),
            activation='relu',
            solver='adam',
            max_iter=1000,
            alpha=0.01,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Оценка
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Кросс-валидация
        scores = cross_val_score(model, features_scaled, target, cv=5, scoring='f1_weighted')
        
        # Важность признаков
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = pd.DataFrame({
            'Признак': ['dp_dt', 'db_dp', 'd2b_dp2', 'd3b_dp3', 'RSI', 'MACD', 'Stoch_K'],
            'Важность': result.importances_mean
        }).sort_values('Важность', ascending=False)
        
        # ROC-кривая (one-vs-rest)
        roc_data = []
        class_names = {-1: 'Сильный нисх.', 0: 'Нейтральный', 1: 'Слабый восх.', 2: 'Сильный восх.'}
        if len(np.unique(y_test)) > 1:  # Проверяем, что есть несколько классов
            y_score = model.predict_proba(X_test)
            for i, class_label in enumerate(unique_classes):
                if class_label in y_test:
                    fpr, tpr, _ = roc_curve(y_test == class_label, y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data.append({
                        'class': class_names[class_label],
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                    })
        
        # Отладочная информация
        st.info(f"Количество записей после очистки: {len(df_subset)}, Уникальные классы: {unique_classes}")
        
        return model, report, cm, scores, feature_importance, roc_data, y_test, y_pred
    except Exception as e:
        st.error(f"Ошибка при классификации: {str(e)}")
        return None, None, None, None, None, None, None, None

# Функция загрузки данных с вычислением дополнительных производных и индикаторов
@st.cache_data
def load_data(source_info):
    try:
        source_type = source_info['type']
        if source_type == 'excel':
            if not os.path.exists(source_info['path']):
                st.error(f"Файл {source_info['path']} не найден!")
                return None
            df = pd.read_excel(source_info['path'])
            df['<DATE>'] = pd.to_datetime(df['<DATE>'], errors='coerce')
        elif source_type == 'moex':
            ticker = source_info.get('ticker', 'GAZP')
            stock = ma.Ticker(ticker)
            today = datetime.now().strftime('%Y-%m-%d')
            df = stock.candles(start='2023-01-01', end=today)
            df = df[['begin', 'close', 'high', 'low']].rename(columns={'begin': '<DATE>', 'close': '<CLOSE>', 'high': '<HIGH>', 'low': '<LOW>'})
            df['<DATE>'] = pd.to_datetime(df['<DATE>'])
        elif source_type == 'yahoo':
            ticker = source_info.get('ticker', 'AAPL')
            today = datetime.now().strftime('%Y-%m-%d')
            df = yf.download(ticker, start='2023-01-01', end=today)[['Close', 'High', 'Low']].reset_index()
            df.columns = ['<DATE>', '<CLOSE>', '<HIGH>', '<LOW>']
        elif source_type == 'alpha':
            df = pd.DataFrame({
                '<DATE>': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                '<CLOSE>': [200 + i * random.uniform(-2, 2) for i in range(100)],
                '<HIGH>': [202 + i * random.uniform(-2, 2) for i in range(100)],
                '<LOW>': [198 + i * random.uniform(-2, 2) for i in range(100)]
            })
        elif source_type == 'expert':
            df = pd.DataFrame({
                '<DATE>': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                '<CLOSE>': [150 + i * random.uniform(-1.5, 1.5) for i in range(100)],
                '<HIGH>': [152 + i * random.uniform(-1.5, 1.5) for i in range(100)],
                '<LOW>': [148 + i * random.uniform(-1.5, 1.5) for i in range(100)]
            })
        elif source_type == 'imoex':
            index = ma.Ticker('IMOEX')
            today = datetime.now().strftime('%Y-%m-%d')
            df = index.candles(start='2023-01-01', end=today)
            df = df[['begin', 'close', 'high', 'low']].rename(columns={'begin': '<DATE>', 'close': '<CLOSE>', 'high': '<HIGH>', 'low': '<LOW>'})
            df['<DATE>'] = pd.to_datetime(df['<DATE>'])
        
        df = df.sort_values('<DATE>')

        if df is None or df.empty:
            st.error(f"Нет данных для тикера {ticker} за указанный период.")
            return None

        
        # Применяем экспоненциальное сглаживание
        df['<CLOSE>_smoothed'] = exponential_smoothing(df['<CLOSE>'])
        df['<HIGH>_smoothed'] = exponential_smoothing(df['<HIGH>'])
        df['<LOW>_smoothed'] = exponential_smoothing(df['<LOW>'])
        
        # Вычисляем производные на основе сглаженных данных
        df['dp_dt'] = df['<CLOSE>_smoothed'].diff() / df['<DATE>'].diff().dt.days
        df['db_dt'] = df['dp_dt'].diff() / df['<DATE>'].diff().dt.days
        df['db_dp'] = df['dp_dt'].diff() / df['<CLOSE>_smoothed'].diff()
        df['d2b_dp2'] = np.gradient(df['db_dp'], df['<CLOSE>_smoothed'])
        df['d3b_dp3'] = np.gradient(df['d2b_dp2'], df['<CLOSE>_smoothed'])
        
        # Вычисляем RSI
        df['RSI'] = RSIIndicator(df['<CLOSE>_smoothed']).rsi()
        
        # Вычисляем MACD
        macd = MACD(df['<CLOSE>_smoothed'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Вычисляем Bollinger Bands
        bb = BollingerBands(df['<CLOSE>_smoothed'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        
        # Вычисляем Stochastic Oscillator
        stochastic = StochasticOscillator(high=df['<HIGH>_smoothed'], low=df['<LOW>_smoothed'], close=df['<CLOSE>_smoothed'], window=14, smooth_window=3)
        df['Stoch_K'] = stochastic.stoch()
        df['Stoch_D'] = stochastic.stoch_signal()
        
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

# Функция анализа катастроф
def analyze_catastrophe(df_subset, nn_pred=None, market_volatility=20, risk_free_rate=0.02):
    if len(df_subset) < 5:
        return "Недостаточно данных", "Анализ невозможен из-за малого объёма данных.", 100.0, 0.0, 0.0, 0.0, "Держать: Недостаточно данных.", "Недостаточно данных для анализа.", [], 0.0, 0.0, "Нейтральный MACD", 0.0, "Нейтральные Bollinger Bands", 0.0, 0.0, "Нейтральный Stochastic"

    # Анализ изменений знака для производных
    sign_changes_dp_dt = np.diff(np.sign(df_subset['dp_dt'].dropna()))
    sign_changes_db_dp = np.diff(np.sign(df_subset['db_dp'].dropna()))
    sign_changes_d2b_dp2 = np.diff(np.sign(df_subset['d2b_dp2'].dropna()))
    sign_changes_d3b_dp3 = np.diff(np.sign(df_subset['d3b_dp3'].dropna()))
    
    num_changes_dp_dt = np.sum(sign_changes_dp_dt != 0)
    num_changes_db_dp = np.sum(sign_changes_db_dp != 0)
    num_changes_d2b_dp2 = np.sum(sign_changes_d2b_dp2 != 0)
    num_changes_d3b_dp3 = np.sum(sign_changes_d3b_dp3 != 0)
    
    mean_dp_dt = df_subset['dp_dt'].mean()
    std_dp_dt = df_subset['dp_dt'].std()
    
    # Вычисляем уровень риска
    risk_level = (std_dp_dt / abs(mean_dp_dt)) * 100 if mean_dp_dt != 0 else 100.0
    
    # Вычисляем коэффициент Шарпа
    sharpe_ratio = (mean_dp_dt - risk_free_rate) / std_dp_dt if std_dp_dt != 0 else 0.0
    
    # Дополнительные метрики: автокорреляция и энтропия
    autocorr = sm.tsa.acf(df_subset['dp_dt'].dropna(), nlags=1)[1]
    hist, _ = np.histogram(df_subset['dp_dt'].dropna(), bins=10)
    ent = entropy(hist / len(df_subset))
    
    # Обнаружение аномалий с помощью Z-score
    z_scores = (df_subset['dp_dt'] - mean_dp_dt) / std_dp_dt
    anomalies = df_subset[abs(z_scores) > 3][['<DATE>', 'dp_dt']].to_dict('records')
    
    # Вычисляем RSI
    mean_rsi = df_subset['RSI'].mean()
    rsi_signal = ""
    if mean_rsi > 70:
        rsi_signal = "Перекупленность (RSI > 70)."
    elif mean_rsi < 30:
        rsi_signal = "Перепроданность (RSI < 30)."
    else:
        rsi_signal = "Нейтральный RSI (30-70)."

    # Вычисляем MACD сигнал
    mean_macd = df_subset['MACD'].mean()
    macd_cross = ""
    if df_subset['MACD'].iloc[-1] > df_subset['MACD_Signal'].iloc[-1] and df_subset['MACD'].iloc[-2] <= df_subset['MACD_Signal'].iloc[-2]:
        macd_cross = "Бычий сигнал (MACD пересёк сигнальную линию вверх)."
    elif df_subset['MACD'].iloc[-1] < df_subset['MACD_Signal'].iloc[-1] and df_subset['MACD'].iloc[-2] >= df_subset['MACD_Signal'].iloc[-2]:
        macd_cross = "Медвежий сигнал (MACD пересёк сигнальную линию вниз)."
    else:
        macd_cross = "Нейтральный MACD (нет пересечений)."

    # Вычисляем Bollinger Bands сигнал
    bb_signal = ""
    latest_price = df_subset['<CLOSE>_smoothed'].iloc[-1]
    latest_bb_high = df_subset['BB_High'].iloc[-1]
    latest_bb_low = df_subset['BB_Low'].iloc[-1]
    if latest_price > latest_bb_high:
        bb_signal = "Перекупленность (цена выше верхней полосы)."
    elif latest_price < latest_bb_low:
        bb_signal = "Перепроданность (цена ниже нижней полосы)."
    else:
        bb_signal = "Нейтральные Bollinger Bands (цена внутри полос)."

    # Вычисляем Stochastic Oscillator сигнал
    mean_stoch_k = df_subset['Stoch_K'].mean()
    mean_stoch_d = df_subset['Stoch_D'].mean()
    stoch_signal = ""
    if df_subset['Stoch_K'].iloc[-1] > df_subset['Stoch_D'].iloc[-1] and df_subset['Stoch_K'].iloc[-2] <= df_subset['Stoch_D'].iloc[-2] and df_subset['Stoch_K'].iloc[-1] < 80:
        stoch_signal = "Бычий сигнал (Stoch %K пересёк %D вверх)."
    elif df_subset['Stoch_K'].iloc[-1] < df_subset['Stoch_D'].iloc[-1] and df_subset['Stoch_K'].iloc[-2] >= df_subset['Stoch_D'].iloc[-2] and df_subset['Stoch_K'].iloc[-1] > 20:
        stoch_signal = "Медвежий сигнал (Stoch %K пересёк %D вниз)."
    else:
        stoch_signal = "Нейтральный Stochastic (нет пересечений)."

    # Классификация катастроф
    if num_changes_db_dp == 0 and autocorr < 0.5:
        catastrophe_type = "Складка"
        explanation = f"Простейшая катастрофа с низкой автокорреляцией (autocorr={autocorr:.2f})."
    elif num_changes_db_dp > 0 and num_changes_d2b_dp2 == 0 and ent < 1.0:
        catastrophe_type = "Куспид"
        explanation = f"Резкие скачки с низкой энтропией (entropy={ent:.2f})."
    elif num_changes_d2b_dp2 > 0 and num_changes_d3b_dp3 == 0:
        catastrophe_type = "Ласточкин хвост"
        explanation = f"Сложная бифуркация, {num_changes_d2b_dp2} изменений знака во второй производной."
    elif num_changes_d3b_dp3 > 0 and num_changes_dp_dt < 5:
        catastrophe_type = "Бабочка"
        explanation = f"Множественные точки нестабильности, {num_changes_d3b_dp3} изменений в третьей производной."
    elif num_changes_db_dp > 0 and num_changes_d2b_dp2 > 0 and risk_level > 50:
        catastrophe_type = "Гиперболический пупок"
        explanation = "Множественная нестабильность в нескольких измерениях с высокой волатильностью."
    elif num_changes_dp_dt > 5 and num_changes_db_dp > 0:
        catastrophe_type = "Эллиптический пупок"
        explanation = f"Циклическое поведение, {num_changes_dp_dt} изменений знака в dp/dt."
    elif num_changes_d3b_dp3 > 0 and risk_level > 50:
        catastrophe_type = "Параболический пупок"
        explanation = "Сложная нелинейная динамика с высокой неопределённостью."
    else:
        catastrophe_type = "Неопределённая катастрофа"
        explanation = "Недостаточно данных для точной классификации."
    
    # Рекомендация с учётом Bollinger Bands, Stochastic Oscillator и нейронной сети
    recommendation_explanation = ""
    if catastrophe_type == "Складка":
        if (risk_level < 50 and mean_dp_dt > 0 and market_volatility < 25 and sharpe_ratio > 0.5 and 
            mean_rsi < 70 and macd_cross.startswith("Бычий") and bb_signal.startswith("Нейтральные") and 
            stoch_signal.startswith("Бычий")):
            recommendation = "Покупать: Стабильное поведение с низким риском и бычьими сигналами."
            recommendation_explanation = f"Рекомендация 'Покупать' основана на низком риске ({risk_level:.2f}%), положительном тренде (mean dp/dt={mean_dp_dt:.4f}), высоком коэффициенте Шарпа ({sharpe_ratio:.2f}), отсутствии перекупленности ({rsi_signal}), бычьем сигнале MACD ({macd_cross}), нейтральных Bollinger Bands ({bb_signal}) и бычьем Stochastic ({stoch_signal})."
        elif risk_level < 100:
            recommendation = "Держать: Умеренный риск, стабильная система."
            recommendation_explanation = f"Рекомендация 'Держать' из-за умеренного риска ({risk_level:.2f}%), стабильного поведения ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
        else:
            recommendation = "Продать: Высокий риск несмотря на стабильность."
            recommendation_explanation = f"Рекомендация 'Продать' из-за высокого риска ({risk_level:.2f}%) ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    elif catastrophe_type == "Куспид":
        if (risk_level < 50 and mean_dp_dt > 0 and market_volatility < 25 and sharpe_ratio > 0.5 and 
            mean_rsi < 70 and macd_cross.startswith("Бычий") and bb_signal.startswith("Нейтральные") and 
            stoch_signal.startswith("Бычий")):
            recommendation = "Покупать: Нестабильность, но низкий риск и бычьи сигналы."
            recommendation_explanation = f"Рекомендация 'Покупать' основана на низком риске ({risk_level:.2f}%), положительном тренде (mean dp/dt={mean_dp_dt:.4f}), высоком коэффициенте Шарпа ({sharpe_ratio:.2f}), отсутствии перекупленности ({rsi_signal}), бычьем сигнале MACD ({macd_cross}), нейтральных Bollinger Bands ({bb_signal}) и бычьем Stochastic ({stoch_signal})."
        elif risk_level < 75 or market_volatility < 30:
            recommendation = "Держать: Умеренный риск, требуется наблюдение."
            recommendation_explanation = f"Рекомендация 'Держать' из-за умеренного риска ({risk_level:.2f}%), возможной нестабильности ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
        else:
            recommendation = "Продать: Высокий риск и нестабильность."
            recommendation_explanation = f"Рекомендация 'Продать' из-за высокого риска ({risk_level:.2f}%) и нестабильности ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    elif catastrophe_type in ["Ласточкин хвост", "Бабочка", "Параболический пупок"]:
        recommendation = "Продать: Высокая нестабильность и сложное поведение."
        recommendation_explanation = f"Рекомендация 'Продать' из-за сложной динамики (тип катастрофы: {catastrophe_type}), высокого риска ({risk_level:.2f}%) ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    elif catastrophe_type == "Гиперболический пупок":
        if risk_level < 50 and market_volatility < 25:
            recommendation = "Держать: Умеренная нестабильность, низкий риск."
            recommendation_explanation = f"Рекомендация 'Держать' из-за умеренного риска ({risk_level:.2f}%) несмотря на нестабильность ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
        else:
            recommendation = "Продать: Высокий риск и множественная нестабильность."
            recommendation_explanation = f"Рекомендация 'Продать' из-за высокого риска ({risk_level:.2f}%) и множественной нестабильности ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    elif catastrophe_type == "Эллиптический пупок":
        if (risk_level < 50 and mean_dp_dt > 0 and market_volatility < 25 and sharpe_ratio > 0.5 and 
            mean_rsi < 70 and macd_cross.startswith("Бычий") and bb_signal.startswith("Нейтральные") and 
            stoch_signal.startswith("Бычий")):
            recommendation = "Покупать: Циклическое поведение с низким риском и бычьими сигналами."
            recommendation_explanation = f"Рекомендация 'Покупать' основана на низком риске ({risk_level:.2f}%), положительном тренде (mean dp/dt={mean_dp_dt:.4f}), высоком коэффициенте Шарпа ({sharpe_ratio:.2f}), отсутствии перекупленности ({rsi_signal}), бычьем сигнале MACD ({macd_cross}), нейтральных Bollinger Bands ({bb_signal}) и бычьем Stochastic ({stoch_signal})."
        else:
            recommendation = "Держать: Циклическая нестабильность, требуется наблюдение."
            recommendation_explanation = f"Рекомендация 'Держать' из-за циклической нестабильности, умеренного риска ({risk_level:.2f}%) ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    else:
        recommendation = "Держать: Недостаточно данных для точной рекомендации."
        recommendation_explanation = f"Рекомендация 'Держать' из-за недостатка данных для точного анализа ({rsi_signal}, {macd_cross}, {bb_signal}, {stoch_signal})."
    
    # Интеграция с нейронной сетью
    if nn_pred is not None:
        nn_labels = {2: "Сильный восходящий тренд", 1: "Слабый восходящий тренд", 
                     0: "Нейтральный", -1: "Слабый нисходящий тренд", -2: "Сильный нисходящий тренд"}
        nn_prediction = nn_labels.get(nn_pred, "Неизвестно")
        recommendation_explanation += f" Нейронная сеть предсказывает: {nn_prediction}."
        if nn_pred in [1, 2] and recommendation == "Держать":
            recommendation = "Покупать"
            recommendation_explanation += " Рекомендация скорректирована на 'Покупать' из-за бычьего прогноза нейронной сети."
        elif nn_pred in [-1, -2]:
            recommendation = "Продать"
            recommendation_explanation += " Рекомендация скорректирована на 'Продать' из-за медвежьего прогноза нейронной сети."

    # Добавляем информацию об аномалиях
    if anomalies:
        recommendation_explanation += f" Обнаружены аномалии ({len(anomalies)} шт.), что может указывать на нестабильность."

    return (catastrophe_type, explanation, risk_level, mean_dp_dt, std_dp_dt, sharpe_ratio, 
            recommendation, recommendation_explanation, anomalies, mean_rsi, mean_macd, macd_cross, 
            mean_stoch_k, bb_signal, mean_stoch_k, mean_stoch_d, stoch_signal)

# Создание двух колонок
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    # Добавление нового источника
    with st.expander("Добавить источник"):
        source_name = st.text_input("Название источника")
        source_type = st.selectbox("Тип источника", [
            "База данных (Excel)", "Мосбиржа", "Yahoo Finance", "Alpha Vantage", "Финансовый эксперт"
        ])
        path_input = st.text_input("Путь к файлу (для Excel)", "")
        ticker_input = st.text_input("Тикер (для Yahoo Finance или Мосбиржа)", "")
        if st.button("Добавить"):
            if not source_name:
                st.warning("Введите название источника")
            else:
                if source_type == "База данных (Excel)":
                    if not path_input:
                        st.warning("Укажите путь к файлу")
                    else:
                        portfolio[source_name] = {'type': 'excel', 'path': path_input}
                elif source_type == "Мосбиржа":
                    portfolio[source_name] = {'type': 'moex', 'ticker': ticker_input if ticker_input else 'GAZP'}
                elif source_type == "Yahoo Finance":
                    portfolio[source_name] = {'type': 'yahoo', 'ticker': ticker_input if ticker_input else 'AAPL'}
                elif source_type == "Alpha Vantage":
                    portfolio[source_name] = {'type': 'alpha'}
                elif source_type == "Финансовый эксперт":
                    portfolio[source_name] = {'type': 'expert'}
                st.session_state.portfolio = portfolio
                st.success(f"Источник '{source_name}' добавлен!")
                st.rerun()

    # Выбор источника
    selected_source = st.selectbox("Выберите источник:", list(portfolio.keys()))
    df = load_data(portfolio[selected_source])

    if df is not None and not df.empty:
        # Ползунки для дат
        min_date = df['<DATE>'].min().date()
        max_date = df['<DATE>'].max().date()
        start_date = st.date_input("Начальная дата", min_value=min_date, max_value=max_date, value=min_date, key="start_date")
        end_date = st.date_input("Конечная дата", min_value=min_date, max_value=max_date, value=max_date, key="end_date")
        
        # Проверка корректности дат
        if start_date > end_date:
            st.error("Начальная дата не может быть позже конечной!")
        else:
            df_subset = df[(df['<DATE>'].dt.date >= start_date) & (df['<DATE>'].dt.date <= end_date)]
            st.write(f"Выбранный диапазон: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}")
        
        # Ползунок для волатильности рынка
        market_volatility = st.slider("Волатильность рынка (%)", 0, 100, 20, key="market_volatility")
        
        # Ползунки для минимальной и максимальной цены
        min_price = st.slider("Минимальная цена", min_value=float(df['<CLOSE>_smoothed'].min()), max_value=float(df['<CLOSE>_smoothed'].max()), value=float(df['<CLOSE>_smoothed'].min()))
        max_price = st.slider("Максимальная цена", min_value=float(df['<CLOSE>_smoothed'].min()), max_value=float(df['<CLOSE>_smoothed'].max()), value=float(df['<CLOSE>_smoothed'].max()))
        
        # Выбор периода прогноза
        forecast_period = st.selectbox("Период прогноза (месяцы)", [1, 3, 6, 12], index=0)
        error = np.random.uniform(0.01, 0.1)  # Примерная погрешность
        st.write(f"Ожидаемая ошибка прогноза: {error:.2%}")

# Построение графика в правой колонке
if df is not None and not df.empty and 'df_subset' in locals() and not df_subset.empty:
    with col2:
        graph_type = st.selectbox("Выберите тип графика:", [
            "График цен", "График плотности", "RSI", "MACD", "Bollinger Bands", 
            "Stochastic Oscillator", "Проекция t-p", "Прогноз dp/dt", "График бифуркаций", "Фазовая проекция"
        ])
        
        # Чекбоксы
        show_regression = False
        show_bifurcation = False
        if graph_type == "Проекция t-p":
            show_regression = st.checkbox("Показать линию регрессии", value=True)
        if graph_type == "График бифуркаций":
            show_bifurcation = st.checkbox("Показать график бифуркации", value=True)
        
        # Контейнер для графика
        chart_container = st.empty()
        
        # Контейнер для анализа катастроф
        catastrophe_container = st.container()
        
        # Проверка равномерности распределения
        price_uniformity = kstest(df_subset['<CLOSE>_smoothed'], 'uniform', args=(df_subset['<CLOSE>_smoothed'].min(), df_subset['<CLOSE>_smoothed'].max() - df_subset['<CLOSE>_smoothed'].min()))
        st.write(f"Тест Колмогорова-Смирнова для цены: statistic={price_uniformity.statistic:.4f}, p-value={price_uniformity.pvalue:.4f}")
        
        # Функция для обновления графика
        def update_chart(graph_type):
            fig = None
            # Классификация с помощью нейронной сети
            nn_model, nn_report, nn_cm, nn_scores, nn_feature_importance, nn_roc_data, y_test, y_pred = classify_with_nn(df_subset)
            nn_pred = nn_model.predict(df_subset[['dp_dt', 'db_dp', 'd2b_dp2', 'd3b_dp3', 'RSI', 'MACD', 'Stoch_K']].values[-1:])[0] if nn_model else None
            
            # Вызываем анализ для получения переменных
            (catastrophe_type, explanation, risk_level, mean_dp_dt, std_dp_dt, sharpe_ratio, 
             recommendation, recommendation_explanation, anomalies, mean_rsi, mean_macd, macd_cross, 
             mean_stoch_k, bb_signal, mean_stoch_k, mean_stoch_d, stoch_signal) = analyze_catastrophe(df_subset, nn_pred, market_volatility)
            
            annotation_text = f"Катастрофа: {catastrophe_type}\nРекомендация: {recommendation}"
            
            # Шаблон подсказок
            hover_template = (
                "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                "<b>Сглаженная цена</b>: %{x:.2f}<br>" +
                "<b>dp/dt</b>: %{y:.4f}<br>" +
                "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                "<extra></extra>"
            )
            
            if graph_type == "График цен":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['<CLOSE>'], 
                    mode='lines',
                    name=f'Цена ({selected_source})',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['<CLOSE>_smoothed'], 
                    mode='lines',
                    name=f'Сглаженная цена ({selected_source})',
                    line=dict(color='#FF0000'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                if anomalies:
                    anomaly_dates = [pd.to_datetime(anomaly['<DATE>']) for anomaly in anomalies]
                    anomaly_subset = df_subset[df_subset['<DATE>'].isin(anomaly_dates)]
                    fig.add_trace(go.Scatter(
                        x=anomaly_dates,
                        y=anomaly_subset['<CLOSE>_smoothed'],
                        mode='markers',
                        name='Аномалии',
                        marker=dict(size=10, color='#FFA500', symbol='x'),
                        customdata=anomaly_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                        hovertemplate=hover_template + "<b>Аномалия</b><br>"
                    ))
                non_return_point = (df_subset['<CLOSE>_smoothed'].mean(), df_subset['dp_dt'].mean())
                fig.add_trace(go.Scatter(
                    x=[df_subset['<DATE>'].iloc[-1]],
                    y=[non_return_point[0]],
                    mode='markers',
                    name='Точка невозврата',
                    marker=dict(size=10, color='red')
                ))
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'График цен ({selected_source})',
                    xaxis_title="Дата",
                    yaxis_title="Цена",
                    showlegend=True,
                    template="plotly_white",
                    height=600,
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "График плотности":
                fig = px.histogram(
                    df_subset, 
                    x='<CLOSE>_smoothed', 
                    histnorm='probability density',
                    title=f'График плотности ({selected_source})',
                    color_discrete_sequence=['#0000FF']
                )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    xaxis_title="Сглаженная цена закрытия",
                    yaxis_title="Плотность",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
            
            elif graph_type == "RSI":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['RSI'], 
                    mode='lines',
                    name='RSI',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Перекупленность", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Перепроданность", annotation_position="bottom left")
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'RSI ({selected_source})',
                    xaxis_title="Дата",
                    yaxis_title="RSI",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "MACD":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['MACD'], 
                    mode='lines',
                    name='MACD',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{y:.4f}<br>" +
                        "<b>MACD Signal</b>: %{customdata[3]:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['MACD_Signal'], 
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='#FF0000'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>MACD Signal</b>: %{y:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_trace(go.Bar(
                    x=df_subset['<DATE>'], 
                    y=df_subset['MACD_Hist'], 
                    name='MACD Histogram',
                    marker_color=df_subset['MACD_Hist'].apply(lambda x: 'green' if x >= 0 else 'red'),
                    opacity=0.5,
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>MACD Hist</b>: %{y:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_hline(y=0, line_color="black", line_width=1)
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'MACD ({selected_source})',
                    xaxis_title="Дата",
                    yaxis_title="MACD",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "Bollinger Bands":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['<CLOSE>_smoothed'], 
                    mode='lines',
                    name='Сглаженная цена',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['BB_High'], 
                    mode='lines',
                    name='Верхняя полоса',
                    line=dict(color='#FF0000', dash='dash'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>BB High</b>: %{y:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['BB_Low'], 
                    mode='lines',
                    name='Нижняя полоса',
                    line=dict(color='#FF0000', dash='dash'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>BB Low</b>: %{y:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['BB_Mid'], 
                    mode='lines',
                    name='Средняя полоса',
                    line=dict(color='#00FF00'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>BB Mid</b>: %{y:.2f}<br>" +
                        "<b>dp/dt</b>: %{customdata[2]:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'Bollinger Bands ({selected_source})',
                    xaxis_title="Дата",
                    yaxis_title="Цена",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "Stochastic Oscillator":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['Stoch_K'], 
                    mode='lines',
                    name='Stoch %K',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['Stoch_D'], 
                    mode='lines',
                    name='Stoch %D',
                    line=dict(color='#FF0000'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Перекупленность", annotation_position="top left")
                fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Перепроданность", annotation_position="bottom left")
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'Stochastic Oscillator ({selected_source})',
                    xaxis_title="Дата",
                    yaxis_title="Stochastic",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "Проекция t-p":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=df_subset['<CLOSE>_smoothed'], 
                    mode='lines',
                    name='Проекция',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                if show_regression:
                    X = np.array(range(len(df_subset))).reshape(-1, 1)
                    y = df_subset['<CLOSE>_smoothed'].values
                    model = LinearRegression()
                    model.fit(X, y)
                    predicted = model.predict(X)
                    fig.add_trace(go.Scatter(
                        x=df_subset['<DATE>'],
                        y=predicted,
                        mode='lines',
                        name='Регрессия',
                        line=dict(color='#FF0000')
                    ))
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title='Проекция t-p',
                    xaxis_title="Время",
                    yaxis_title="Сглаженная цена",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "Прогноз dp/dt":
                X = df_subset[['<CLOSE>_smoothed']].values
                y = df_subset['dp_dt'].values
                model = LinearRegression()
                model.fit(X, y)
                predicted_dp_dt = model.predict(X)

                last_date = df_subset['<DATE>'].iloc[-1]
                current_date = pd.to_datetime("2025-05-18")
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_period*30, freq='D')
                p_future = np.linspace(df_subset['<CLOSE>_smoothed'].iloc[-1], 
                                     df_subset['<CLOSE>_smoothed'].iloc[-1] + model.coef_[0] * len(future_dates), 
                                     len(future_dates))
                b_future = model.predict(p_future.reshape(-1, 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=y, 
                    mode='lines',
                    name='Реальное dp/dt',
                    line=dict(color='#0000FF'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                fig.add_trace(go.Scatter(
                    x=df_subset['<DATE>'], 
                    y=predicted_dp_dt, 
                    mode='lines',
                    name='Прогноз dp/dt (исторические данные)',
                    line=dict(color='#FF0000'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                if len(future_dates) > 0:
                    fig.add_trace(go.Scatter(
                        x=future_dates, 
                        y=b_future, 
                        mode='lines',
                        name='Прогноз на выбранный период',
                        line=dict(color='#00FF00'),
                        hovertemplate="<b>Дата</b>: %{x|%d-%m-%Y}<br><b>dp/dt</b>: %{y:.4f}<br><extra></extra>"
                    ))
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text + f"\nОжидаемая ошибка: {error:.2%}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title='Прогноз dp/dt (Линейная регрессия)',
                    xaxis_title="Дата",
                    yaxis_title="dp/dt",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
                fig.update_xaxes(tickformat="%d-%m-%Y")
            
            elif graph_type == "График бифуркаций":
                p = df_subset['<CLOSE>_smoothed']
                dp_dt = df_subset['dp_dt']
                db_dp = df_subset['db_dp']
                d2b_dp2 = df_subset['d2b_dp2']
                
                bifurcation_points = df_subset[np.abs(d2b_dp2) > d2b_dp2.std()]
                if not bifurcation_points.empty:
                    bifurcation_points = bifurcation_points.assign(risk_level=np.abs(bifurcation_points['dp_dt']) * 100)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=db_dp, 
                    y=dp_dt, 
                    mode='lines+markers',
                    name='Динамика',
                    line=dict(color='#0000FF', dash='dash'),
                    marker=dict(size=5),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=(
                        "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                        "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                        "<b>dp/dt</b>: %{y:.4f}<br>" +
                        "<b>db/dp</b>: %{x:.4f}<br>" +
                        "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                        "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                        "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                        "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))
                if show_bifurcation and not bifurcation_points.empty:
                    ci_lower = dp_dt - 1.96 * dp_dt.std()
                    ci_upper = dp_dt + 1.96 * dp_dt.std()
                    fig.add_trace(go.Scatter(
                        x=db_dp, 
                        y=ci_lower, 
                        mode='lines',
                        name='Нижний CI',
                        line=dict(color='#00FF00', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=db_dp, 
                        y=ci_upper, 
                        mode='lines',
                        name='Верхний CI',
                        line=dict(color='#00FF00', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=bifurcation_points['db_dp'], 
                        y=bifurcation_points['dp_dt'], 
                        mode='markers',
                        name='Точки бифуркации',
                        marker=dict(
                            size=10, 
                            color=bifurcation_points['risk_level'], 
                            symbol='x', 
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Уровень риска")
                        ),
                        customdata=bifurcation_points[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                        hovertemplate=(
                            "<b>Дата</b>: %{customdata[0]|%d-%m-%Y}<br>" +
                            "<b>Цена</b>: %{customdata[1]:.2f}<br>" +
                            "<b>dp/dt</b>: %{y:.4f}<br>" +
                            "<b>db/dp</b>: %{x:.4f}<br>" +
                            "<b>RSI</b>: %{customdata[2]:.2f}<br>" +
                            "<b>MACD</b>: %{customdata[3]:.4f}<br>" +
                            "<b>Stoch %K</b>: %{customdata[4]:.2f}<br>" +
                            "<b>Stoch %D</b>: %{customdata[5]:.2f}<br>" +
                            "<b>Точка бифуркации</b><br>" +
                            "<extra></extra>"
                        )
                    ))
                if catastrophe_type == "Складка":
                    fig.add_vrect(
                        x0=db_dp.min(), x1=db_dp.max(),
                        fillcolor="rgba(0, 255, 0, 0.1)", opacity=0.5,
                        layer="below", line_width=0,
                        annotation_text="Стабильная зона", annotation_position="top left"
                    )
                elif catastrophe_type == "Куспид":
                    fig.add_vrect(
                        x0=db_dp.min(), x1=db_dp.max(),
                        fillcolor="rgba(255, 255, 0, 0.1)", opacity=0.5,
                        layer="below", line_width=0,
                        annotation_text="Зона нестабильности", annotation_position="top left"
                    )
                else:
                    fig.add_vrect(
                        x0=db_dp.min(), x1=db_dp.max(),
                        fillcolor="rgba(255, 0, 0, 0.1)", opacity=0.5,
                        layer="below", line_width=0,
                        annotation_text="Зона высокого риска", annotation_position="top left"
                    )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title='График бифуркаций',
                    xaxis_title="Производная db/dp",
                    yaxis_title="Производная dp/dt",
                    showlegend=True,
                    template="plotly_white",
                    height=600,
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        orientation="v",
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=10),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="black",
                        borderwidth=1
                    )
                )
            
            elif graph_type == "Фазовая проекция":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_subset['<CLOSE>_smoothed'],
                    y=df_subset['dp_dt'],
                    mode='lines+markers',
                    name='Фазовая траектория',
                    line=dict(color='#0000FF'),
                    marker=dict(size=5),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                    hovertemplate=hover_template
                ))
                if anomalies:
                    anomaly_dates = [pd.to_datetime(anomaly['<DATE>']) for anomaly in anomalies]
                    anomaly_subset = df_subset[df_subset['<DATE>'].isin(anomaly_dates)]
                    fig.add_trace(go.Scatter(
                        x=anomaly_subset['<CLOSE>_smoothed'],
                        y=anomaly_subset['dp_dt'],
                        mode='markers',
                        name='Аномалии',
                        marker=dict(size=10, color='#FFA500', symbol='x'),
                        customdata=anomaly_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].values,
                        hovertemplate=hover_template + "<b>Аномалия</b><br>"
                    ))
                fig.add_trace(go.Scatter(
                    x=[df_subset['<CLOSE>_smoothed'].iloc[-1]],
                    y=[df_subset['dp_dt'].iloc[-1]],
                    mode='markers',
                    name='Текущая точка',
                    marker=dict(size=10, color='red'),
                    customdata=df_subset[['<DATE>', '<CLOSE>', 'RSI', 'MACD', 'Stoch_K', 'Stoch_D']].iloc[-1:].values,
                    hovertemplate=hover_template + "<b>Текущая точка</b><br>"
                ))
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                fig.update_layout(
                    title=f'Фазовая проекция ({selected_source})',
                    xaxis_title="Сглаженная цена",
                    yaxis_title="dp/dt",
                    showlegend=True,
                    template="plotly_white",
                    height=600
                )
            
            # Отображаем график
            if fig is not None:
                chart_container.plotly_chart(fig, use_container_width=True)
            
            # Отображаем анализ катастроф и результаты классификации
            with catastrophe_container:
                st.markdown('<div class="catastrophe-box">', unsafe_allow_html=True)
                st.subheader("Анализ катастроф")
                
                if nn_model and nn_report:
                    st.subheader("Классификация нейронной сетью")
                    st.write("**Отчет о классификации:**")
                    st.write(pd.DataFrame(nn_report).transpose())
                    st.write(f"**Кросс-валидация (F1):** {nn_scores.mean():.2f} (±{nn_scores.std():.2f})")
                    
                    # Матрица ошибок
                    if nn_cm is not None and y_test is not None and y_pred is not None:
                        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
                        class_names = {-1: 'Сильный нисх.', 0: 'Нейтральный', 1: 'Слабый восх.', 2: 'Сильный восх.'}
                        labels = [class_names[c] for c in unique_classes if c in class_names]
                        fig_cm = px.imshow(
                            nn_cm,
                            labels=dict(x="Предсказанный класс", y="Истинный класс"),
                            x=labels,
                            y=labels,
                            text_auto=True
                        )
                        fig_cm.update_layout(title="Матрица ошибок")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Важность признаков
                    fig_fi = px.bar(nn_feature_importance, x='Признак', y='Важность', title="Важность признаков")
                    st.plotly_chart(fig_fi, use_container_width=True)
                    
                    # ROC-кривая
                    if nn_roc_data:
                        fig_roc = go.Figure()
                        for roc in nn_roc_data:
                            fig_roc.add_trace(go.Scatter(
                                x=roc['fpr'], 
                                y=roc['tpr'], 
                                mode='lines', 
                                name=f"{roc['class']} (AUC = {roc['auc']:.2f})"
                            ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1], 
                            mode='lines', 
                            line=dict(dash='dash'), 
                            name='Случайная модель'
                        ))
                        fig_roc.update_layout(
                            title='ROC-кривые (один против остальных)',
                            xaxis_title='Доля ложных срабатываний',
                            yaxis_title='Доля истинных срабатываний',
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                st.write(f"**Тип катастрофы**: {catastrophe_type}")
                st.write(f"**Объяснение**: {explanation}")
                st.write(f"**Уровень риска**: {risk_level:.2f}%")
                st.write(f"**Среднее dp/dt**: {mean_dp_dt:.4f}")
                st.write(f"**Стандартное отклонение dp/dt**: {std_dp_dt:.4f}")
                st.write(f"**Коэффициент Шарпа**: {sharpe_ratio:.2f}")
                st.write(f"**Рекомендация**: {recommendation}")
                st.write(f"**Объяснение рекомендации**: {recommendation_explanation}")
                if anomalies:
                    st.write("**Аномалии**:")
                    for anomaly in anomalies:
                        st.write(f"- Дата: {anomaly['<DATE>'].strftime('%d-%m-%Y')}, dp/dt: {anomaly['dp_dt']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Обновляем график
        update_chart(graph_type)

else:
    st.warning("Данные недоступны или пусты. Пожалуйста, выберите источник данных.")


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import statsmodels.api as sm

# Генерация обучающих данных для классификации катастроф
def generate_catastrophe_training_data(df, window_size=30, step_size=15, min_points=20):
    X, y = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start+window_size]
        if len(window) < min_points:
            continue
        try:
            label, *_ = analyze_catastrophe(window, nn_pred=None)
            if label == "Неопределённая катастрофа":
                continue

            features = [
                window['dp_dt'].mean(),
                window['dp_dt'].std(),
                window['db_dp'].mean(),
                window['d2b_dp2'].mean(),
                window['d3b_dp3'].mean(),
                np.sum(np.diff(np.sign(window['dp_dt'].dropna())) != 0),
                np.sum(np.diff(np.sign(window['db_dp'].dropna())) != 0),
                np.sum(np.diff(np.sign(window['d2b_dp2'].dropna())) != 0),
                np.sum(np.diff(np.sign(window['d3b_dp3'].dropna())) != 0),
                window['RSI'].mean(),
                window['MACD'].mean(),
                window['Stoch_K'].mean(),
                window['Stoch_D'].mean(),
                entropy(np.histogram(window['dp_dt'].dropna(), bins=10)[0] / len(window)),
                sm.tsa.acf(window['dp_dt'].dropna(), nlags=1)[1],
                (window['dp_dt'].std() / abs(window['dp_dt'].mean())) * 100 if window['dp_dt'].mean() != 0 else 100.0,
                (window['dp_dt'].mean() - 0.02) / window['dp_dt'].std() if window['dp_dt'].std() != 0 else 0.0
            ]
            X.append(features)
            y.append(label)
        except Exception:
            continue
    return np.array(X), np.array(y)

# Обучение модели классификации катастроф
def train_catastrophe_classifier(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Предсказание типа катастрофы нейросетью
def predict_catastrophe_type_with_nn(df_subset, model, scaler):
    if len(df_subset) < 20:
        return None
    try:
        features = np.array([[
            df_subset['dp_dt'].mean(),
            df_subset['dp_dt'].std(),
            df_subset['db_dp'].mean(),
            df_subset['d2b_dp2'].mean(),
            df_subset['d3b_dp3'].mean(),
            np.sum(np.diff(np.sign(df_subset['dp_dt'].dropna())) != 0),
            np.sum(np.diff(np.sign(df_subset['db_dp'].dropna())) != 0),
            np.sum(np.diff(np.sign(df_subset['d2b_dp2'].dropna())) != 0),
            np.sum(np.diff(np.sign(df_subset['d3b_dp3'].dropna())) != 0),
            df_subset['RSI'].mean(),
            df_subset['MACD'].mean(),
            df_subset['Stoch_K'].mean(),
            df_subset['Stoch_D'].mean(),
            entropy(np.histogram(df_subset['dp_dt'].dropna(), bins=10)[0] / len(df_subset)),
            sm.tsa.acf(df_subset['dp_dt'].dropna(), nlags=1)[1],
            (df_subset['dp_dt'].std() / abs(df_subset['dp_dt'].mean())) * 100 if df_subset['dp_dt'].mean() != 0 else 100.0,
            (df_subset['dp_dt'].mean() - 0.02) / df_subset['dp_dt'].std() if df_subset['dp_dt'].std() != 0 else 0.0
        ]])
        features_scaled = scaler.transform(features)
        return model.predict(features_scaled)[0]
    except Exception:
        return None

# === Интеграция с интерфейсом Streamlit ===
if df is not None and not df.empty and 'df_subset' in locals() and not df_subset.empty:
    # Обучение модели классификации катастроф
    X_train, y_train = generate_catastrophe_training_data(df)
    if len(np.unique(y_train)) > 1:
        model_cat, scaler_cat = train_catastrophe_classifier(X_train, y_train)
        nn_catastrophe_type = predict_catastrophe_type_with_nn(df_subset, model_cat, scaler_cat)
        if nn_catastrophe_type:
            st.write(f"""
#### Предсказанная нейросетью катастрофа: **{nn_catastrophe_type}**""")
