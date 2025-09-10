# ==============================================================================
# SCRIPT DE OPTIMIZACIÓN Y COMPARACIÓN DE MODELOS (GRID SEARCH + HORSE RACE)
# ==============================================================================
# 1. Optimiza individualmente cada modelo (LGBM, XGB, RF, NN) con Grid Search.
# 2. Guarda el progreso de cada optimización para ser robusto a interrupciones.
# 3. Compara la mejor versión de cada modelo y guarda al ganador en un archivo JSON.
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from itertools import product
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import json
import ast

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PREPARACIÓN DE DATOS ---
def actualizar_datos():
    """Carga los datos desde los archivos CSV locales."""
    logging.info("Paso 1: Cargando datos...")
    try:
        spy_df = pd.read_csv('spy_15y_daily.csv', index_col='date', parse_dates=True)
        vix_df = pd.read_csv('vix_15y_daily.csv', index_col='date', parse_dates=True)
        merged_df = pd.merge(spy_df[['open', 'high', 'low', 'close']], vix_df[['close']], left_index=True, right_index=True, suffixes=('_SPY', '_VIX'))
        merged_df.rename(columns={'close_SPY': 'close', 'close_VIX': 'vix_level'}, inplace=True)
        merged_df['returns_SPY'] = np.log(merged_df['close']).diff()
        return merged_df.dropna()
    except FileNotFoundError as e:
        logging.error(f"Error crítico: No se encontraron los archivos de datos. {e}"); return None

def preparar_datos_y_features(df):
    """Aplica toda la ingeniería de características y objetivos."""
    logging.info("Paso 2: Creando características y objetivos...")
    features_df = df.copy()
    features_df['volatility_21d'] = features_df['returns_SPY'].rolling(window=21).std()
    high_low = features_df['high'] - features_df['low']
    high_close = np.abs(features_df['high'] - features_df['close'].shift())
    low_close = np.abs(features_df['low'] - features_df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features_df['atr_14d'] = tr.rolling(window=14).mean()
    features_df['momentum_21d'] = features_df['close'].pct_change(21)
    features_df['momentum_63d'] = features_df['close'].pct_change(63)
    features_df['momentum_126d'] = features_df['close'].pct_change(126)
    horizon = 21
    future_min = features_df['close'].shift(-horizon).rolling(window=horizon).min()
    future_max = features_df['close'].shift(-horizon).rolling(window=horizon).max()
    features_df['target_lower_pct'] = (future_min - features_df['close']) / features_df['close']
    features_df['target_upper_pct'] = (future_max - features_df['close']) / features_df['close']
    features_df.dropna(inplace=True)
    return features_df

# --- FUNCIONES PARA RED NEURONAL ---
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

def crear_modelo_nn(quantile, num_features, params):
    model = Sequential([
        Dense(params.get('hidden_layer_1', 32), activation='relu', input_shape=(num_features,)),
        Dense(params.get('hidden_layer_2', 16), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(quantile, y_true, y_pred))
    return model

# --- FUNCIÓN DE EVALUACIÓN REUTILIZABLE ---
def evaluar_combinacion(features_df, nombre_modelo, params, quantiles):
    """Ejecuta un backtest completo para una combinación específica de modelo y parámetros."""
    breaches = 0
    lower_q, upper_q = quantiles
    
    backtest_start_date = features_df.index[-1] - pd.DateOffset(months=24)
    test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index

    for trade_date in test_dates:
        history_df = features_df[features_df.index < trade_date].tail(504)
        evaluation_end = trade_date + pd.DateOffset(days=21)
        test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
        
        if history_df.empty or test_df.empty: continue
        
        features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
        X_train, y_train_lower, y_train_upper = history_df[features_to_use], history_df['target_lower_pct'], history_df['target_upper_pct']
        X_test = test_df.head(1)[features_to_use]
        current_price = test_df.head(1)['close'].values[0]

        train_params = params.copy()
        
        if nombre_modelo == 'LightGBM':
            model_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **train_params).fit(X_train, y_train_lower)
            model_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **train_params).fit(X_train, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test)[0], model_upper.predict(X_test)[0]
        
        elif nombre_modelo == 'XGBoost':
            model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=lower_q, **train_params).fit(X_train, y_train_lower)
            model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=upper_q, **train_params).fit(X_train, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test)[0], model_upper.predict(X_test)[0]

        elif nombre_modelo == 'RandomForest':
            model_lower = RandomForestRegressor(n_jobs=-1, **train_params).fit(X_train, y_train_lower)
            model_upper = RandomForestRegressor(n_jobs=-1, **train_params).fit(X_train, y_train_upper)
            preds_lower_trees = np.array([tree.predict(X_test) for tree in model_lower.estimators_])
            preds_upper_trees = np.array([tree.predict(X_test) for tree in model_upper.estimators_])
            lower_pct_pred, upper_pct_pred = np.quantile(preds_lower_trees, lower_q), np.quantile(preds_upper_trees, upper_q)

        elif nombre_modelo == 'NeuralNetwork':
            scaler = StandardScaler()
            X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
            epochs = train_params.pop('epochs', 20)
            batch_size = train_params.pop('batch_size', 32)
            
            model_lower = crear_modelo_nn(lower_q, X_train.shape[1], train_params)
            model_upper = crear_modelo_nn(upper_q, X_train.shape[1], train_params)
            model_lower.fit(X_train_scaled, y_train_lower, epochs=epochs, batch_size=batch_size, verbose=0)
            model_upper.fit(X_train_scaled, y_train_upper, epochs=epochs, batch_size=batch_size, verbose=0)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_scaled, verbose=0)[0,0], model_upper.predict(X_test_scaled, verbose=0)[0,0]

        lower_bound, upper_bound = current_price * (1 + lower_pct_pred), current_price * (1 + upper_pct_pred)
        actual_max_price, actual_min_price = test_df['close'].max(), test_df['close'].min()
        if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
            breaches += 1
            
    return (breaches / len(test_dates)) * 100

# --- FUNCIÓN ORQUESTADORA DE GRID SEARCH POR MODELO ---
def ejecutar_grid_search_para_modelo(features_df, nombre_modelo, param_grid, quantiles):
    logging.info(f"--- INICIANDO GRID SEARCH PARA: {nombre_modelo} ---")
    filename = f"grid_search_results_{nombre_modelo}.csv"
    try:
        df_resultados_previos = pd.read_csv(filename)
        param_cols = [col for col in df_resultados_previos.columns if col != 'breach_pct']
        params_ya_evaluados = set(tuple(sorted(d.items())) for d in df_resultados_previos[param_cols].to_dict('records'))
        logging.info(f"Cargados {len(params_ya_evaluados)} resultados previos para {nombre_modelo}.")
    except FileNotFoundError:
        df_resultados_previos = pd.DataFrame()
        params_ya_evaluados = set()
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    params_a_probar = [p for p in param_combinations if tuple(sorted(p.items())) not in params_ya_evaluados]

    if not params_a_probar:
        logging.info(f"No hay combinaciones nuevas para {nombre_modelo}. Usando resultados existentes.")
    else:
        logging.info(f"Total combinaciones para {nombre_modelo}: {len(param_combinations)}. Pendientes: {len(params_a_probar)}.")
        resultados_nuevos = []
        for params in tqdm(params_a_probar, desc=f"Optimizando {nombre_modelo}"):
            breach_pct = evaluar_combinacion(features_df, nombre_modelo, params, quantiles)
            result = {**params, 'breach_pct': breach_pct}
            resultados_nuevos.append(result)
            df_progreso = pd.concat([df_resultados_previos, pd.DataFrame(resultados_nuevos)], ignore_index=True)
            df_progreso.to_csv(filename, index=False)
    
    df_resultados_final = pd.read_csv(filename)
    
    lower_q, upper_q = quantiles
    target_breach_rate = (lower_q + (1 - upper_q)) * 100
    ganador_idx = (df_resultados_final['breach_pct'] - target_breach_rate).abs().idxmin()
    mejor_combinacion = df_resultados_final.loc[ganador_idx]
    
    print(f"\nResultados del Grid Search para {nombre_modelo}:")
    print(df_resultados_final.sort_values(by='breach_pct').to_string())
    print(f"\n🏆 Mejor combinación para {nombre_modelo}:")
    print(mejor_combinacion)
    
    best_params_dict = mejor_combinacion.drop('breach_pct').to_dict()
    return {
        'modelo': nombre_modelo,
        'best_params': best_params_dict,
        'breach_pct_optimizado': mejor_combinacion['breach_pct']
    }

# --- ORQUESTADOR PRINCIPAL ---
if __name__ == '__main__':
    param_grid_lgbm = {'n_estimators': [100, 250], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 40]}
    param_grid_xgb = {'n_estimators': [100, 250], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    param_grid_rf = {'n_estimators': [100, 250], 'max_depth': [10, 20], 'min_samples_leaf': [10, 20]}
    param_grid_nn = {'epochs': [20, 40], 'batch_size': [32, 64]}
    grids = {"LightGBM": param_grid_lgbm, "XGBoost": param_grid_xgb, "RandomForest": param_grid_rf, "NeuralNetwork": param_grid_nn}
    PARAMETROS_QUANTILES_OPTIMOS = (0.015, 0.985)
    
    merged_data = actualizar_datos()
    if merged_data is not None:
        features_data = preparar_datos_y_features(merged_data)
        
        campeones_de_cada_modelo = []
        for nombre_modelo, param_grid in grids.items():
            mejor_resultado = ejecutar_grid_search_para_modelo(features_data, nombre_modelo, param_grid, PARAMETROS_QUANTILES_OPTIMOS)
            campeones_de_cada_modelo.append(mejor_resultado)
        
        df_campeones = pd.DataFrame(campeones_de_cada_modelo)
        
        print("\n\n==================================================")
        print("🏁 RESULTADO FINAL DE LA CARRERA DE CABALLOS 🏁")
        print("==================================================")
        df_campeones['best_params'] = df_campeones['best_params'].astype(str)
        print(df_campeones.to_string())

        target_breach = (PARAMETROS_QUANTILES_OPTIMOS[0] + (1 - PARAMETROS_QUANTILES_OPTIMOS[1])) * 100
        ganador_final_idx = (df_campeones['breach_pct_optimizado'] - target_breach).abs().idxmin()
        ganador_final = df_campeones.loc[ganador_final_idx]

        print("\n==================================================")
        print(f"🎉 EL CAMPEÓN GENERAL ES: {ganador_final['modelo']} 🎉")
        print("==================================================")

        # --- GUARDAR EL CAMPEÓN EN UN ARCHIVO JSON ---
        campeon_a_guardar = {
            'model': ganador_final['modelo'],
            'params': ganador_final['best_params'],
            'quantiles': PARAMETROS_QUANTILES_OPTIMOS
        }
        if isinstance(campeon_a_guardar['params'], str):
            campeon_a_guardar['params'] = ast.literal_eval(campeon_a_guardar['params'])
        with open('champion_model.json', 'w') as f:
            json.dump(campeon_a_guardar, f, indent=4)
        logging.info("✅ Parámetros del modelo campeón guardados en 'champion_model.json'")