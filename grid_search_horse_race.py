# ==============================================================================
# SCRIPT DE OPTIMIZACIÓN v2 (CON NUEVO MODELO Y DATA SPLIT)
# ==============================================================================
# - Agrega QuantileRegressor de Scikit-learn como un nuevo modelo a probar.
# - Separa los datos para que la optimización solo se realice en el
#   conjunto de entrenamiento/validación, reservando datos para una prueba final.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import warnings
import json

# Imports de Modelos y Herramientas
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor ## NUEVO ##
import optuna

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- PREPARACIÓN DE DATOS (sin cambios) ---
def actualizar_datos():
    logging.info("Paso 1: Cargando datos...")
    try:
        spy_df = pd.read_csv('spy_15y_daily_20250912.csv', index_col='date', parse_dates=True)
        vix_df = pd.read_csv('vix_15y_daily_20250912.csv', index_col='date', parse_dates=True)
        merged_df = pd.merge(spy_df[['open', 'high', 'low', 'close']], vix_df[['close']], left_index=True, right_index=True, suffixes=('_SPY', '_VIX'))
        merged_df.rename(columns={'close_SPY': 'close', 'close_VIX': 'vix_level'}, inplace=True)
        merged_df['returns_SPY'] = np.log(merged_df['close']).diff()
        return merged_df.dropna()
    except FileNotFoundError as e:
        logging.error(f"Error crítico: No se encontraron los archivos de datos. {e}"); return None

def preparar_datos_y_features(df):
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
    features_df['vix_vol_ratio'] = features_df['vix_level'] / (features_df['volatility_21d'] * 100 * np.sqrt(252))
    features_df['vix_vol_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

    horizon = 21
    future_min = features_df['close'].shift(-horizon).rolling(window=horizon).min()
    future_max = features_df['close'].shift(-horizon).rolling(window=horizon).max()
    features_df['target_lower_pct'] = (future_min - features_df['close']) / features_df['close']
    features_df['target_upper_pct'] = (future_max - features_df['close']) / features_df['close']
    
    features_df.dropna(inplace=True)
    return features_df

# --- FUNCIÓN DE EVALUACIÓN (OBJETIVO PARA OPTUNA) ---
def evaluar_combinacion(features_df, nombre_modelo, params, quantiles):
    breaches = 0
    lower_q, upper_q = quantiles
    backtest_start_date = features_df.index[-1] - pd.DateOffset(months=24)
    test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index
    breach_magnitudes = []
    
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d', 'vix_vol_ratio']

    for trade_date in test_dates:
        history_df = features_df[features_df.index < trade_date].tail(504)
        evaluation_end = trade_date + pd.DateOffset(days=21)
        test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
        if history_df.empty or test_df.empty: continue

        X_train_df = history_df[features_to_use].dropna()
        y_train_lower = history_df.loc[X_train_df.index, 'target_lower_pct']
        y_train_upper = history_df.loc[X_train_df.index, 'target_upper_pct']

        X_test_df = test_df.head(1)[features_to_use]
        current_price = test_df.head(1)['close'].values[0]

        train_params = params.copy()
        lower_pct_pred, upper_pct_pred = 0, 0

        if nombre_modelo == 'LightGBM':
            model_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **train_params).fit(X_train_df, y_train_lower)
            model_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **train_params).fit(X_train_df, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]
        elif nombre_modelo == 'XGBoost':
            model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=lower_q, **train_params).fit(X_train_df, y_train_lower)
            model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=upper_q, **train_params).fit(X_train_df, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]
        ## NUEVO: Lógica para QuantileRegressor ##
        elif nombre_modelo == 'QuantileRegressor':
            model_lower = QuantileRegressor(quantile=lower_q, solver='highs', **train_params).fit(X_train_df, y_train_lower)
            model_upper = QuantileRegressor(quantile=upper_q, solver='highs', **train_params).fit(X_train_df, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]

        lower_bound, upper_bound = current_price * (1 + lower_pct_pred), current_price * (1 + upper_pct_pred)
        actual_max_price, actual_min_price = test_df['close'].max(), test_df['close'].min()
        if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
            breaches += 1
            magnitude = 0
            if actual_max_price > upper_bound: magnitude = (actual_max_price - upper_bound) / actual_max_price
            elif actual_min_price < lower_bound: magnitude = (lower_bound - actual_min_price) / actual_min_price
            breach_magnitudes.append(abs(magnitude))

    breach_percentage = (breaches / len(test_dates)) * 100 if len(test_dates) > 0 else 0
    max_error_pct = np.max(breach_magnitudes) * 100 if breach_magnitudes else 0
    
    target_breach_rate = (quantiles[0] + (1 - quantiles[1])) * 100
    score = abs(breach_percentage - target_breach_rate) + (max_error_pct / 10)
    if max_error_pct > 20.0: score += 100

    return score, breach_percentage, max_error_pct

# --- ORQUESTADOR PRINCIPAL ---
if __name__ == '__main__':
    PARAMETROS_QUANTILES_OPTIMOS = (0.015, 0.985)
    N_TRIALS_OPTUNA = 50 
    ## NUEVO: Definir fecha de corte para separar datos de prueba final ##
    FECHA_CORTE_ENTRENAMIENTO = '2023-12-31'

    def objective_factory(model_name, features_df):
        def objective(trial):
            if model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                }
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                }
            ## NUEVO: Espacio de búsqueda para QuantileRegressor ##
            elif model_name == 'QuantileRegressor':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True)
                }
            else:
                params = {}

            score, _, _ = evaluar_combinacion(features_df, model_name, params, PARAMETROS_QUANTILES_OPTIMOS)
            return score
        return objective

    merged_data = actualizar_datos()
    if merged_data is not None:
        features_data = preparar_datos_y_features(merged_data)
        
        ## MODIFICADO: Usar solo datos hasta la fecha de corte para la optimización ##
        features_data_train_val = features_data.loc[features_data.index <= FECHA_CORTE_ENTRENAMIENTO]
        logging.info(f"Optimización se realizará con datos hasta {FECHA_CORTE_ENTRENAMIENTO}")
        
        ## MODIFICADO: Agregar QuantileRegressor a la lista de modelos ##
        modelos_a_probar = ["LightGBM", "XGBoost", "QuantileRegressor"]
        campeones_de_cada_modelo = []

        for nombre_modelo in modelos_a_probar:
            logging.info(f"--- INICIANDO OPTIMIZACIÓN PARA: {nombre_modelo} ---")
            study = optuna.create_study(direction='minimize')
            objective_func = objective_factory(nombre_modelo, features_data_train_val)
            study.optimize(objective_func, n_trials=N_TRIALS_OPTUNA)
            
            best_params = study.best_params
            score, breach_pct, max_error = evaluar_combinacion(features_data_train_val, nombre_modelo, best_params, PARAMETROS_QUANTILES_OPTIMOS)
            
            resultado = {
                'modelo': nombre_modelo, 
                'best_params': best_params,
                'score_optimizado': score,
                'breach_pct_optimizado': breach_pct, 
                'max_error_optimizado': max_error
            }
            campeones_de_cada_modelo.append(resultado)

        df_campeones = pd.DataFrame(campeones_de_cada_modelo)
        
        logging.info("--- SELECCIONANDO ENSAMBLE DE CAMPEONES ---")
        df_campeones_validos = df_campeones[df_campeones['max_error_optimizado'] < 20.0].sort_values(by='score_optimizado')

        if len(df_campeones_validos) < 2:
            raise ValueError("No se encontraron suficientes modelos válidos para crear un ensamble.")

        ensamble_final = df_campeones_validos.head(2).to_dict('records')
        print("\n🏆 ENSAMBLE GANADOR (2 MEJORES MODELOS) 🏆:")
        print(df_campeones)
        
        ensamble_a_guardar = {
            'ensemble_models': ensamble_final,
            'quantiles': PARAMETROS_QUANTILES_OPTIMOS
        }
        
        with open('ensemble_champion_v2.json', 'w') as f:
            json.dump(ensamble_a_guardar, f, indent=4)
        logging.info("✅ Parámetros del ensamble campeón guardados en 'ensemble_champion_v2.json'")