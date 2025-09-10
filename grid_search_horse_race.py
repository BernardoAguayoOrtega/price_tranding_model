# ==============================================================================
# SCRIPT DE OPTIMIZACIÓN Y COMPARACIÓN DE MODELOS (VERSIÓN FINAL)
# ==============================================================================
# - Compara 4 modelos robustos basados en árboles: LightGBM, XGBoost, RandomForest,
#   y el clásico GradientBoostingRegressor de Scikit-Learn.
# - La selección del campeón penaliza a los modelos con errores catastróficos.
# - Guarda al campeón final en 'champion_model.json' para su uso diario.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from itertools import product
import warnings
import json
import ast
from tqdm import tqdm

# Imports de Modelos
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PREPARACIÓN DE DATOS ---
def actualizar_datos():
    """Carga los datos desde los archivos CSV locales."""
    logging.info("Paso 1: Cargando datos...")
    try:
        spy_df = pd.read_csv('spy_15y_daily_20250909.csv', index_col='date', parse_dates=True)
        vix_df = pd.read_csv('vix_15y_daily_20250909.csv', index_col='date', parse_dates=True)
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

# --- FUNCIÓN DE EVALUACIÓN REUTILIZABLE ---
def evaluar_combinacion(features_df, nombre_modelo, params, quantiles):
    """Ejecuta un backtest completo y calcula la magnitud del error de forma robusta."""
    breaches = 0
    lower_q, upper_q = quantiles
    backtest_start_date = features_df.index[-1] - pd.DateOffset(months=24)
    test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index
    breach_magnitudes = []

    for trade_date in test_dates:
        history_df = features_df[features_df.index < trade_date].tail(504)
        evaluation_end = trade_date + pd.DateOffset(days=21)
        test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
        if history_df.empty or test_df.empty: continue
        
        features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
        X_train_df, y_train_lower, y_train_upper = history_df[features_to_use], history_df['target_lower_pct'], history_df['target_upper_pct']
        X_test_df, current_price = test_df.head(1)[features_to_use], test_df.head(1)['close'].values[0]

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

        elif nombre_modelo == 'GradientBoosting':
            model_lower = GradientBoostingRegressor(loss='quantile', alpha=lower_q, **train_params).fit(X_train_df, y_train_lower)
            model_upper = GradientBoostingRegressor(loss='quantile', alpha=upper_q, **train_params).fit(X_train_df, y_train_upper)
            lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]

        elif nombre_modelo == 'RandomForest':
            model_lower = RandomForestRegressor(n_jobs=-1, **train_params).fit(X_train_df, y_train_lower)
            model_upper = RandomForestRegressor(n_jobs=-1, **train_params).fit(X_train_df, y_train_upper)
            preds_lower_trees = np.array([tree.predict(X_test_df) for tree in model_lower.estimators_])
            preds_upper_trees = np.array([tree.predict(X_test_df) for tree in model_upper.estimators_])
            lower_pct_pred, upper_pct_pred = np.quantile(preds_lower_trees, lower_q), np.quantile(preds_upper_trees, upper_q)

        lower_bound, upper_bound = current_price * (1 + lower_pct_pred), current_price * (1 + upper_pct_pred)
        actual_max_price, actual_min_price = test_df['close'].max(), test_df['close'].min()
        if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
            breaches += 1
            magnitude = 0
            if actual_max_price > upper_bound:
                magnitude = (actual_max_price - upper_bound) / actual_max_price
            elif actual_min_price < lower_bound:
                magnitude = (lower_bound - actual_min_price) / actual_min_price
            breach_magnitudes.append(abs(magnitude))
            
    breach_percentage = (breaches / len(test_dates)) * 100 if len(test_dates) > 0 else 0
    max_error_pct = np.max(breach_magnitudes) * 100 if breach_magnitudes else 0
    
    return {'breach_pct': breach_percentage, 'max_breach_error_pct': max_error_pct}

# --- FUNCIÓN ORQUESTADORA DE GRID SEARCH POR MODELO ---
def ejecutar_grid_search_para_modelo(features_df, nombre_modelo, param_grid, quantiles):
    logging.info(f"--- INICIANDO GRID SEARCH PARA: {nombre_modelo} ---")
    filename = f"grid_search_results_{nombre_modelo}.csv"
    try:
        df_resultados_previos = pd.read_csv(filename)
        param_cols = [col for col in df_resultados_previos.columns if 'breach' not in col]
        params_ya_evaluados = set(tuple(sorted(d.items())) for d in df_resultados_previos[param_cols].to_dict('records'))
        logging.info(f"Cargados {len(params_ya_evaluados)} resultados previos para {nombre_modelo}.")
    except FileNotFoundError:
        df_resultados_previos = pd.DataFrame()
        params_ya_evaluados = set()
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    params_a_probar = [p for p in param_combinations if tuple(sorted(p.items())) not in params_ya_evaluados]

    if not params_a_probar:
        logging.info(f"No hay combinaciones nuevas para {nombre_modelo}.")
    else:
        logging.info(f"Total combinaciones para {nombre_modelo}: {len(param_combinations)}. Pendientes: {len(params_a_probar)}.")
        resultados_nuevos = []
        for params in tqdm(params_a_probar, desc=f"Optimizando {nombre_modelo}"):
            resultados_eval = evaluar_combinacion(features_df, nombre_modelo, params, quantiles)
            result = {**params, **resultados_eval}
            resultados_nuevos.append(result)
            df_progreso = pd.concat([df_resultados_previos, pd.DataFrame(resultados_nuevos)], ignore_index=True)
            df_progreso.to_csv(filename, index=False)
    
    df_resultados_final = pd.read_csv(filename)
    
    # Lógica de selección de ganador (penalizando errores grandes)
    df_validos = df_resultados_final[df_resultados_final['max_breach_error_pct'] < 20.0] # Descalifica modelos con errores > 20%
    if df_validos.empty:
        logging.warning(f"Ninguna combinación de {nombre_modelo} pasó el filtro de error máximo. Seleccionando el de menor error.")
        mejor_combinacion = df_resultados_final.sort_values(by='max_breach_error_pct').iloc[0]
    else:
        lower_q, upper_q = quantiles
        target_breach_rate = (lower_q + (1 - upper_q)) * 100
        ganador_idx = (df_validos['breach_pct'] - target_breach_rate).abs().idxmin()
        mejor_combinacion = df_validos.loc[ganador_idx]
    
    print(f"\nResultados del Grid Search para {nombre_modelo}:")
    print(df_resultados_final.sort_values(by=['breach_pct', 'max_breach_error_pct']).to_string())
    print(f"\n🏆 Mejor combinación para {nombre_modelo} (considerando errores):")
    print(mejor_combinacion)
    
    best_params_dict = mejor_combinacion.drop([col for col in mejor_combinacion.index if 'breach' in col]).to_dict()
    return {'modelo': nombre_modelo, 'best_params': best_params_dict, 'breach_pct_optimizado': mejor_combinacion['breach_pct'], 'max_error_optimizado': mejor_combinacion['max_breach_error_pct']}

# --- ORQUESTADOR PRINCIPAL ---
if __name__ == '__main__':
    param_grid_lgbm = {'n_estimators': [100, 250], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 40]}
    param_grid_xgb = {'n_estimators': [100, 250], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    param_grid_rf = {'n_estimators': [100, 250], 'max_depth': [10, 20], 'min_samples_leaf': [10, 20]}
    param_grid_gbr = {'n_estimators': [100, 250], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    
    grids = {
        "LightGBM": param_grid_lgbm,
        "XGBoost": param_grid_xgb,
        "RandomForest": param_grid_rf,
        "GradientBoosting": param_grid_gbr
    }
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

        # Selección final del campeón general, penalizando errores máximos altos
        df_validos_final = df_campeones[df_campeones['max_error_optimizado'] < 20.0]
        if df_validos_final.empty:
            logging.warning("Ningún modelo final pasó el filtro de error máximo. Seleccionando el de menor error.")
            ganador_final = df_campeones.sort_values(by='max_error_optimizado').iloc[0]
        else:
            target_breach = (PARAMETROS_QUANTILES_OPTIMOS[0] + (1 - PARAMETROS_QUANTILES_OPTIMOS[1])) * 100
            ganador_final_idx = (df_validos_final['breach_pct_optimizado'] - target_breach).abs().idxmin()
            ganador_final = df_validos_final.loc[ganador_final_idx]

        print("\n==================================================")
        print(f"🎉 EL CAMPEÓN GENERAL ES: {ganador_final['modelo']} 🎉")
        print("==================================================")

        campeon_a_guardar = {'model': ganador_final['modelo'], 'params': ganador_final['best_params'], 'quantiles': PARAMETROS_QUANTILES_OPTIMOS}
        if isinstance(campeon_a_guardar['params'], str):
            campeon_a_guardar['params'] = ast.literal_eval(campeon_a_guardar['params'])

        params_dict = campeon_a_guardar['params']
        int_params = ['n_estimators', 'num_leaves', 'max_depth', 'min_samples_leaf', 'n_pasos', 'lstm_units', 'dense_units']
        for param in int_params:
            if param in params_dict:
                params_dict[param] = int(params_dict[param])
        
        with open('champion_model.json', 'w') as f:
            json.dump(campeon_a_guardar, f, indent=4)
        logging.info("✅ Parámetros del modelo campeón (con tipos corregidos) guardados en 'champion_model.json'")