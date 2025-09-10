# ==============================================================================
# SCRIPT DEFINITIVO CON REPORTE DE CALIBRACIÓN Y BACKTEST FINAL (VERSIÓN 2.1)
# ==============================================================================
# - CORRECCIÓN: Solucionado ValueError: Input y contains NaN. El backtest ahora
#   limpia los valores nulos del target antes de entrenar.
# - Compatible con todos los modelos de árboles (LightGBM, XGBoost, RF, GBR).
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import json

# Imports para todos los modelos de árboles
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN Y PREPARACIÓN DE DATOS ---
warnings.filterwarnings('ignore')
log_filename = f"pronostico_log_{datetime.now().strftime('%Y%m%d')}.txt"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def actualizar_datos():
    """Carga los datos desde los archivos CSV locales."""
    logging.info("Paso 1: Cargando datos actualizados desde archivos CSV locales...")
    try:
        spy_df = pd.read_csv('spy_15y_daily_20250909.csv', index_col='date', parse_dates=True)
        vix_df = pd.read_csv('vix_15y_daily_20250909.csv', index_col='date', parse_dates=True)
        merged_df = pd.merge(spy_df[['open', 'high', 'low', 'close']], vix_df[['close']], left_index=True, right_index=True, suffixes=('_SPY', '_VIX'))
        merged_df.rename(columns={'close_SPY': 'close', 'close_VIX': 'vix_level'}, inplace=True)
        merged_df['returns'] = np.log(merged_df['close']).diff()
        return merged_df.dropna()
    except FileNotFoundError as e:
        logging.error(f"Error crítico al cargar archivos de datos: {e}"); raise

def preparar_datos_y_features(df):
    """Crea todas las características (features) necesarias para el modelo."""
    logging.info("Paso 2: Creando características y objetivos...")
    features_df = df.copy()
    features_df['volatility_21d'] = features_df['returns'].rolling(window=21).std()
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
    features_to_check = features_df.columns.drop(['target_lower_pct', 'target_upper_pct'])
    return features_df.dropna(subset=features_to_check)

# --- FUNCIÓN DE BACKTEST (CON CORRECCIÓN) ---
def ejecutar_backtest(features_df, quantiles_a_probar, model_name, model_params, desc=""):
    results_data = []
    backtest_start_date = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24)
    for lower_q, upper_q in tqdm(quantiles_a_probar, desc=desc):
        breaches, monthly_predictions, breach_magnitudes = 0, [], []
        test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index
        for trade_date in test_dates:
            history_df = features_df[features_df.index < trade_date].tail(504)
            evaluation_end = trade_date + pd.DateOffset(days=21)
            test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
            if history_df.empty or test_df.empty: continue
            
            features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
            X_test_df, current_price = test_df.head(1)[features_to_use], test_df.head(1)['close'].values[0]
            
            # --- INICIO DE LA CORRECCIÓN ---
            # Preparamos y limpiamos los datos de entrenamiento por separado para cada modelo (lower y upper)
            train_df_lower = history_df[features_to_use + ['target_lower_pct']].dropna()
            X_train_lower, y_train_lower = train_df_lower[features_to_use], train_df_lower['target_lower_pct']
            
            train_df_upper = history_df[features_to_use + ['target_upper_pct']].dropna()
            X_train_upper, y_train_upper = train_df_upper[features_to_use], train_df_upper['target_upper_pct']
            # --- FIN DE LA CORRECCIÓN ---

            lower_pct_pred, upper_pct_pred = 0, 0
            
            if model_name == 'LightGBM':
                model_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **model_params).fit(X_train_lower, y_train_lower)
                model_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **model_params).fit(X_train_upper, y_train_upper)
                lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]
            elif model_name == 'XGBoost':
                model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=lower_q, **model_params).fit(X_train_lower, y_train_lower)
                model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=upper_q, **model_params).fit(X_train_upper, y_train_upper)
                lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]
            elif model_name == 'GradientBoosting':
                model_lower = GradientBoostingRegressor(loss='quantile', alpha=lower_q, **model_params).fit(X_train_lower, y_train_lower)
                model_upper = GradientBoostingRegressor(loss='quantile', alpha=upper_q, **model_params).fit(X_train_upper, y_train_upper)
                lower_pct_pred, upper_pct_pred = model_lower.predict(X_test_df)[0], model_upper.predict(X_test_df)[0]
            elif model_name == 'RandomForest':
                model_lower = RandomForestRegressor(n_jobs=-1, **model_params).fit(X_train_lower, y_train_lower)
                model_upper = RandomForestRegressor(n_jobs=-1, **model_params).fit(X_train_upper, y_train_upper)
                preds_lower_trees = np.array([tree.predict(X_test_df) for tree in model_lower.estimators_])
                preds_upper_trees = np.array([tree.predict(X_test_df) for tree in model_upper.estimators_])
                lower_pct_pred, upper_pct_pred = np.quantile(preds_lower_trees, lower_q), np.quantile(preds_upper_trees, upper_q)

            lower_bound, upper_bound = current_price * (1 + lower_pct_pred), current_price * (1 + upper_pct_pred)
            monthly_predictions.append({'date': trade_date, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
            actual_max_price, actual_min_price = test_df['close'].max(), test_df['close'].min()
            if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
                breaches += 1
                magnitude = 0
                if actual_max_price > upper_bound: magnitude = (actual_max_price - upper_bound) / actual_max_price
                elif actual_min_price < lower_bound: magnitude = (lower_bound - actual_min_price) / actual_min_price
                breach_magnitudes.append(abs(magnitude))

        breach_percentage = (breaches / len(test_dates)) * 100 if len(test_dates) > 0 else 0
        max_error_pct = np.max(breach_magnitudes) * 100 if breach_magnitudes else 0
        avg_error_pct = np.mean(breach_magnitudes) * 100 if breach_magnitudes else 0
        results_data.append({'quantiles': (lower_q, upper_q), 'breach_pct': breach_percentage, 'max_breach_error_pct': max_error_pct, 'avg_breach_error_pct': avg_error_pct, 'backtest_df': pd.DataFrame(monthly_predictions).set_index('date')})
    return pd.DataFrame(results_data)

# --- OTRAS FUNCIONES ---
def entrenar_modelos(features_df, quantiles, model_name, model_params):
    # Esta función ya limpiaba los NaNs correctamente, no necesita cambios.
    train_df = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct'])
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
    X_train_df = train_df[features_to_use]
    models = {}
    logging.info(f"Entrenando modelos finales ({model_name}) con cuantiles: {quantiles}")
    
    for q in quantiles:
        y_train = train_df['target_lower_pct'] if q < 0.5 else train_df['target_upper_pct']
        model = None
        if model_name == 'LightGBM':
            model = lgb.LGBMRegressor(objective='quantile', alpha=q, verbosity=-1, **model_params)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **model_params)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(loss='quantile', alpha=q, **model_params)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(n_jobs=-1, **model_params)
        
        model.fit(X_train_df, y_train)
        models[f'model_{q}'] = model
    return models

def generar_pronostico(models, latest_data, model_name, model_params, quantiles):
    current_price = latest_data['close'].values[0]
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
    X_predict_df = latest_data[features_to_use]
    forecast_bounds = {}
    
    for q in quantiles:
        model = models[f'model_{q}']
        pct_pred = 0
        if model_name == 'RandomForest':
            preds_trees = np.array([tree.predict(X_predict_df) for tree in model.estimators_])
            pct_pred = np.quantile(preds_trees, q)
        else:
            pct_pred = model.predict(X_predict_df)[0]
        
        forecast_bounds[q] = current_price * (1 + pct_pred)
    return forecast_bounds, current_price

def graficar_pronostico_avanzado(latest_data, forecast_bounds, history_df, quantiles):
    logging.info("Generando gráfico del pronóstico...")
    last_date = latest_data.index[0]
    future_dates = pd.date_range(start=last_date, periods=22, freq='B')[1:]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 9))
    plt.plot(history_df.index[-180:], history_df['close'][-180:], label='Precio Histórico SPY', color='black', linewidth=2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    num_ranges = len(quantiles) // 2
    for i in range(num_ranges):
        lower_q, upper_q = quantiles[i], quantiles[-(i+1)]
        conf_level = (upper_q - lower_q) * 100
        lower_bound, upper_bound = forecast_bounds[lower_q], forecast_bounds[upper_q]
        label_text = f'Rango {conf_level:.0f}% (${lower_bound:.2f} - ${upper_bound:.2f})'
        plt.fill_between(future_dates, lower_bound, upper_bound, color=colors[i % len(colors)], alpha=0.2, label=label_text)
        plt.hlines(y=[lower_bound, upper_bound], xmin=future_dates[0], xmax=future_dates[-1], colors=colors[i % len(colors)], linestyles='--')
    plt.title(f'Pronóstico de Rangos de Probabilidad para SPY a 21 Días\nGenerado el {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
    plt.ylabel('Precio SPY ($)'), plt.xlabel('Fecha')
    plt.legend(loc='upper left'), plt.tight_layout()
    plt.savefig('pronostico_avanzado_actual.png', dpi=300)
    plt.show()

def graficar_backtest(results_df, backtest_start_date, merged_df, title):
    logging.info(f"Generando gráfico del backtest: {title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 9))
    plot_data = merged_df[merged_df.index >= backtest_start_date]
    plt.plot(plot_data.index, plot_data['close'], label='Precio Real del SPY', color='black', zorder=5)
    plt.plot(results_df.index, results_df['upper_bound'], color='red', linestyle='--', label='Límite Superior Pronosticado')
    plt.plot(results_df.index, results_df['lower_bound'], color='blue', linestyle='--', label='Límite Inferior Pronosticado')
    plt.fill_between(results_df.index, results_df['lower_bound'], results_df['upper_bound'], color='gray', alpha=0.2, label='Rango Pronosticado')
    plt.title(title, fontsize=16)
    plt.ylabel('Precio SPY ($)'), plt.xlabel('Fecha')
    plt.legend(), plt.tight_layout()
    plt.savefig('backtest_final.png', dpi=300)
    plt.show()

# --- ORQUESTADOR PRINCIPAL ---
if __name__ == '__main__':
    try:
        logging.info("==================================================")
        logging.info("🚀 INICIANDO SCRIPT DE PRONÓSTICO (VERSIÓN 2.1 - MODELOS DE ÁRBOLES)")
        logging.info("==================================================")

        try:
            with open('champion_model.json', 'r') as f: PARAMETROS_CAMPEONES = json.load(f)
            logging.info(f"Parámetros del campeón '{PARAMETROS_CAMPEONES['model']}' cargados.")
        except FileNotFoundError:
            logging.error("❌ No se encontró 'champion_model.json'."); raise
        
        logging.info("Validando y corrigiendo tipos de datos...")
        params_dict = PARAMETROS_CAMPEONES['params']
        int_params = ['n_estimators', 'num_leaves', 'max_depth', 'min_samples_leaf']
        for param in int_params:
            if param in params_dict and isinstance(params_dict[param], float):
                params_dict[param] = int(params_dict[param])
        
        RANGO_ACEPTABLE_RUPTURAS = (2.0, 10.0)
        merged_data = actualizar_datos()
        features_data = preparar_datos_y_features(merged_data)
        
        # ETAPA 1: VALIDACIÓN
        logging.info(f"--- FASE 1: VALIDANDO MODELO '{PARAMETROS_CAMPEONES['model']}' ---")
        df_validacion = ejecutar_backtest(features_data, [tuple(PARAMETROS_CAMPEONES['quantiles'])], PARAMETROS_CAMPEONES['model'], PARAMETROS_CAMPEONES['params'], desc="Validando Parámetros").iloc[0]
        logging.info(f"Tasa de ruptura (Breach Rate): {df_validacion['breach_pct']:.2f}%")
        logging.info(f"Error Máximo en rupturas: {df_validacion['max_breach_error_pct']:.2f}%")
        logging.info(f"Error Promedio en rupturas: {df_validacion['avg_breach_error_pct']:.2f}%")

        # ETAPA 2: AUTO-CALIBRACIÓN
        parametros_finales = tuple(PARAMETROS_CAMPEONES['quantiles'])
        if not (RANGO_ACEPTABLE_RUPTURAS[0] <= df_validacion['breach_pct'] <= RANGO_ACEPTABLE_RUPTURAS[1]):
            logging.warning(f"⚠️ ¡Alerta de Calibración!")
        else:
            logging.info("✅ Validación exitosa. El modelo está bien calibrado.")

        # ETAPA 3: BACKTEST FINAL Y GRÁFICO
        logging.info(f"--- FASE 3: BACKTEST FINAL CON CUANTILES {parametros_finales} ---")
        df_backtest_final = ejecutar_backtest(features_data, [parametros_finales], PARAMETROS_CAMPEONES['model'], PARAMETROS_CAMPEONES['params'], desc="Ejecutando Backtest Final").iloc[0]
        titulo_backtest = (f"Backtest Final (24 Meses) - Modelo: {PARAMETROS_CAMPEONES['model']}\n"
                           f"Tasa de Ruptura: {df_backtest_final['breach_pct']:.2f}% | Error Máx: {df_backtest_final['max_breach_error_pct']:.2f}%")
        graficar_backtest(df_backtest_final['backtest_df'], features_data.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24), merged_data, titulo_backtest)

        # ETAPA 4: PRONÓSTICO FINAL
        logging.info(f"--- FASE 4: GENERANDO PRONÓSTICO FINAL ---")
        QUANTILES_A_PREDECIR = sorted(list(set([parametros_finales[0], parametros_finales[1], 0.05, 0.95, 0.075, 0.925, 0.10, 0.90, 0.125, 0.875, 0.15, 0.85])))
        trained_models = entrenar_modelos(features_data.dropna(subset=['target_lower_pct', 'target_upper_pct']), QUANTILES_A_PREDECIR, PARAMETROS_CAMPEONES['model'], PARAMETROS_CAMPEONES['params'])
        latest_data_point = features_data.tail(1)
        forecast, last_price = generar_pronostico(trained_models, latest_data_point, PARAMETROS_CAMPEONES['model'], PARAMETROS_CAMPEONES['params'], QUANTILES_A_PREDECIR)
        graficar_pronostico_avanzado(latest_data_point, forecast, merged_data, QUANTILES_A_PREDECIR)
        
        logging.info("==================================================")
        logging.info("🎉 PROCESO DE PRONÓSTICO COMPLETADO EXITOSAMENTE 🎉")
        logging.info("==================================================")
        
    except Exception as e:
        logging.critical(f"❌ El script falló con un error inesperado: {e}", exc_info=True)