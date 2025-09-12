# ==============================================================================
# SCRIPT DE PRONÓSTICO DEFINITIVO CON ENSAMBLE Y CALIBRACIÓN (VERSIÓN 3.1)
# ==============================================================================
# - Carga el ensamble de modelos desde 'ensemble_champion.json'.
# - Ejecuta un backtest de validación COMPLETO sobre el ensamble.
# - Implementa calibración dinámica para ajustar los cuantiles si es necesario.
# - Genera el pronóstico final promediando las predicciones del ensamble.
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

# --- CONFIGURACIÓN Y PREPARACIÓN DE DATOS ---
warnings.filterwarnings('ignore')
log_filename = f"pronostico_log_{datetime.now().strftime('%Y%m%d')}.txt"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def actualizar_datos():
    """Carga los datos desde los archivos CSV locales."""
    logging.info("Paso 1: Cargando datos actualizados desde archivos CSV locales...")
    try:
        spy_df = pd.read_csv('spy_15y_daily_20250912.csv', index_col='date', parse_dates=True)
        vix_df = pd.read_csv('vix_15y_daily_20250912.csv', index_col='date', parse_dates=True)
        merged_df = pd.merge(spy_df[['open', 'high', 'low', 'close']], vix_df[['close']], left_index=True, right_index=True, suffixes=('_SPY', '_VIX'))
        merged_df.rename(columns={'close_SPY': 'close', 'close_VIX': 'vix_level'}, inplace=True)
        merged_df['returns_SPY'] = np.log(merged_df['close']).diff()
        return merged_df.dropna()
    except FileNotFoundError as e:
        logging.error(f"Error crítico al cargar archivos de datos: {e}"); raise

def preparar_datos_y_features(df):
    """Crea todas las características (features) necesarias para el modelo."""
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
    
    features_to_check = features_df.columns.drop(['target_lower_pct', 'target_upper_pct'])
    return features_df.dropna(subset=features_to_check)

# --- FUNCIÓN DE BACKTEST COMPLETA PARA ENSAMBLE ---
def ejecutar_backtest_ensamble(features_df, quantiles, ensemble_configs, desc="Validando Ensamble"):
    """
    Ejecuta un backtest completo para un ensamble de modelos, promediando sus predicciones.
    """
    lower_q, upper_q = quantiles
    breaches = 0
    breach_magnitudes = []
    monthly_predictions = []
    
    backtest_start_date = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24)
    test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index

    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d', 'vix_vol_ratio']

    for trade_date in tqdm(test_dates, desc=desc):
        history_df = features_df[features_df.index < trade_date].tail(504)
        evaluation_end = trade_date + pd.DateOffset(days=21)
        test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
        
        if history_df.empty or test_df.empty: continue

        train_df_cleaned = history_df[features_to_use + ['target_lower_pct', 'target_upper_pct']].dropna()
        X_train_df = train_df_cleaned[features_to_use]
        y_train_lower = train_df_cleaned['target_lower_pct']
        y_train_upper = train_df_cleaned['target_upper_pct']
        
        if X_train_df.empty: continue
        
        X_test_df = test_df.head(1)[features_to_use]
        current_price = test_df.head(1)['close'].values[0]

        ensemble_lower_preds = []
        ensemble_upper_preds = []

        for config in ensemble_configs:
            model_name = config['modelo']
            params = config['best_params']
            
            if model_name == 'LightGBM':
                model_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **params).fit(X_train_df, y_train_lower)
                model_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **params).fit(X_train_df, y_train_upper)
            elif model_name == 'XGBoost':
                model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=lower_q, **params).fit(X_train_df, y_train_lower)
                model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=upper_q, **params).fit(X_train_df, y_train_upper)
            
            ensemble_lower_preds.append(model_lower.predict(X_test_df)[0])
            ensemble_upper_preds.append(model_upper.predict(X_test_df)[0])

        avg_lower_pct_pred = np.mean(ensemble_lower_preds)
        avg_upper_pct_pred = np.mean(ensemble_upper_preds)

        lower_bound = current_price * (1 + avg_lower_pct_pred)
        upper_bound = current_price * (1 + avg_upper_pct_pred)
        monthly_predictions.append({'date': trade_date, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
        
        actual_max_price = test_df['close'].max()
        actual_min_price = test_df['close'].min()

        if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
            breaches += 1
            magnitude = 0
            if actual_max_price > upper_bound: magnitude = (actual_max_price - upper_bound) / actual_max_price
            elif actual_min_price < lower_bound: magnitude = (lower_bound - actual_min_price) / actual_min_price
            breach_magnitudes.append(abs(magnitude))

    breach_percentage = (breaches / len(test_dates)) * 100 if len(test_dates) > 0 else 0
    max_error_pct = np.max(breach_magnitudes) * 100 if breach_magnitudes else 0
    avg_error_pct = np.mean(breach_magnitudes) * 100 if breach_magnitudes else 0

    return {
        'breach_pct': breach_percentage, 
        'max_breach_error_pct': max_error_pct, 
        'avg_breach_error_pct': avg_error_pct, 
        'backtest_df': pd.DataFrame(monthly_predictions).set_index('date')
    }

# --- FUNCIONES DE PRONÓSTICO Y GRÁFICOS ---
def entrenar_y_predecir_ensamble(features_df, latest_data, quantiles, ensemble_configs):
    logging.info(f"Entrenando y generando pronóstico con ensamble de {len(ensemble_configs)} modelos.")
    all_forecasts = []
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d', 'vix_vol_ratio']
    
    train_df = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct'])
    X_train_df = train_df[features_to_use]
    X_predict_df = latest_data[features_to_use]
    current_price = latest_data['close'].values[0]

    for config in ensemble_configs:
        model_name = config['modelo']
        model_params = config['best_params']
        forecast_bounds = {}

        for q in quantiles:
            y_train = train_df['target_lower_pct'] if q < 0.5 else train_df['target_upper_pct']
            model = None
            if model_name == 'LightGBM':
                model = lgb.LGBMRegressor(objective='quantile', alpha=q, verbosity=-1, **model_params)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **model_params)
            
            model.fit(X_train_df, y_train)
            pct_pred = model.predict(X_predict_df)[0]
            forecast_bounds[q] = current_price * (1 + pct_pred)
        
        all_forecasts.append(forecast_bounds)

    final_forecast = {}
    for q in quantiles:
        final_forecast[q] = np.mean([f[q] for f in all_forecasts])

    return final_forecast, current_price

def graficar_pronostico_avanzado(latest_data, forecast_bounds, history_df, quantiles):
    logging.info("Generando gráfico del pronóstico...")
    last_date = latest_data.index[0]
    future_dates = pd.date_range(start=last_date, periods=22, freq='B')[1:]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 9))
    plt.plot(history_df.index[-180:], history_df['close'][-180:], label='Precio Histórico SPY', color='black', linewidth=2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
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
        logging.info("🚀 INICIANDO SCRIPT DE PRONÓSTICO CON ENSAMBLE (V3.1)")
        logging.info("==================================================")

        with open('ensemble_champion.json', 'r') as f:
            PARAMETROS_ENSAMBLE = json.load(f)
        ensemble_models = PARAMETROS_ENSAMBLE['ensemble_models']
        logging.info(f"Parámetros de ensamble cargados ({len(ensemble_models)} modelos).")

        RANGO_ACEPTABLE_RUPTURAS = (2.0, 10.0)
        merged_data = actualizar_datos()
        features_data = preparar_datos_y_features(merged_data)

        # ETAPA 1 Y 2: VALIDACIÓN Y CALIBRACIÓN DINÁMICA
        logging.info("--- FASE 1 & 2: VALIDACIÓN Y CALIBRACIÓN DINÁMICA ---")
        parametros_finales = tuple(PARAMETROS_ENSAMBLE['quantiles'])
        df_validacion = ejecutar_backtest_ensamble(features_data, parametros_finales, ensemble_models)
        
        breach_pct_actual = df_validacion['breach_pct']
        logging.info(f"Tasa de ruptura (backtest inicial): {breach_pct_actual:.2f}%")
        logging.info(f"Error Máximo en rupturas: {df_validacion['max_breach_error_pct']:.2f}%")
        logging.info(f"Error Promedio en rupturas: {df_validacion['avg_breach_error_pct']:.2f}%")
        
        if not (RANGO_ACEPTABLE_RUPTURAS[0] <= breach_pct_actual <= RANGO_ACEPTABLE_RUPTURAS[1]):
            logging.warning(f"⚠️ ¡ALERTA DE CALIBRACIÓN! Tasa ({breach_pct_actual:.2f}%) fuera del rango {RANGO_ACEPTABLE_RUPTURAS}.")
            
            target_midpoint = np.mean(RANGO_ACEPTABLE_RUPTURAS)
            ajuste = (breach_pct_actual - target_midpoint) / 100 / 2 # Dividimos entre 2 para no sobreajustar
            
            nuevo_lower_q = max(0.005, parametros_finales[0] - ajuste)
            nuevo_upper_q = min(0.995, parametros_finales[1] + ajuste)
            parametros_finales = (round(nuevo_lower_q, 3), round(nuevo_upper_q, 3))
            
            logging.info(f"Ajustando cuantiles dinámicamente a: {parametros_finales}")
        else:
            logging.info("✅ Validación exitosa. El modelo está bien calibrado.")

        # ETAPA 3: BACKTEST FINAL Y GRÁFICO
        logging.info(f"--- FASE 3: BACKTEST FINAL CON CUANTILES {parametros_finales} ---")
        df_backtest_final = ejecutar_backtest_ensamble(features_data, parametros_finales, ensemble_models, desc="Ejecutando Backtest Final")
        titulo_backtest = (f"Backtest Final (24 Meses) - Ensamble\n"
                           f"Tasa de Ruptura: {df_backtest_final['breach_pct']:.2f}% | Error Máx: {df_backtest_final['max_breach_error_pct']:.2f}%")
        backtest_start_date = features_data.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24)
        graficar_backtest(df_backtest_final['backtest_df'], backtest_start_date, merged_data, titulo_backtest)

        # ETAPA 4: PRONÓSTICO FINAL
        logging.info(f"--- FASE 4: GENERANDO PRONÓSTICO FINAL ---")
        QUANTILES_A_PREDECIR = sorted(list(set([parametros_finales[0], parametros_finales[1], 0.05, 0.95, 0.10, 0.90])))
        latest_data_point = features_data.tail(1)
        forecast, last_price = entrenar_y_predecir_ensamble(features_data, latest_data_point, QUANTILES_A_PREDECIR, ensemble_models)
        graficar_pronostico_avanzado(latest_data_point, forecast, merged_data, QUANTILES_A_PREDECIR)

        logging.info("==================================================")
        logging.info("🎉 PROCESO DE PRONÓSTICO COMPLETADO EXITOSAMENTE 🎉")
        logging.info("==================================================")

    except Exception as e:
        logging.critical(f"❌ El script falló con un error inesperado: {e}", exc_info=True)