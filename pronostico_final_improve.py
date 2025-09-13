# ==============================================================================
# SCRIPT DE PRONÓSTICO FINAL (v5)
# ==============================================================================
# - Versión final compatible con los 5 modelos optimizados.
# - Carga el ensamble desde 'ensemble_champion_v3.json'.
# - Implementa el backtesting robusto de 3 etapas: Entrenamiento, Validación y
#   Prueba Final (Out-of-Sample).
# - La calibración se realiza SOLO en el set de Validación.
# - El rendimiento final se reporta sobre el set de Prueba.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import json

# Imports para todos los modelos
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

# --- CONFIGURACIÓN Y PREPARACIÓN DE DATOS ---
warnings.filterwarnings('ignore')
log_filename = f"pronostico_log_{datetime.now().strftime('%Y%m%d')}.txt"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def actualizar_datos():
    """Carga los datos desde los archivos CSV locales."""
    logging.info("Paso 1: Cargando datos actualizados...")
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
def ejecutar_backtest_ensamble(features_df, quantiles, ensemble_configs, start_date, end_date, desc="Backtesting Ensamble"):
    lower_q, upper_q = quantiles
    breaches = 0
    breach_magnitudes = []
    monthly_predictions = []
    
    test_dates = features_df.loc[start_date:end_date].resample('MS').first().index

    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d', 'vix_vol_ratio']

    for trade_date in tqdm(test_dates, desc=desc):
        history_df = features_df[features_df.index < trade_date].tail(504)
        evaluation_end = trade_date + pd.DateOffset(days=21)
        test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
        
        if history_df.empty or test_df.empty: continue

        train_df_cleaned = history_df[features_to_use + ['target_lower_pct', 'target_upper_pct']].dropna()
        X_train_df, y_train_lower, y_train_upper = train_df_cleaned[features_to_use], train_df_cleaned['target_lower_pct'], train_df_cleaned['target_upper_pct']
        
        if X_train_df.empty: continue
        
        X_test_df = test_df.head(1)[features_to_use]
        current_price = test_df.head(1)['close'].values[0]

        ensemble_lower_preds, ensemble_upper_preds = [], []

        for config in ensemble_configs:
            model_name, params = config['modelo'], config['best_params']
            model_lower, model_upper = None, None
            
            if model_name == 'LightGBM':
                model_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **params).fit(X_train_df, y_train_lower)
                model_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **params).fit(X_train_df, y_train_upper)
            elif model_name == 'XGBoost':
                model_lower = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=lower_q, **params).fit(X_train_df, y_train_lower)
                model_upper = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=upper_q, **params).fit(X_train_df, y_train_upper)
            elif model_name == 'RandomForest':
                model_lower = RandomForestRegressor(random_state=42, **params).fit(X_train_df, y_train_lower)
                model_upper = RandomForestRegressor(random_state=42, **params).fit(X_train_df, y_train_upper)
            elif model_name == 'GradientBoosting':
                model_lower = GradientBoostingRegressor(loss='quantile', alpha=lower_q, random_state=42, **params).fit(X_train_df, y_train_lower)
                model_upper = GradientBoostingRegressor(loss='quantile', alpha=upper_q, random_state=42, **params).fit(X_train_df, y_train_upper)
            elif model_name == 'QuantileRegressor':
                model_lower = QuantileRegressor(quantile=lower_q, solver='highs', **params).fit(X_train_df, y_train_lower)
                model_upper = QuantileRegressor(quantile=upper_q, solver='highs', **params).fit(X_train_df, y_train_upper)

            if model_lower and model_upper:
                ensemble_lower_preds.append(model_lower.predict(X_test_df)[0])
                ensemble_upper_preds.append(model_upper.predict(X_test_df)[0])

        if not ensemble_lower_preds or not ensemble_upper_preds: continue

        avg_lower_pct_pred = np.mean(ensemble_lower_preds)
        avg_upper_pct_pred = np.mean(ensemble_upper_preds)

        lower_bound = current_price * (1 + avg_lower_pct_pred)
        upper_bound = current_price * (1 + avg_upper_pct_pred)
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

    return {
        'breach_pct': breach_percentage, 
        'max_breach_error_pct': max_error_pct, 
        'avg_breach_error_pct': avg_error_pct, 
        'backtest_df': pd.DataFrame(monthly_predictions).set_index('date')
    }

# --- FUNCIONES DE PRONÓSTICO Y GRÁFICOS ---
def entrenar_y_predecir_ensamble(features_df, latest_data, quantiles, ensemble_configs):
    """Entrena el ensamble con todos los datos y genera el pronóstico final."""
    all_forecasts = []
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d', 'vix_vol_ratio']
    
    train_df = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct'])
    X_train_df = train_df[features_to_use]
    X_predict_df = latest_data[features_to_use]
    current_price = latest_data['close'].values[0]

    for config in ensemble_configs:
        model_name, model_params = config['modelo'], config['best_params']
        forecast_bounds = {}

        for q in quantiles:
            y_train = train_df['target_lower_pct'] if q < 0.5 else train_df['target_upper_pct']
            model = None
            if model_name == 'LightGBM':
                model = lgb.LGBMRegressor(objective='quantile', alpha=q, verbosity=-1, **model_params)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **model_params)
            elif model_name == 'RandomForest':
                 # Para la predicción, entrenamos dos modelos separados para RandomForest
                 y_rf_train = train_df['target_lower_pct'] if q < 0.5 else train_df['target_upper_pct']
                 model = RandomForestRegressor(random_state=42, **model_params).fit(X_train_df, y_rf_train)
            elif model_name == 'GradientBoosting':
                # Para GradientBoosting usamos la pérdida de cuantil
                model = GradientBoostingRegressor(loss='quantile', alpha=q, random_state=42, **model_params)
            elif model_name == 'QuantileRegressor':
                model = QuantileRegressor(quantile=q, solver='highs', **model_params)

            if model_name != 'RandomForest':
                 model.fit(X_train_df, y_train)

            pct_pred = model.predict(X_predict_df)[0]
            forecast_bounds[q] = current_price * (1 + pct_pred)
        
        all_forecasts.append(forecast_bounds)

    final_forecast = {}
    for q in quantiles: final_forecast[q] = np.mean([f[q] for f in all_forecasts])
    return final_forecast, current_price

def graficar_backtest(results_df, plot_data_df, title, filename):
    """Genera y guarda un gráfico del rendimiento del backtest."""
    logging.info(f"Generando gráfico del backtest: {title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 9))
    plt.plot(plot_data_df.index, plot_data_df['close'], label='Precio Real del SPY', color='black', zorder=5)
    plt.plot(results_df.index, results_df['upper_bound'], color='red', linestyle='--', label='Límite Superior Pronosticado')
    plt.plot(results_df.index, results_df['lower_bound'], color='blue', linestyle='--', label='Límite Inferior Pronosticado')
    plt.fill_between(results_df.index, results_df['lower_bound'], results_df['upper_bound'], color='gray', alpha=0.2, label='Rango Pronosticado')
    plt.title(title, fontsize=16)
    plt.ylabel('Precio SPY ($)'), plt.xlabel('Fecha')
    plt.legend(), plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def graficar_pronostico_avanzado(latest_data, forecast_bounds, history_df, quantiles):
    """Genera y guarda el gráfico del pronóstico final con múltiples rangos."""
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


# --- ORQUESTADOR PRINCIPAL ---
if __name__ == '__main__':
    try:
        logging.info("🚀 INICIANDO SCRIPT DE PRONÓSTICO FINAL CON BACKTESTING ROBUSTO")
        
        # Cargar el ensamble campeón desde el último archivo de optimización
        with open('ensemble_champion_v3.json', 'r') as f: PARAMETROS_ENSAMBLE = json.load(f)
        ensemble_models = PARAMETROS_ENSAMBLE['ensemble_models']
        logging.info(f"Ensamble cargado con los modelos: {[m['modelo'] for m in ensemble_models]}")
        
        # Definir los 3 periodos para un backtesting robusto
        FECHA_INICIO_VALIDACION = '2022-01-01'
        FECHA_FIN_VALIDACION = '2023-12-31'
        FECHA_INICIO_PRUEBA_FINAL = '2024-01-01'
        
        RANGO_ACEPTABLE_RUPTURAS = (2.0, 10.0)
        merged_data = actualizar_datos()
        features_data = preparar_datos_y_features(merged_data)

        # ETAPA 1: CALIBRACIÓN DINÁMICA SOBRE EL SET DE VALIDACIÓN
        logging.info(f"--- FASE 1: CALIBRACIÓN en PERIODO DE VALIDACIÓN ({FECHA_INICIO_VALIDACION} a {FECHA_FIN_VALIDACION}) ---")
        parametros_originales = tuple(PARAMETROS_ENSAMBLE['quantiles'])
        df_validacion = ejecutar_backtest_ensamble(features_data, parametros_originales, ensemble_models, FECHA_INICIO_VALIDACION, FECHA_FIN_VALIDACION, "Calibrando en Set de Validación")
        
        breach_pct_val = df_validacion['breach_pct']
        logging.info(f"Tasa de ruptura (Validación): {breach_pct_val:.2f}%")
        
        parametros_calibrados = parametros_originales
        if not (RANGO_ACEPTABLE_RUPTURAS[0] <= breach_pct_val <= RANGO_ACEPTABLE_RUPTURAS[1]):
            logging.warning(f"⚠️ ¡ALERTA DE CALIBRACIÓN! Tasa ({breach_pct_val:.2f}%) fuera del rango.")
            ajuste = (breach_pct_val - np.mean(RANGO_ACEPTABLE_RUPTURAS)) / 100 / 2
            nuevo_lower_q = max(0.005, parametros_originales[0] - ajuste)
            nuevo_upper_q = min(0.995, parametros_originales[1] + ajuste)
            parametros_calibrados = (round(nuevo_lower_q, 3), round(nuevo_upper_q, 3))
            logging.info(f"Ajustando cuantiles dinámicamente a: {parametros_calibrados}")
        else:
            logging.info("✅ Validación exitosa. El modelo está bien calibrado.")

        # ETAPA 2: PRUEBA FINAL SOBRE DATOS "FUERA DE MUESTRA" (Out-of-Sample)
        logging.info(f"--- FASE 2: PRUEBA FINAL en PERIODO 'Out-of-Sample' (Desde {FECHA_INICIO_PRUEBA_FINAL}) ---")
        df_prueba_final = ejecutar_backtest_ensamble(features_data, parametros_calibrados, ensemble_models, FECHA_INICIO_PRUEBA_FINAL, features_data.index[-1], "Ejecutando Prueba Final (OOS)")
        
        logging.info("==================================================")
        logging.info("🏆 RESULTADO FINAL (OUT-OF-SAMPLE) 🏆")
        logging.info(f"Tasa de Ruptura Final: {df_prueba_final['breach_pct']:.2f}%")
        logging.info(f"Error Máximo en Rupturas: {df_prueba_final['max_breach_error_pct']:.2f}%")
        logging.info(f"Error Promedio en Rupturas: {df_prueba_final['avg_breach_error_pct']:.2f}%")
        logging.info("==================================================")
        
        # ETAPA 3: VISUALIZACIÓN DE RESULTADOS
        titulo_val = f"Backtest de Calibración ({FECHA_INICIO_VALIDACION} - {FECHA_FIN_VALIDACION})\nTasa Ruptura: {df_validacion['breach_pct']:.2f}%"
        graficar_backtest(df_validacion['backtest_df'], merged_data.loc[FECHA_INICIO_VALIDACION:FECHA_FIN_VALIDACION], titulo_val, 'backtest_validacion.png')
        
        titulo_oos = f"Backtest Final 'Out-of-Sample' (Desde {FECHA_INICIO_PRUEBA_FINAL})\nTasa Ruptura: {df_prueba_final['breach_pct']:.2f}%"
        graficar_backtest(df_prueba_final['backtest_df'], merged_data.loc[FECHA_INICIO_PRUEBA_FINAL:], titulo_oos, 'backtest_prueba_final_OOS.png')

        # ETAPA 4: PRONÓSTICO FINAL
        logging.info(f"--- FASE 4: GENERANDO PRONÓSTICO FINAL ---")
        QUANTILES_A_PREDECIR = sorted(list(set([parametros_calibrados[0], parametros_calibrados[1], 0.05, 0.95, 0.10, 0.90])))
        latest_data_point = features_data.tail(1)
        forecast, last_price = entrenar_y_predecir_ensamble(features_data, latest_data_point, QUANTILES_A_PREDECIR, ensemble_models)
        graficar_pronostico_avanzado(latest_data_point, forecast, merged_data, QUANTILES_A_PREDECIR)

        logging.info("🎉 PROCESO DE PRONÓSTICO COMPLETADO EXITOSAMENTE 🎉")

    except Exception as e:
        logging.critical(f"❌ El script falló con un error inesperado: {e}", exc_info=True)