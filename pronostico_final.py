# ==============================================================================
# SCRIPT DEFINITIVO CON REPORTE DE CALIBRACIÓN Y BACKTEST FINAL (VERSIÓN MEJORADA)
# ==============================================================================
# - Carga el modelo y los hiperparámetros ganadores desde 'champion_model.json'.
# - Valida la calibración actual del modelo.
# - Se recalibra automáticamente si el rendimiento se desvía.
# - Genera un backtest gráfico y un pronóstico final con múltiples rangos de probabilidad.
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from datetime import datetime
from tqdm import tqdm
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import json

# --- CONFIGURACIÓN Y PREPARACIÓN DE DATOS ---
warnings.filterwarnings('ignore')
log_filename = f"pronostico_log_{datetime.now().strftime('%Y%m%d')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def actualizar_datos():
    """Descarga los datos más recientes de SPY y VIX desde Yahoo Finance."""
    logging.info("Paso 1: Descargando datos actualizados de SPY y ^VIX...")
    try:
        spy_df = yf.download('SPY', start='2009-01-01', progress=False, auto_adjust=True)
        vix_df = yf.download('^VIX', start='2009-01-01', progress=False, auto_adjust=True)
        merged_df = pd.merge(spy_df[['Open', 'High', 'Low', 'Close']], vix_df[['Close']],
                             left_index=True, right_index=True, suffixes=('_SPY', '_VIX'))
        merged_df.columns = ['open', 'high', 'low', 'close', 'vix_level']
        merged_df['returns'] = np.log(merged_df['close']).diff()
        logging.info(f"Datos cargados exitosamente. Último registro: {merged_df.index[-1].date()}")
        return merged_df.dropna()
    except Exception as e:
        logging.error(f"Error crítico al descargar datos: {e}")
        raise

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

# --- FUNCIONES DE MODELADO, PRONÓSTICO Y GRÁFICOS ---
def entrenar_modelos(features_df, quantiles, model_params):
    """Entrena los modelos LGBM finales con los datos más recientes."""
    train_df = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct'])
    features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
    X_train = train_df[features_to_use]
    models = {}
    logging.info(f"Entrenando modelos finales con cuantiles: {quantiles}")
    for q in quantiles:
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, verbosity=-1, **model_params)
        y_train = train_df['target_lower_pct'] if q < 0.5 else train_df['target_upper_pct']
        model.fit(X_train, y_train)
        models[f'lgbm_{q}'] = model
    return models

def generar_pronostico(models, latest_data, quantiles):
    """Genera el pronóstico de rangos para los próximos 21 días."""
    current_price = latest_data['close'].values[0]
    features_to_use = list(models.values())[0].feature_name_
    X_predict = latest_data[features_to_use]
    forecast_bounds = {}
    for q in quantiles:
        pct_pred = models[f'lgbm_{q}'].predict(X_predict)[0]
        forecast_bounds[q] = current_price * (1 + pct_pred)
    return forecast_bounds, current_price

def graficar_pronostico_avanzado(latest_data, forecast_bounds, history_df, quantiles):
    """Crea y guarda un gráfico avanzado con múltiples rangos de probabilidad."""
    logging.info("Generando gráfico del pronóstico...")
    last_date = latest_data.index[0]
    future_dates = pd.date_range(start=last_date, periods=22, freq='B')[1:]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 9))
    plt.plot(history_df.index[-180:], history_df['close'][-180:], label='Precio Histórico SPY', color='black', linewidth=2)
    
    # Lista de colores expandida para más rangos
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    num_ranges = len(quantiles) // 2
    for i in range(num_ranges):
        lower_q, upper_q = quantiles[i], quantiles[-(i+1)]
        conf_level = (upper_q - lower_q) * 100
        lower_bound, upper_bound = forecast_bounds[lower_q], forecast_bounds[upper_q]
        label_text = f'Rango {conf_level:.0f}% (${lower_bound:.2f} - ${upper_bound:.2f})'
        # Usamos i % len(colors) para evitar errores si hay más rangos que colores
        plt.fill_between(future_dates, lower_bound, upper_bound, color=colors[i % len(colors)], alpha=0.2, label=label_text)
        plt.hlines(y=[lower_bound, upper_bound], xmin=future_dates[0], xmax=future_dates[-1], colors=colors[i % len(colors)], linestyles='--')
        
    plt.title(f'Pronóstico de Rangos de Probabilidad para SPY a 21 Días\nGenerado el {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
    plt.ylabel('Precio SPY ($)'), plt.xlabel('Fecha')
    plt.legend(loc='upper left'), plt.tight_layout()
    plt.savefig('pronostico_avanzado_actual.png', dpi=300)
    plt.show()

# --- FUNCIONES DE BACKTEST Y CALIBRACIÓN ---
def ejecutar_backtest(features_df, quantiles_a_probar, model_params, desc=""):
    """Función central para ejecutar backtests."""
    results_data = []
    backtest_start_date = features_df.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24)
    for lower_q, upper_q in tqdm(quantiles_a_probar, desc=desc):
        breaches = 0
        monthly_predictions = []
        test_dates = features_df.loc[backtest_start_date:].resample('MS').first().index
        for trade_date in test_dates:
            history_df = features_df[features_df.index < trade_date].tail(504)
            evaluation_end = trade_date + pd.DateOffset(days=21)
            test_df = features_df[(features_df.index >= trade_date) & (features_df.index <= evaluation_end)]
            if history_df.empty or test_df.empty: continue
            features_to_use = ['volatility_21d', 'atr_14d', 'vix_level', 'momentum_21d', 'momentum_63d', 'momentum_126d']
            X_train, y_train_lower, y_train_upper = history_df[features_to_use], history_df['target_lower_pct'], history_df['target_upper_pct']
            X_test, current_price = test_df.head(1)[features_to_use], test_df.head(1)['close'].values[0]
            lgb_lower = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, verbosity=-1, **model_params).fit(X_train, y_train_lower)
            lgb_upper = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, verbosity=-1, **model_params).fit(X_train, y_train_upper)
            lower_bound = current_price * (1 + lgb_lower.predict(X_test)[0])
            upper_bound = current_price * (1 + lgb_upper.predict(X_test)[0])
            monthly_predictions.append({'date': trade_date, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
            actual_max_price, actual_min_price = test_df['close'].max(), test_df['close'].min()
            if (actual_max_price > upper_bound) or (actual_min_price < lower_bound):
                breaches += 1
        breach_percentage = (breaches / len(test_dates)) * 100
        results_data.append({'quantiles': (lower_q, upper_q), 'breach_pct': breach_percentage, 'backtest_df': pd.DataFrame(monthly_predictions).set_index('date')})
    return pd.DataFrame(results_data)

def graficar_backtest(results_df, backtest_start_date, merged_df, title):
    """Crea y guarda el gráfico del backtest final."""
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
        logging.info("🚀 INICIANDO SCRIPT DE PRONÓSTICO CON AUTO-OPTIMIZACIÓN")
        logging.info("==================================================")

        # --- CARGAR PARÁMETROS DEL CAMPEÓN DESDE EL ARCHIVO JSON ---
        try:
            with open('champion_model.json', 'r') as f:
                PARAMETROS_CAMPEONES = json.load(f)
            logging.info(f"Parámetros del campeón '{PARAMETROS_CAMPEONES['model']}' cargados desde archivo.")
        except FileNotFoundError:
            logging.error("❌ No se encontró el archivo 'champion_model.json'. Ejecuta primero el script de optimización.")
            raise
        
        RANGO_ACEPTABLE_RUPTURAS = (2.0, 8.0)
        
        merged_data = actualizar_datos()
        features_data = preparar_datos_y_features(merged_data)
        
        # --- ETAPA 1: VALIDACIÓN ---
        logging.info(f"--- FASE 1: VALIDANDO PARÁMETROS ACTUALES: {PARAMETROS_CAMPEONES['params']} ---")
        df_validacion = ejecutar_backtest(features_data, [tuple(PARAMETROS_CAMPEONES['quantiles'])], PARAMETROS_CAMPEONES['params'], desc="Validando Parámetros Actuales")
        breach_pct_actual = df_validacion['breach_pct'].iloc[0]
        logging.info(f"Tasa de ruptura (Breach Rate) actual: {breach_pct_actual:.2f}%")
        
        # --- ETAPA 2: AUTO-CALIBRACIÓN ---
        parametros_finales = tuple(PARAMETROS_CAMPEONES['quantiles'])
        if not (RANGO_ACEPTABLE_RUPTURAS[0] <= breach_pct_actual <= RANGO_ACEPTABLE_RUPTURAS[1]):
            logging.warning(f"⚠️ ¡Alerta de Calibración! Iniciando recalibración automática...")
            QUANTILES_DE_CALIBRACION = [(0.05, 0.95), (0.025, 0.975), (0.015, 0.985), (0.01, 0.99)]
            resultados_calibracion = ejecutar_backtest(features_data, QUANTILES_DE_CALIBRACION, PARAMETROS_CAMPEONES['params'], desc="Calibrando nuevos cuantiles")
            print("\n--- Resultados de la Calibración ---")
            print(resultados_calibracion[['quantiles', 'breach_pct']].to_string())
            target_breach = 5.0
            optimal_row = resultados_calibracion.iloc[(resultados_calibracion['breach_pct'] - target_breach).abs().idxmin()]
            parametros_finales = optimal_row['quantiles']
            logging.info(f"✅ Nuevos cuantiles óptimos seleccionados: {parametros_finales}")
        else:
            logging.info("✅ Validación exitosa. El modelo está bien calibrado.")

        # --- ETAPA 3: BACKTEST FINAL Y GRÁFICO ---
        logging.info(f"--- FASE 3: BACKTEST FINAL CON CUANTILES {parametros_finales} ---")
        df_backtest_final = ejecutar_backtest(features_data, [parametros_finales], PARAMETROS_CAMPEONES['params'], desc="Ejecutando Backtest Final").iloc[0]
        graficar_backtest(df_backtest_final['backtest_df'],
                          features_data.dropna(subset=['target_lower_pct', 'target_upper_pct']).index[-1] - pd.DateOffset(months=24),
                          merged_data,
                          f"Backtest Final (24 Meses) con Cuantiles {parametros_finales}\nTasa de Ruptura (Breach Rate): {df_backtest_final['breach_pct']:.2f}%")

        # --- ETAPA 4: PRONÓSTICO FINAL ---
        logging.info(f"--- FASE 4: GENERANDO PRONÓSTICO FINAL CON CUANTILES {parametros_finales} ---")
        
        # Lista de cuantiles actualizada para más rangos en el gráfico
        QUANTILES_A_PREDECIR = sorted(list(set([
            # Rango 97% (viene de los parámetros finales)
            parametros_finales[0], parametros_finales[1],
            # Rango 90%
            0.05, 0.95,
            # Rango 85%
            0.075, 0.925,
            # Rango 80%
            0.10, 0.90,
            # Rango 75%
            0.125, 0.875,
            # Rango 70%
            0.15, 0.85
        ])))
        
        trained_models = entrenar_modelos(features_data.tail(504 + 21), QUANTILES_A_PREDECIR, PARAMETROS_CAMPEONES['params'])
        latest_data_point = features_data.tail(1)
        forecast, last_price = generar_pronostico(trained_models, latest_data_point, QUANTILES_A_PREDECIR)
        graficar_pronostico_avanzado(latest_data_point, forecast, merged_data, QUANTILES_A_PREDECIR)
        
        logging.info("==================================================")
        logging.info("🎉 PROCESO DE PRONÓSTICO COMPLETADO EXITOSAMENTE 🎉")
        logging.info("==================================================")
        
    except Exception as e:
        logging.critical(f"❌ El script falló con un error inesperado: {e}", exc_info=True)