#!/bin/bash
# Este script ejecuta el ciclo completo del sistema de pronóstico.

echo "🚀 Iniciando el ciclo completo de optimización y pronóstico..."

# Paso 1: Ejecutar el script de optimización pesada.
# Esto generará un nuevo archivo champion_model.json.
echo "--- (1/2) Ejecutando optimización de modelos (Horse Race)... ---"
python grid_search_horse_race.py

# Verificar que el primer script terminó con éxito antes de continuar.
if [ $? -ne 0 ]; then
    echo "❌ El script de optimización falló. Abortando."
    exit 1
fi

# Paso 2: Ejecutar el script de pronóstico diario.
# Esto usará el champion_model.json recién creado.
echo "--- (2/2) Ejecutando el pronóstico final diario... ---"
python pronostico_final.py

echo "✅ ¡Ciclo completado exitosamente!"