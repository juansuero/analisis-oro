"""
Análisis de Fat Tails en los Precios del Oro Con Precios Diarios
Este programa analiza qué tan "fat tails" (colas gruesas) tiene la distribución
de los retornos logarítmicos del oro comparándola con una distribución normal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
from datetime import datetime

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_datos(archivo):
    """
    Carga los datos del archivo (CSV o Excel)
    """
    try:
        # Intentar leer el archivo
        print(f"\n📂 Intentando cargar: {archivo}")
        
        # Detectar el tipo de archivo por extensión
        if archivo.lower().endswith('.csv'):
            df = pd.read_csv(archivo)
            print(f"✓ Archivo CSV cargado")
        elif archivo.lower().endswith(('.xlsx', '.xls')):
            # Primero, obtener información sobre las hojas disponibles
            excel_file = pd.ExcelFile(archivo)
            print(f"✓ Archivo Excel encontrado")
            print(f"✓ Hojas disponibles: {excel_file.sheet_names}")
            
            # Leer la primera hoja (o especificar cual usar)
            if len(excel_file.sheet_names) > 1:
                print(f"\n⚠️  El archivo tiene múltiples hojas. Usando la primera: '{excel_file.sheet_names[0]}'")
            
            df = pd.read_excel(archivo, sheet_name=0)
        else:
            print(f"\n❌ Error: Formato de archivo no soportado. Use .csv, .xlsx o .xls")
            return None
        
        print(f"✓ Datos cargados exitosamente: {len(df)} filas")
        print(f"✓ Columnas disponibles: {list(df.columns)}")
        
        # Mostrar primeras filas para diagnóstico
        if len(df) > 0:
            print(f"\n📋 Primeras 3 filas del archivo:")
            print(df.head(3))
        else:
            print("\n⚠️  ADVERTENCIA: El archivo no contiene datos (0 filas)")
        
        return df
    except FileNotFoundError:
        print(f"\n❌ Error: No se encuentra el archivo '{archivo}'")
        print("Asegúrate de que el archivo existe y la ruta es correcta")
        return None
    except Exception as e:
        print(f"\n❌ Error al cargar el archivo: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        return None

def analizar_fat_tails(retornos):
    """
    Analiza las características de fat tails de los retornos
    """
    # Eliminar valores NaN
    retornos_limpios = retornos.dropna()
    
    # Estadísticas básicas
    media = retornos_limpios.mean()
    desviacion = retornos_limpios.std()
    
    # Kurtosis - Medida clave de fat tails
    # Kurtosis > 3 indica fat tails (más que una distribución normal)
    kurtosis_exceso = stats.kurtosis(retornos_limpios, fisher=True)  # Exceso sobre 3
    kurtosis_total = stats.kurtosis(retornos_limpios, fisher=False)  # Total
    
    # Skewness - Asimetría
    asimetria = stats.skew(retornos_limpios)
    
    # Test de normalidad Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(retornos_limpios)
    
    # Test de normalidad Shapiro-Wilk (más potente para muestras pequeñas)
    if len(retornos_limpios) <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(retornos_limpios)
    else:
        sw_stat, sw_pvalue = None, None
    
    # Calcular percentiles extremos
    percentil_1 = np.percentile(retornos_limpios, 1)
    percentil_99 = np.percentile(retornos_limpios, 99)
    percentil_5 = np.percentile(retornos_limpios, 5)
    percentil_95 = np.percentile(retornos_limpios, 95)
    
    # Contar observaciones extremas (más allá de 3 desviaciones estándar)
    limite_superior = media + 3 * desviacion
    limite_inferior = media - 3 * desviacion
    extremos = np.sum((retornos_limpios > limite_superior) | (retornos_limpios < limite_inferior))
    porcentaje_extremos = (extremos / len(retornos_limpios)) * 100
    
    # En una distribución normal, esperamos ~0.27% más allá de 3 sigma
    extremos_esperados_normal = len(retornos_limpios) * 0.0027
    
    resultados = {
        'n_observaciones': len(retornos_limpios),
        'media': media,
        'desviacion': desviacion,
        'kurtosis_exceso': kurtosis_exceso,
        'kurtosis_total': kurtosis_total,
        'asimetria': asimetria,
        'jb_statistic': jb_stat,
        'jb_pvalue': jb_pvalue,
        'sw_statistic': sw_stat,
        'sw_pvalue': sw_pvalue,
        'percentil_1': percentil_1,
        'percentil_99': percentil_99,
        'percentil_5': percentil_5,
        'percentil_95': percentil_95,
        'extremos_observados': extremos,
        'porcentaje_extremos': porcentaje_extremos,
        'extremos_esperados_normal': extremos_esperados_normal
    }
    
    return resultados, retornos_limpios

def calcular_hill_estimator(datos, k=None):
    """
    Calcula el estimador de Hill para el exponente de cola.
    
    Parámetros:
    - datos: array de valores (debe ser para una cola, valores extremos)
    - k: número de observaciones extremas a usar (si None, usa regla empírica)
    
    Retorna:
    - alpha: exponente de cola (índice de cola)
    - k_usado: número de observaciones usadas
    
    Interpretación:
    - α < 2: Varianza infinita (colas muy gruesas, distribución no tiene varianza)
    - α < 3: Colas más gruesas que la normal
    - α < 4: Kurtosis infinita
    - α más pequeño = colas más gruesas
    """
    n = len(datos)
    
    # Ordenar datos de mayor a menor (para cola superior)
    datos_ordenados = np.sort(datos)[::-1]
    
    # Si no se especifica k, usar regla empírica: k ≈ sqrt(n) o n^0.5
    if k is None:
        k = max(int(np.sqrt(n)), 10)  # Mínimo 10 observaciones
        k = min(k, n // 2)  # Máximo la mitad de los datos
    
    # Asegurar que k es válido
    k = max(1, min(k, n - 1))
    
    # Estimador de Hill
    # α_hat = [1/k * Σ(log(X_i) - log(X_(k+1)))]^(-1)
    X_k_plus_1 = datos_ordenados[k]
    
    if X_k_plus_1 <= 0:
        return np.nan, k
    
    suma_logs = np.sum(np.log(datos_ordenados[:k]) - np.log(X_k_plus_1))
    alpha = k / suma_logs if suma_logs > 0 else np.nan
    
    return alpha, k

def analizar_exponentes_cola(retornos_limpios):
    """
    Analiza los exponentes de cola usando el estimador de Hill
    para las colas superior e inferior.
    """
    print("\n" + "="*70)
    print("ANÁLISIS DE EXPONENTES DE COLA (ESTIMADOR DE HILL)")
    print("="*70)
    
    # Separar retornos positivos (cola superior) y negativos (cola inferior)
    retornos_positivos = retornos_limpios[retornos_limpios > 0].values
    retornos_negativos = np.abs(retornos_limpios[retornos_limpios < 0].values)
    
    # Calcular diferentes valores de k para ver la estabilidad
    n_pos = len(retornos_positivos)
    n_neg = len(retornos_negativos)
    
    # Usar varios valores de k para verificar robustez
    k_values = [
        int(np.sqrt(n_pos)),
        int(n_pos * 0.05),  # 5% superior
        int(n_pos * 0.10),  # 10% superior
    ]
    
    print(f"\n📊 COLA SUPERIOR (retornos positivos):")
    print(f"   Total de observaciones: {n_pos}")
    
    alphas_superior = []
    for k in k_values:
        if k > 0 and k < n_pos:
            alpha, k_usado = calcular_hill_estimator(retornos_positivos, k)
            if not np.isnan(alpha):
                alphas_superior.append(alpha)
                print(f"   k = {k_usado:4d} ({100*k_usado/n_pos:5.1f}%) → α = {alpha:.4f}")
    
    if alphas_superior:
        alpha_superior_promedio = np.mean(alphas_superior)
        print(f"\n   ✓ Exponente promedio (α superior): {alpha_superior_promedio:.4f}")
        
        # Interpretación
        if alpha_superior_promedio < 2:
            print(f"   ⚠️⚠️⚠️  COLA EXTREMADAMENTE GRUESA (α < 2): Varianza infinita!")
        elif alpha_superior_promedio < 3:
            print(f"   ⚠️⚠️  COLA MUY GRUESA (2 ≤ α < 3): Más gruesa que distribución normal")
        elif alpha_superior_promedio < 4:
            print(f"   ⚠️  COLA GRUESA (3 ≤ α < 4): Kurtosis infinita")
        else:
            print(f"   ✓ Cola moderada (α ≥ 4)")
    
    print(f"\n📊 COLA INFERIOR (retornos negativos, en valor absoluto):")
    print(f"   Total de observaciones: {n_neg}")
    
    k_values_neg = [
        int(np.sqrt(n_neg)),
        int(n_neg * 0.05),
        int(n_neg * 0.10),
    ]
    
    alphas_inferior = []
    for k in k_values_neg:
        if k > 0 and k < n_neg:
            alpha, k_usado = calcular_hill_estimator(retornos_negativos, k)
            if not np.isnan(alpha):
                alphas_inferior.append(alpha)
                print(f"   k = {k_usado:4d} ({100*k_usado/n_neg:5.1f}%) → α = {alpha:.4f}")
    
    if alphas_inferior:
        alpha_inferior_promedio = np.mean(alphas_inferior)
        print(f"\n   ✓ Exponente promedio (α inferior): {alpha_inferior_promedio:.4f}")
        
        # Interpretación
        if alpha_inferior_promedio < 2:
            print(f"   ⚠️⚠️⚠️  COLA EXTREMADAMENTE GRUESA (α < 2): Varianza infinita!")
        elif alpha_inferior_promedio < 3:
            print(f"   ⚠️⚠️  COLA MUY GRUESA (2 ≤ α < 3): Más gruesa que distribución normal")
        elif alpha_inferior_promedio < 4:
            print(f"   ⚠️  COLA GRUESA (3 ≤ α < 4): Kurtosis infinita")
        else:
            print(f"   ✓ Cola moderada (α ≥ 4)")
    
    # Comparación
    if alphas_superior and alphas_inferior:
        print(f"\n📈 COMPARACIÓN DE COLAS:")
        print(f"   α superior: {alpha_superior_promedio:.4f}")
        print(f"   α inferior: {alpha_inferior_promedio:.4f}")
        
        diferencia = abs(alpha_superior_promedio - alpha_inferior_promedio)
        if diferencia < 0.5:
            print(f"   → Colas aproximadamente simétricas")
        elif alpha_superior_promedio < alpha_inferior_promedio:
            print(f"   → Cola superior más gruesa (más riesgo de subidas extremas)")
        else:
            print(f"   → Cola inferior más gruesa (más riesgo de caídas extremas)")
    
    print("\n" + "="*70)
    print("INTERPRETACIÓN DEL EXPONENTE α:")
    print("  • α < 2  : Varianza infinita (muy peligroso)")
    print("  • 2≤α<3  : Colas más gruesas que la normal")
    print("  • 3≤α<4  : Kurtosis infinita")
    print("  • α ≥ 4  : Comportamiento más cercano a la normal")
    print("  • Menor α = Mayor riesgo de eventos extremos")
    print("="*70)
    
    resultados_hill = {
        'alpha_superior': alpha_superior_promedio if alphas_superior else np.nan,
        'alpha_inferior': alpha_inferior_promedio if alphas_inferior else np.nan,
        'alphas_superior_lista': alphas_superior,
        'alphas_inferior_lista': alphas_inferior
    }
    
    return resultados_hill

def imprimir_resultados(resultados):
    """
    Imprime un reporte detallado de los resultados
    """
    print("\n" + "="*70)
    print("ANÁLISIS DE FAT TAILS - RETORNOS DEL ORO (PRECIOS DIARIOS)")
    print("="*70)
    
    print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   Número de observaciones: {resultados['n_observaciones']}")
    print(f"   Media: {resultados['media']:.6f}")
    print(f"   Desviación estándar: {resultados['desviacion']:.6f}")
    
    print(f"\n🎯 MEDIDAS DE FAT TAILS:")
    print(f"   Kurtosis (exceso): {resultados['kurtosis_exceso']:.4f}")
    print(f"   Kurtosis (total): {resultados['kurtosis_total']:.4f}")
    print(f"   → Distribución normal tiene kurtosis = 3.0 (exceso = 0)")
    
    if resultados['kurtosis_exceso'] > 0:
        print(f"   ⚠️  CONCLUSIÓN: Fat tails detectadas (kurtosis exceso > 0)")
        if resultados['kurtosis_exceso'] > 3:
            print(f"   ⚠️⚠️  Fat tails SEVERAS (kurtosis exceso > 3)")
    else:
        print(f"   ✓ No hay evidencia de fat tails (kurtosis exceso ≤ 0)")
    
    print(f"\n📐 ASIMETRÍA:")
    print(f"   Skewness: {resultados['asimetria']:.4f}")
    if abs(resultados['asimetria']) < 0.5:
        print(f"   → Distribución aproximadamente simétrica")
    elif resultados['asimetria'] > 0:
        print(f"   → Distribución sesgada a la derecha (cola derecha más larga)")
    else:
        print(f"   → Distribución sesgada a la izquierda (cola izquierda más larga)")
    
    print(f"\n🧪 TESTS DE NORMALIDAD:")
    print(f"   Jarque-Bera statistic: {resultados['jb_statistic']:.4f}")
    print(f"   Jarque-Bera p-value: {resultados['jb_pvalue']:.6f}")
    if resultados['jb_pvalue'] < 0.05:
        print(f"   ✗ Se rechaza normalidad (p < 0.05)")
    else:
        print(f"   ✓ No se rechaza normalidad (p ≥ 0.05)")
    
    if resultados['sw_pvalue'] is not None:
        print(f"\n   Shapiro-Wilk statistic: {resultados['sw_statistic']:.4f}")
        print(f"   Shapiro-Wilk p-value: {resultados['sw_pvalue']:.6f}")
        if resultados['sw_pvalue'] < 0.05:
            print(f"   ✗ Se rechaza normalidad (p < 0.05)")
        else:
            print(f"   ✓ No se rechaza normalidad (p ≥ 0.05)")
    
    print(f"\n📍 PERCENTILES:")
    print(f"   1%: {resultados['percentil_1']:.6f}")
    print(f"   5%: {resultados['percentil_5']:.6f}")
    print(f"   95%: {resultados['percentil_95']:.6f}")
    print(f"   99%: {resultados['percentil_99']:.6f}")
    
    print(f"\n⚡ EVENTOS EXTREMOS (más allá de ±3σ):")
    print(f"   Observados: {resultados['extremos_observados']} ({resultados['porcentaje_extremos']:.2f}%)")
    print(f"   Esperados (dist. normal): {resultados['extremos_esperados_normal']:.1f} (0.27%)")
    ratio = resultados['extremos_observados'] / max(resultados['extremos_esperados_normal'], 1)
    print(f"   Ratio observados/esperados: {ratio:.2f}x")
    
    if ratio > 2:
        print(f"   ⚠️  Eventos extremos mucho más frecuentes que en dist. normal")
    
    print("\n" + "="*70)

def crear_visualizaciones(retornos_limpios, resultados):
    """
    Crea visualizaciones para analizar fat tails.
    Guarda cada gráfico en un archivo PNG separado dentro de una carpeta con la fecha.
    """
    # Crear carpeta con fecha y hora actual
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta_salida = f"analisis_oro_preciosdiarios{fecha_hora}"
    
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"\n📁 Carpeta creada: {carpeta_salida}")
    
    mu, sigma = resultados['media'], resultados['desviacion']
    x = np.linspace(retornos_limpios.min(), retornos_limpios.max(), 100)
    
    # 1. Histograma con curva normal superpuesta
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax1.hist(retornos_limpios, bins=50, density=True, 
                                  alpha=0.7, color='skyblue', edgecolor='black')
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label='Distribución Normal')
    ax1.set_xlabel('Log Returns', fontsize=12)
    ax1.set_ylabel('Densidad', fontsize=12)
    ax1.set_title('Histograma vs Distribución Normal', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo1 = os.path.join(carpeta_salida, '01_histograma_normal.png')
    plt.savefig(archivo1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo1}")
    
    # 2. Q-Q Plot (Quantile-Quantile)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    stats.probplot(retornos_limpios, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo2 = os.path.join(carpeta_salida, '02_qq_plot.png')
    plt.savefig(archivo2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo2}")
    
    # 3. Distribución de las colas (escala logarítmica)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(retornos_limpios, bins=50, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label='Normal')
    ax3.set_yscale('log')
    ax3.set_xlabel('Log Returns', fontsize=12)
    ax3.set_ylabel('Densidad (escala log)', fontsize=12)
    ax3.set_title('Colas en Escala Logarítmica', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo3 = os.path.join(carpeta_salida, '03_colas_escala_log.png')
    plt.savefig(archivo3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo3}")
    
    # 4. Box Plot
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.boxplot(retornos_limpios, vert=True)
    ax4.set_ylabel('Log Returns', fontsize=12)
    ax4.set_title('Box Plot - Valores Extremos', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo4 = os.path.join(carpeta_salida, '04_box_plot.png')
    plt.savefig(archivo4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo4}")
    
    # 5. Serie temporal de retornos
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(retornos_limpios.values, linewidth=0.5, alpha=0.7, color='steelblue')
    ax5.axhline(y=mu, color='r', linestyle='--', label='Media', linewidth=2)
    ax5.axhline(y=mu + 3*sigma, color='orange', linestyle='--', label='±3σ', linewidth=2)
    ax5.axhline(y=mu - 3*sigma, color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('Observación', fontsize=12)
    ax5.set_ylabel('Log Returns', fontsize=12)
    ax5.set_title('Serie Temporal de Retornos', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo5 = os.path.join(carpeta_salida, '05_serie_temporal.png')
    plt.savefig(archivo5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo5}")
    
    # 6. Comparación de percentiles
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles_observados = [np.percentile(retornos_limpios, p) for p in percentiles]
    percentiles_normal = [stats.norm.ppf(p/100, mu, sigma) for p in percentiles]
    
    ax6.plot(percentiles, percentiles_observados, 'o-', label='Observado', 
             linewidth=2, markersize=8, color='steelblue')
    ax6.plot(percentiles, percentiles_normal, 's--', label='Normal teórico', 
             linewidth=2, markersize=8, color='red', alpha=0.7)
    ax6.set_xlabel('Percentil', fontsize=12)
    ax6.set_ylabel('Valor', fontsize=12)
    ax6.set_title('Comparación de Percentiles', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    archivo6 = os.path.join(carpeta_salida, '06_percentiles.png')
    plt.savefig(archivo6, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {archivo6}")
    
    print(f"\n✅ Todos los gráficos guardados en la carpeta: {carpeta_salida}")
    print(f"   Total de archivos: 6 imágenes PNG")

def main():
    """
    Función principal
    """
    # Solicitar nombre del archivo
    print("="*70)
    print("ANÁLISIS DE FAT TAILS EN DEL ORO (PRECIOS DIARIOS)")
    print("="*70)

    # Cambiar a archivo CSV
    archivo = "Gold_Spot_historical_data.csv"

    # Cargar datos
    df = cargar_datos(archivo)
    if df is None:
        return
    
    # Verificar que el DataFrame no esté vacío
    if len(df) == 0:
        print("\n❌ Error: El archivo no contiene datos")
        return

    # Mostrar información sobre las columnas
    print(f"\n📊 Información del DataFrame:")
    print(f"   Forma: {df.shape} (filas, columnas)")
    print(f"   Columnas: {list(df.columns)}")
    
    # Intentar identificar la columna de log returns o returns
    # Buscar variaciones comunes del nombre
    posibles_nombres_returns = [
        'returns', 'Returns', 'RETURNS',
        'log returns', 'Log Returns', 'LOG RETURNS', 
        'log_returns', 'LogReturns', 'log return',
        'Log Return', 'return', 'Return'
    ]
    
    columna_returns = None
    for nombre in posibles_nombres_returns:
        if nombre in df.columns:
            columna_returns = nombre
            print(f"\n✓ Encontrada columna de retornos: '{columna_returns}'")
            break
    
    if columna_returns is None:
        print("\n❌ No se encuentra una columna de returns")
        print("Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        # Si no hay columna de returns pero hay precio, calcularla
        posibles_nombres_precio = ['price', 'Price', 'PRICE', 'gold price', 
                                   'Gold Price', 'Close', 'close']
        columna_precio = None
        for nombre in posibles_nombres_precio:
            if nombre in df.columns:
                columna_precio = nombre
                break
        
        if columna_precio is not None:
            print(f"\n✓ Encontrada columna de precio: '{columna_precio}'")
            print("📊 Calculando log returns...")
            df['log_returns'] = np.log(df[columna_precio] / df[columna_precio].shift(1))
            columna_returns = 'log_returns'
            print(f"✓ Log returns calculados: {df[columna_returns].notna().sum()} valores válidos")
        else:
            print("\n❌ Tampoco se encuentra una columna de precios para calcular returns")
            print("\nPor favor, asegúrate de que el archivo tenga una columna con:")
            print("  - Returns (con nombres como 'returns', 'Returns', etc.)")
            print("  - O una columna de precios (con nombres como 'price', 'Price', etc.)")
            return
    
    # Convertir la columna de returns a numérico (por si tiene formato de porcentaje como texto)
    # Primero, eliminar el símbolo % si existe y convertir
    if df[columna_returns].dtype == 'object':
        print(f"\n📝 Convirtiendo columna '{columna_returns}' de texto a numérico...")
        # Eliminar % y convertir a decimal
        df[columna_returns] = df[columna_returns].str.rstrip('%').astype('float') / 100.0
        print(f"✓ Conversión completada")
    
    # Analizar fat tails
    print(f"\n🔍 Analizando fat tails usando la columna: '{columna_returns}'")
    resultados, retornos_limpios = analizar_fat_tails(df[columna_returns])
    
    # Imprimir resultados
    imprimir_resultados(resultados)
    
    # Calcular exponentes de cola (Estimador de Hill)
    print("\n🔬 Calculando exponentes de cola...")
    resultados_hill = analizar_exponentes_cola(retornos_limpios)
    
    # Crear visualizaciones
    print("\n📈 Generando visualizaciones...")
    crear_visualizaciones(retornos_limpios, resultados)
    
    print("\n✓ Análisis completado exitosamente!")

if __name__ == "__main__":
    main()
