# Análisis de Fat Tails en Precios del Oro

Este repositorio contiene herramientas para analizar las características de "fat tails" (colas gruesas) en los retornos del oro, comparándolos con una distribución normal.

## 📊 Características

- **Análisis estadístico completo**: Kurtosis, skewness, tests de normalidad
- **Estimador de Hill**: Cálculo del exponente de cola (α) para cuantificar el grosor de las colas
- **Visualizaciones**: 6 gráficos detallados por cada análisis
- **Soporte para datos diarios y mensuales**
- **Exportación automática**: Carpetas con fecha/hora para cada ejecución

## 📁 Archivos

- `oro_diario.py`: Análisis con precios diarios del oro
- `oro_mensual.py`: Análisis con precios mensuales del oro
- `CMO-Historical-Data-Monthly.csv`: Datos históricos mensuales
- `Gold_Spot_historical_data.csv`: Datos históricos diarios

## 🚀 Uso

### Instalación de dependencias

```bash
pip install pandas numpy matplotlib scipy seaborn openpyxl
```

### Ejecutar análisis de precios diarios

```bash
python oro_diario.py
```

### Ejecutar análisis de precios mensuales

```bash
python oro_mensual.py
```

## 📈 Resultados

Cada ejecución genera una carpeta con:
- `01_histograma_normal.png`: Distribución vs normal
- `02_qq_plot.png`: Q-Q Plot
- `03_colas_escala_log.png`: Análisis de colas en escala logarítmica
- `04_box_plot.png`: Detección de outliers
- `05_serie_temporal.png`: Evolución temporal de retornos
- `06_percentiles.png`: Comparación de percentiles

## 🔬 Metodología

### Kurtosis
- Mide el "grosor" de las colas
- Kurtosis > 3: Fat tails (más extremos que distribución normal)

### Estimador de Hill
- Cuantifica el exponente de cola (α)
- **α < 2**: Varianza infinita (muy peligroso)
- **2 ≤ α < 3**: Colas más gruesas que la normal
- **3 ≤ α < 4**: Kurtosis infinita
- **α ≥ 4**: Comportamiento más cercano a la normal

## 📊 Resultados Principales

### Precios Diarios (6,308 observaciones)
- Kurtosis exceso: 5.16
- α superior: 3.45 (cola gruesa)
- α inferior: 3.36 (cola gruesa)
- Eventos extremos: 4.81x más frecuentes que lo normal

### Precios Mensuales (692 observaciones)
- Kurtosis exceso: 8.56
- α superior: 2.87 (cola muy gruesa)
- α inferior: 2.44 (cola muy gruesa)
- Eventos extremos: 6.42x más frecuentes que lo normal

## 📚 Referencias

- Hill, B. M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution"
- Taleb, N. N. (2007). "The Black Swan"

## 🛠️ Requisitos

- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy
- seaborn

## 📄 Licencia

MIT License

## 👤 Autor

Juan S.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.
