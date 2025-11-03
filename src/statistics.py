import numpy as np
from scipy.stats import chi2

def calcular_residuos(data, coord):
    """Calcula residuos y suma de residuos al cuadrado para una coordenada"""
    media = data[coord].mean()
    data[f"residuo_{coord}"] = media - data[coord]
    data[f"R_{coord[0].upper()}2"] = data[f"residuo_{coord}"] ** 2
    return data[f"R_{coord[0].upper()}2"].sum()

def calcular_std_por_dimension(sum_r2, gl):
    """Calcula la desviación estándar para una dimensión"""
    return np.sqrt(sum_r2 / gl)

def calcular_umbral_iso(sigma_fabrica, gl, confianza=0.95):
    """Calcula el umbral ISO según chi-cuadrado"""
    chi_val = chi2.ppf(confianza, gl)
    return sigma_fabrica * np.sqrt(chi_val / gl)

def calcular_estadisticas(df1, df2, sigma_fab_xy, sigma_fab_z, nro_rovers):
    """Calcula todas las estadísticas necesarias para la evaluación ISO"""
    resultados = {}
    if 'code' in df1.columns:
        col_id='code'
    elif 'description' in df1.columns:
        col_id='description'
    else:
        raise ValueError("Contactar con I+D (J.Aguero) por error no identificado en el DataFrame")
    serie_id = df1[col_id].dropna().astype(str).str.strip()
    if serie_id.empty:
        raise ValueError(
            f"La columna '{col_id}' no contiene datos válidos.")
    n_series = df1['code'].nunique()
    n_obs_por_serie = df1['code'].value_counts().iloc[0]
    resultados['gl'] = (n_series * n_obs_por_serie - 1) * nro_rovers

    # Cálculo de residuos
    sum_RE2 = calcular_residuos(df1, 'easting') + calcular_residuos(df2, 'easting')
    sum_RN2 = calcular_residuos(df1, 'northing') + calcular_residuos(df2, 'northing')
    sum_Rh2 = calcular_residuos(df1, 'altitude') + calcular_residuos(df2, 'altitude')

    # Desviaciones estándar
    resultados['sx'] = calcular_std_por_dimension(sum_RE2, resultados['gl'])
    resultados['sy'] = calcular_std_por_dimension(sum_RN2, resultados['gl'])
    resultados['sh'] = calcular_std_por_dimension(sum_Rh2, resultados['gl'])

    # Aplicar corrección
    resultados['s_xy'] = np.sqrt(resultados['sx']**2 + resultados['sy']**2) + 0.003
    resultados['sh_corregido'] = resultados['sh'] + 0.003

    # Umbrales ISO
    resultados['umbral_xy'] = calcular_umbral_iso(sigma_fab_xy, resultados['gl'])
    resultados['umbral_z'] = calcular_umbral_iso(sigma_fab_z, resultados['gl'])

    # Evaluación
    resultados['aprobado'] = (resultados['s_xy'] <= resultados['umbral_xy'] and 
                             resultados['sh_corregido'] <= resultados['umbral_z'])
    
    return resultados