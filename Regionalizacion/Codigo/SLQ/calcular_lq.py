"""
Cálculo de Cocientes de Localización para Jalisco
Usando PIBE 2023 (a precios constantes de 2018)

Secuencia:
  1. Extraer VAB 2023 de Jalisco y Nacional del PIBE
  2. Construir el vector de 34 sectores alineado con la MIP agregada
  3. Calcular SLQ (especialización sectorial de Jalisco vs. Nacional)
  4. Calcular CILQ (relación proveedor-comprador)
  5. Calcular FLQ con δ provisional = 0.30
  6. Guardar resultados

Fuentes:
  - PIBE entidad Jalisco 2024 (datos hasta 2023<R>)
  - PIBE entidad Nacional 2024 (datos hasta 2023<R>)
  - MIP coeficientes técnicos agregada 34x34 (output del paso anterior)

Referencia metodológica:
  Flegg, A.T. & Tohmo, T. (2014). Regional Studies, 50(2), 310-325.
  Miller, R. & Blair, P. (2009). Input-Output Analysis. Cambridge.
"""

import pandas as pd
import numpy as np
import re
import os

# ── Rutas ──────────────────────────────────────────────────────────────────────
DIR_INPUTS  = "/mnt/user-data/uploads"
DIR_OUTPUTS = "/mnt/user-data/outputs"

PIBE_JAL = os.path.join(DIR_INPUTS, "conjunto_de_datos_pibe_entidad_jal2024_p.csv")
PIBE_NAC = os.path.join(DIR_INPUTS, "conjunto_de_datos_pibe_entidad_nac2024_p.csv")
MIP_CTEC = os.path.join(DIR_OUTPUTS, "mip_ctec_agregada_36x36.csv")

SALIDA_SLQ    = os.path.join(DIR_OUTPUTS, "lq_slq_sectorial.csv")
SALIDA_FLQ    = os.path.join(DIR_OUTPUTS, "lq_flq_matriz.csv")
SALIDA_MIP_R  = os.path.join(DIR_OUTPUTS, "mip_regionalizada_jalisco.csv")

# Parámetro delta provisional (se optimizará con ML en el siguiente paso)
DELTA = 0.30

# ── Mapeo PIBE → grupos MIP ────────────────────────────────────────────────────
# Indica qué código del PIBE corresponde a cada grupo de nuestra MIP agregada
# y en qué nivel del PIBE se encuentra (4 o 5)
# Formato: codigo_mip_grupo : (codigo_pibe, nivel_pibe)

MAPEO_PIBE_A_MIP = {
    # Primarias — nivel 5 del PIBE
    "111":     ("111",     5),
    "112":     ("112",     5),
    "113-115": ("113,115", 5),
    "114":     ("114",     5),
    # Minería — nivel 5
    "21-1":    ("21-1",    5),
    "21-2":    ("21-2",    5),
    # Energía y construcción — nivel 4
    "22":      ("22",      4),
    "23":      ("23",      4),
    # Manufacturas — nivel 5
    "311":     ("311",     5),
    "312":     ("312",     5),
    "313-314": ("313-314", 5),
    "315-316": ("315-316", 5),
    "321":     ("321",     5),
    "322-323": ("322-323", 5),
    "324-326": ("324-326", 5),
    "327":     ("327",     5),
    "331-332": ("331-332", 5),
    "333-336": ("333-336", 5),
    "337":     ("337",     5),
    "339":     ("339",     5),
    # Comercio — nivel 4
    "43":      ("43",      4),
    "46":      ("46",      4),
    # Servicios — nivel 4
    "48-49":   ("48-49",   4),
    "51":      ("51",      4),
    "52":      ("52",      4),
    "53":      ("53",      4),
    "54":      ("54",      4),
    "55":      ("55",      4),
    "56":      ("56",      4),
    "61":      ("61",      4),
    "62":      ("62",      4),
    "71":      ("71",      4),
    "72":      ("72",      4),
    "81":      ("81",      4),
}


# ── Funciones ──────────────────────────────────────────────────────────────────

def extraer_vab_pibe(ruta_csv):
    """
    Lee un archivo PIBE y extrae el VAB a precios constantes 2018 para el año 2023.
    Devuelve un DataFrame con columnas: codigo_pibe, n_niveles, valor_2023
    """
    df = pd.read_csv(ruta_csv, encoding='utf-8-sig')

    # Filtrar solo filas de VAB a precios constantes 2018
    mask = df['Descriptores'].str.startswith('Millones de pesos a precios de 2018|B.1bV')
    vab = df[mask].copy()

    # Contar niveles jerárquicos (número de "|" en el descriptor)
    vab['n_niveles'] = vab['Descriptores'].apply(lambda x: x.count('|'))

    # Extraer el código del último nivel del descriptor
    def extraer_codigo(desc):
        desc_limpio = re.sub(r'<[^>]+>', '', desc)   # quitar <R>, <P>, <C1>
        ultimo = desc_limpio.split('|')[-1].strip()
        m = re.match(r'([\d,\-]+)---(.*)', ultimo)
        return m.group(1).strip() if m else None

    vab['codigo_pibe'] = vab['Descriptores'].apply(extraer_codigo)

    # Identificar la columna 2023 (puede llamarse "2023<R>" o "2023<P>")
    col_2023 = [c for c in df.columns if '2023' in c][0]
    vab['valor_2023'] = pd.to_numeric(vab[col_2023], errors='coerce')

    return vab[['codigo_pibe', 'n_niveles', 'valor_2023']].dropna(subset=['codigo_pibe'])


def construir_vector_pibe(vab_df, mapeo):
    """
    A partir del DataFrame del PIBE y el mapeo, construye un Series
    indexado por los 34 grupos MIP con el valor de VAB 2023.
    """
    resultado = {}
    for grupo_mip, (codigo_pibe, nivel) in mapeo.items():
        fila = vab_df[
            (vab_df['codigo_pibe'] == codigo_pibe) &
            (vab_df['n_niveles'] == nivel)
        ]
        if len(fila) == 0:
            print(f"  ADVERTENCIA: no se encontró código PIBE '{codigo_pibe}' nivel {nivel} para grupo MIP '{grupo_mip}'")
            resultado[grupo_mip] = np.nan
        elif len(fila) > 1:
            print(f"  ADVERTENCIA: múltiples filas para '{codigo_pibe}' nivel {nivel} — tomando la primera")
            resultado[grupo_mip] = fila['valor_2023'].iloc[0]
        else:
            resultado[grupo_mip] = fila['valor_2023'].iloc[0]
    return pd.Series(resultado)


def calcular_slq(pibe_jal, pibe_nac):
    """
    SLQᵢ = (PIBEᵢ_jal / PIBE_total_jal) / (PIBEᵢ_nac / PIBE_total_nac)

    Valores > 1: Jalisco más especializado que el promedio nacional
    Valores < 1: Jalisco menos especializado (importa más de ese sector)
    """
    total_jal = pibe_jal.sum()
    total_nac = pibe_nac.sum()

    participacion_jal = pibe_jal / total_jal   # peso de cada sector en Jalisco
    participacion_nac = pibe_nac / total_nac   # peso de cada sector en México

    slq = participacion_jal / participacion_nac
    return slq, total_jal, total_nac


def calcular_cilq(slq):
    """
    CILQᵢⱼ = SLQᵢ / SLQⱼ

    Resultado: matriz 34x34
    Fila i = sector proveedor, columna j = sector comprador

    Caso especial: si SLQⱼ = 0, el sector comprador no existe en Jalisco.
    La columna completa se fija en 0 porque si el sector no existe en la
    región no puede demandar insumos localmente de ningún proveedor.
    Si SLQᵢ = 0, la fila también es 0 (no puede proveer localmente).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        matriz = np.where(
            slq.values[None, :] == 0,
            0.0,
            slq.values[:, None] / slq.values[None, :]
        )
    cilq = pd.DataFrame(matriz, index=slq.index, columns=slq.index)

    sectores_cero = slq[slq == 0].index.tolist()
    if sectores_cero:
        print(f"    Sectores con SLQ=0 (no existen en Jalisco): {sectores_cero}")
        print(f"    Sus filas y columnas en CILQ se fijan en 0.")
    return cilq


def calcular_flq(cilq, total_jal, total_nac, delta):
    """
    λ* = [log₂(1 + total_jal / total_nac)] ^ delta
    FLQᵢⱼ = CILQᵢⱼ × λ*

    El factor λ* penaliza el tamaño pequeño de Jalisco respecto al país.
    """
    # Factor de tamaño regional
    lambda_star = (np.log2(1 + total_jal / total_nac)) ** delta

    flq = cilq * lambda_star
    return flq, lambda_star


def regionalizar_mip(A_nacional, flq):
    """
    aᵢⱼ_regional = aᵢⱼ_nacional × min(FLQᵢⱼ, 1)

    Si FLQ >= 1: la región puede autoabastecerse → coeficiente nacional
    Si FLQ <  1: la región importa parte → coeficiente se reduce
    """
    # Aplicar min(FLQ, 1) celda por celda
    flq_acotado = flq.clip(upper=1.0)

    # Multiplicar elemento a elemento
    A_regional = A_nacional * flq_acotado
    return A_regional, flq_acotado


# ── Ejecución principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Cálculo de Cocientes de Localización — Jalisco 2023")
    print("=" * 60)

    # ── 1. Leer PIBE ──────────────────────────────────────────────────────────
    print("\n[1] Leyendo PIBE Jalisco y Nacional (2023)...")
    vab_jal_raw = extraer_vab_pibe(PIBE_JAL)
    vab_nac_raw = extraer_vab_pibe(PIBE_NAC)

    pibe_jal = construir_vector_pibe(vab_jal_raw, MAPEO_PIBE_A_MIP)
    pibe_nac = construir_vector_pibe(vab_nac_raw, MAPEO_PIBE_A_MIP)

    print(f"    Sectores cargados: {pibe_jal.notna().sum()} / {len(pibe_jal)}")
    print(f"    VAB total Jalisco 2023:  {pibe_jal.sum():>12,.1f} millones de pesos")
    print(f"    VAB total Nacional 2023: {pibe_nac.sum():>12,.1f} millones de pesos")
    print(f"    Participación Jalisco en PIB nacional: {pibe_jal.sum()/pibe_nac.sum()*100:.2f}%")

    # ── 2. SLQ ────────────────────────────────────────────────────────────────
    print("\n[2] Calculando SLQ...")
    slq, total_jal, total_nac = calcular_slq(pibe_jal, pibe_nac)

    especializados   = slq[slq >= 1].sort_values(ascending=False)
    no_especializados = slq[slq < 1].sort_values(ascending=False)

    print(f"\n    Sectores con SLQ >= 1 (Jalisco más especializado): {len(especializados)}")
    for s, v in especializados.items():
        print(f"      {s:<12} SLQ = {v:.4f}")

    print(f"\n    Sectores con SLQ < 1 (Jalisco importa relativamente más): {len(no_especializados)}")
    for s, v in no_especializados.items():
        print(f"      {s:<12} SLQ = {v:.4f}")

    # ── 3. CILQ ───────────────────────────────────────────────────────────────
    print("\n[3] Calculando CILQ (34×34)...")
    cilq = calcular_cilq(slq)
    print(f"    Dimensión: {cilq.shape}")
    print(f"    Rango de valores: [{cilq.min().min():.4f}, {cilq.max().max():.4f}]")

    # ── 4. FLQ ────────────────────────────────────────────────────────────────
    print(f"\n[4] Calculando FLQ con δ = {DELTA}...")
    flq, lambda_star = calcular_flq(cilq, total_jal, total_nac, DELTA)

    print(f"    Factor de tamaño λ* = {lambda_star:.6f}")
    print(f"    (Jalisco representa {total_jal/total_nac*100:.2f}% del PIB nacional)")
    print(f"    Celdas con FLQ >= 1 (se mantiene coef. nacional): {(flq >= 1).sum().sum()}")
    print(f"    Celdas con FLQ <  1 (coef. se reduce):            {(flq < 1).sum().sum()}")

    # ── 5. Regionalización ────────────────────────────────────────────────────
    print("\n[5] Aplicando FLQ a la MIP nacional...")
    A_nac = pd.read_csv(MIP_CTEC, index_col=0)

    # Alinear índices (asegurar mismo orden)
    sectores_comunes = [s for s in A_nac.index if s in flq.index]
    sectores_sin_pibe = [s for s in A_nac.index if s not in flq.index]

    if sectores_sin_pibe:
        print(f"    ADVERTENCIA: estos sectores no tienen LQ y se mantienen sin cambio: {sectores_sin_pibe}")

    A_nac_alin   = A_nac.loc[sectores_comunes, sectores_comunes]
    flq_alin     = flq.loc[sectores_comunes, sectores_comunes]
    A_reg, flq_acotado = regionalizar_mip(A_nac_alin, flq_alin)

    reduccion_promedio = 1 - (A_reg.sum().sum() / A_nac_alin.sum().sum())
    print(f"    Reducción promedio de coeficientes: {reduccion_promedio*100:.1f}%")
    print(f"    (Los coeficientes se redujeron en promedio {reduccion_promedio*100:.1f}%")
    print(f"     porque Jalisco importa esa proporción del resto del país)")

    # ── 6. Guardar ────────────────────────────────────────────────────────────
    print("\n[6] Guardando resultados...")

    # SLQ con información descriptiva
    slq_df = pd.DataFrame({
        'grupo_mip':        slq.index,
        'slq':              slq.values,
        'pibe_jalisco_2023': pibe_jal[slq.index].values,
        'pibe_nacional_2023': pibe_nac[slq.index].values,
        'particip_jalisco': (pibe_jal[slq.index] / total_jal).values,
        'particip_nacional': (pibe_nac[slq.index] / total_nac).values,
        'especializado':    (slq >= 1).values,
    })
    slq_df.to_csv(SALIDA_SLQ, index=False)
    print(f"    SLQ sectorial:        {SALIDA_SLQ}")

    flq.loc[sectores_comunes, sectores_comunes].to_csv(SALIDA_FLQ)
    print(f"    Matriz FLQ (34×34):   {SALIDA_FLQ}")

    A_reg.to_csv(SALIDA_MIP_R)
    print(f"    MIP regionalizada:    {SALIDA_MIP_R}")

    print(f"\n✓ Proceso completado. δ provisional = {DELTA}")
    print(f"  (Este δ se optimizará con ML en el siguiente paso)")
