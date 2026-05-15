"""
Agregación de la MIP Nacional 2018: de 78 sectores a 36 sectores
(excluyendo 814 Hogares con empleados domésticos y 931 Gobierno)

Metodología: Miller & Blair (2009), Cap. 2
  1. Recuperar flujos monetarios Z (78x78) y producción x (78)
  2. Agregar flujos sumando filas y columnas de sectores en el mismo grupo
  3. Recalcular coeficientes técnicos: A_agr = Z_agr / x_agr (división por columnas)

Resultado: matriz A de coeficientes técnicos (36x36) lista para el FLQ
"""

import pandas as pd
import numpy as np
import re
import os

# ── Rutas ──────────────────────────────────────────────────────────────────────
DIR_DATOS   = "/mnt/user-data/uploads"
ARCHIVO_FLUJOS = os.path.join(DIR_DATOS, "conjunto_de_datos_mip_d_pb_pxp_22018.csv")
SALIDA_CTEC    = "/mnt/user-data/outputs/mip_ctec_agregada_36x36.csv"
SALIDA_FLUJOS  = "/mnt/user-data/outputs/mip_flujos_agregada_36x36.csv"
SALIDA_PROD    = "/mnt/user-data/outputs/mip_produccion_sectorial.csv"

# ── Tabla de correspondencia MIP (78) → grupos agregados (36) ─────────────────
# Formato: código MIP : (código_grupo, nombre_grupo)
# Excluidos: 814, 931

MAPEO = {
    "111": ("111",     "Agricultura"),
    "112": ("112",     "Cría y explotación de animales"),
    "113": ("113-115", "Aprovechamiento forestal y servicios agropecuarios"),
    "114": ("114",     "Pesca, caza y captura"),
    "115": ("113-115", "Aprovechamiento forestal y servicios agropecuarios"),
    "211": ("21-1",    "Minería petrolera"),
    "212": ("21-2",    "Minería no petrolera y servicios mineros"),
    "213": ("21-2",    "Minería no petrolera y servicios mineros"),
    "221": ("22",      "Energía eléctrica, agua y gas"),
    "236": ("23",      "Construcción"),
    "237": ("23",      "Construcción"),
    "238": ("23",      "Construcción"),
    "311": ("311",     "Industria alimentaria"),
    "312": ("312",     "Industria de las bebidas y del tabaco"),
    "313": ("313-314", "Insumos y productos textiles"),
    "314": ("313-314", "Insumos y productos textiles"),
    "315": ("315-316", "Prendas de vestir, cuero y piel"),
    "316": ("315-316", "Prendas de vestir, cuero y piel"),
    "321": ("321",     "Industria de la madera"),
    "322": ("322-323", "Industria del papel e impresión"),
    "323": ("322-323", "Industria del papel e impresión"),
    "324": ("324-326", "Petróleo, carbón, química, plástico y hule"),
    "325": ("324-326", "Petróleo, carbón, química, plástico y hule"),
    "326": ("324-326", "Petróleo, carbón, química, plástico y hule"),
    "327": ("327",     "Minerales no metálicos"),
    "331": ("331-332", "Industrias metálicas y productos metálicos"),
    "332": ("331-332", "Industrias metálicas y productos metálicos"),
    "333": ("333-336", "Maquinaria, electrónica, eléctrico y transporte"),
    "334": ("333-336", "Maquinaria, electrónica, eléctrico y transporte"),
    "335": ("333-336", "Maquinaria, electrónica, eléctrico y transporte"),
    "336": ("333-336", "Maquinaria, electrónica, eléctrico y transporte"),
    "337": ("337",     "Muebles, colchones y persianas"),
    "339": ("339",     "Otras industrias manufactureras"),
    "431": ("43",      "Comercio al por mayor"),
    "461": ("46",      "Comercio al por menor"),
    "481": ("48-49",   "Transportes, correos y almacenamiento"),
    "482": ("48-49",   "Transportes, correos y almacenamiento"),
    "483": ("48-49",   "Transportes, correos y almacenamiento"),
    "484": ("48-49",   "Transportes, correos y almacenamiento"),
    "485": ("48-49",   "Transportes, correos y almacenamiento"),
    "486": ("48-49",   "Transportes, correos y almacenamiento"),
    "487": ("48-49",   "Transportes, correos y almacenamiento"),
    "488": ("48-49",   "Transportes, correos y almacenamiento"),
    "491": ("48-49",   "Transportes, correos y almacenamiento"),
    "492": ("48-49",   "Transportes, correos y almacenamiento"),
    "493": ("48-49",   "Transportes, correos y almacenamiento"),
    "511": ("51",      "Información en medios masivos"),
    "512": ("51",      "Información en medios masivos"),
    "515": ("51",      "Información en medios masivos"),
    "517": ("51",      "Información en medios masivos"),
    "518": ("51",      "Información en medios masivos"),
    "519": ("51",      "Información en medios masivos"),
    "521": ("52",      "Servicios financieros y de seguros"),
    "522": ("52",      "Servicios financieros y de seguros"),
    "523": ("52",      "Servicios financieros y de seguros"),
    "524": ("52",      "Servicios financieros y de seguros"),
    "531": ("53",      "Servicios inmobiliarios y de alquiler"),
    "532": ("53",      "Servicios inmobiliarios y de alquiler"),
    "533": ("53",      "Servicios inmobiliarios y de alquiler"),
    "541": ("54",      "Servicios profesionales, científicos y técnicos"),
    "551": ("55",      "Corporativos"),
    "561": ("56",      "Servicios de apoyo a los negocios y manejo de residuos"),
    "562": ("56",      "Servicios de apoyo a los negocios y manejo de residuos"),
    "611": ("61",      "Servicios educativos"),
    "621": ("62",      "Servicios de salud y asistencia social"),
    "622": ("62",      "Servicios de salud y asistencia social"),
    "623": ("62",      "Servicios de salud y asistencia social"),
    "624": ("62",      "Servicios de salud y asistencia social"),
    "711": ("71",      "Servicios de esparcimiento culturales y deportivos"),
    "712": ("71",      "Servicios de esparcimiento culturales y deportivos"),
    "713": ("71",      "Servicios de esparcimiento culturales y deportivos"),
    "721": ("72",      "Alojamiento temporal y preparación de alimentos"),
    "722": ("72",      "Alojamiento temporal y preparación de alimentos"),
    "811": ("81",      "Otros servicios excepto actividades gubernamentales"),
    "812": ("81",      "Otros servicios excepto actividades gubernamentales"),
    "813": ("81",      "Otros servicios excepto actividades gubernamentales"),
    # Excluidos: sin entrada en el mapeo
    # "814": excluido - Hogares con empleados domésticos
    # "931": excluido - Actividades gubernamentales
}

# Orden canónico de los 36 grupos (para que la matriz quede ordenada)
ORDEN_GRUPOS = [
    "111", "112", "113-115", "114",
    "21-1", "21-2", "22", "23",
    "311", "312", "313-314", "315-316", "321", "322-323",
    "324-326", "327", "331-332", "333-336", "337", "339",
    "43", "46",
    "48-49", "51", "52", "53", "54", "55", "56",
    "61", "62", "71", "72", "81",
]

# ── Funciones auxiliares ───────────────────────────────────────────────────────

def extraer_codigo_mip(descriptor):
    """Extrae el código de 3 dígitos del descriptor '111---Agricultura'."""
    m = re.match(r'^(\d{3})---', str(descriptor))
    return m.group(1) if m else None


def leer_mip_flujos(ruta):
    """
    Lee el archivo de flujos y devuelve:
      - Z: DataFrame (78x78) de flujos monetarios en millones de pesos
      - x: Series (78) de producción bruta por sector
      - codigos: lista de códigos MIP en el mismo orden que Z
    """
    df = pd.read_csv(ruta, encoding='utf-8-sig')

    # Identificar filas de sectores productivos (descriptor empieza con 3 dígitos)
    mask_sector = df['Descriptores'].str.match(r'^\d{3}---')
    df_sect = df[mask_sector].copy()

    # Extraer códigos
    df_sect['codigo'] = df_sect['Descriptores'].apply(extraer_codigo_mip)
    codigos = df_sect['codigo'].tolist()

    # Columnas de demanda intermedia (una por cada sector comprador)
    di_prefix = 'DI---Demanda intermedia|'
    di_cols = [c for c in df.columns if c.startswith(di_prefix) and 'Total' not in c]

    # Verificar que hay exactamente 78 columnas DI
    assert len(di_cols) == 78, f"Se esperaban 78 columnas DI, se encontraron {len(di_cols)}"

    # Extraer códigos de las columnas DI (para ordenar igual que las filas)
    codigos_col = []
    for c in di_cols:
        m = re.search(r'\|(\d{3})---', c)
        codigos_col.append(m.group(1) if m else None)

    # Construir matriz Z (filas = sectores proveedores, columnas = sectores compradores)
    Z = df_sect[di_cols].values.astype(float)
    Z = pd.DataFrame(Z, index=codigos, columns=codigos_col)

    # Producción bruta por sector (columna UPPB de la fila del sector)
    x = df_sect.set_index('codigo')['UPPB---Utilización de la producción a precios básicos']
    x = x.astype(float)

    return Z, x, codigos


def agregar_mip(Z, x, mapeo, orden_grupos):
    """
    Agrega la matriz de flujos Z (78x78) y el vector de producción x (78)
    según el mapeo de sectores.

    Devuelve:
      - Z_agr: DataFrame (36x36) de flujos agregados
      - x_agr: Series (36) de producción agregada
      - A_agr: DataFrame (36x36) de coeficientes técnicos recalculados
    """
    # Sectores incluidos (los que tienen mapeo)
    codigos_incluidos = list(mapeo.keys())
    codigos_excluidos = [c for c in Z.index if c not in mapeo]

    print(f"\nSectores incluidos: {len(codigos_incluidos)}")
    print(f"Sectores excluidos: {codigos_excluidos}")

    # Filtrar Z y x a los sectores incluidos
    Z_inc = Z.loc[codigos_incluidos, codigos_incluidos]
    x_inc = x[codigos_incluidos]

    # Añadir columna de grupo a cada sector
    grupos_fila = [mapeo[c][0] for c in Z_inc.index]
    grupos_col  = [mapeo[c][0] for c in Z_inc.columns]

    Z_inc = Z_inc.copy()
    Z_inc.index   = grupos_fila
    Z_inc.columns = grupos_col

    # Agregar sumando filas del mismo grupo y luego columnas del mismo grupo
    Z_agr = Z_inc.groupby(level=0).sum()
    Z_agr = Z_agr.T.groupby(level=0).sum().T

    # Agregar producción sumando sectores del mismo grupo
    x_inc_copy = x_inc.copy()
    x_inc_copy.index = grupos_fila
    x_agr = x_inc_copy.groupby(level=0).sum()

    # Reordenar según el orden canónico
    grupos_presentes = [g for g in orden_grupos if g in Z_agr.index]
    Z_agr = Z_agr.loc[grupos_presentes, grupos_presentes]
    x_agr = x_agr[grupos_presentes]

    # Recalcular coeficientes técnicos: A = Z / x (dividir cada columna j entre xⱼ)
    A_agr = Z_agr.div(x_agr, axis='columns')

    return Z_agr, x_agr, A_agr


def verificar_matriz(A_agr, x_agr, Z_agr):
    """
    Verificaciones de consistencia:
    1. Todos los coeficientes >= 0
    2. Suma de cada columna de A < 1 (condición necesaria para estabilidad)
    3. Reconstruir Z desde A y x, comparar con Z original
    """
    print("\n── Verificaciones de consistencia ──")

    # 1. No negativos
    neg = (A_agr < 0).sum().sum()
    print(f"  Coeficientes negativos: {neg}  {'✓' if neg == 0 else '✗ REVISAR'}")

    # 2. Suma de columnas < 1
    col_sums = A_agr.sum(axis=0)
    max_col = col_sums.max()
    sectores_inestables = col_sums[col_sums >= 1].index.tolist()
    print(f"  Máxima suma de columna: {max_col:.4f}  {'✓' if max_col < 1 else '✗ INESTABLE'}")
    if sectores_inestables:
        print(f"  Sectores con suma >= 1: {sectores_inestables}")

    # 3. Reconstrucción
    Z_reconstruido = A_agr.mul(x_agr, axis='columns')
    diff_max = (Z_reconstruido - Z_agr).abs().max().max()
    print(f"  Error máximo de reconstrucción: {diff_max:.6f} millones de pesos  {'✓' if diff_max < 0.01 else '✗ REVISAR'}")

    # 4. Producción total
    print(f"\n  Producción total nacional 2018:")
    print(f"    Suma x_agr: {x_agr.sum():,.1f} millones de pesos")
    print(f"    Top 5 sectores por producción:")
    top5 = x_agr.sort_values(ascending=False).head(5)
    for sector, val in top5.items():
        print(f"      {sector}: {val:,.1f}")

    return col_sums


# ── Ejecución principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Agregación MIP Nacional 2018: 78 → 36 sectores")
    print("=" * 60)

    # 1. Leer datos
    print("\n[1] Leyendo archivo de flujos...")
    Z, x, codigos = leer_mip_flujos(ARCHIVO_FLUJOS)
    print(f"    Matriz Z leída: {Z.shape}")
    print(f"    Vector x leído: {len(x)} sectores")
    print(f"    Producción total: {x.sum():,.1f} millones de pesos")

    # 2. Agregar
    print("\n[2] Aplicando tabla de correspondencia y agregando...")
    Z_agr, x_agr, A_agr = agregar_mip(Z, x, MAPEO, ORDEN_GRUPOS)
    print(f"    Matriz agregada: {A_agr.shape}")

    # 3. Verificar
    col_sums = verificar_matriz(A_agr, x_agr, Z_agr)

    # 4. Guardar resultados
    print("\n[4] Guardando archivos...")

    # Coeficientes técnicos (lo que se usará en el FLQ)
    A_agr.to_csv(SALIDA_CTEC)
    print(f"    Coeficientes técnicos: {SALIDA_CTEC}")

    # Flujos en pesos (para referencia y validación)
    Z_agr.to_csv(SALIDA_FLUJOS)
    print(f"    Flujos en pesos:       {SALIDA_FLUJOS}")

    # Producción por sector (se usará en el RAS)
    x_df = pd.DataFrame({
        'grupo': x_agr.index,
        'nombre': [MAPEO.get(
            next((k for k, v in MAPEO.items() if v[0] == g), g), (g, g)
        )[1] if g not in [v[0] for v in MAPEO.values()] else
            next(v[1] for k, v in MAPEO.items() if v[0] == g)
            for g in x_agr.index],
        'produccion_millones_pesos': x_agr.values,
        'suma_columna_A': col_sums.values
    })
    x_df.to_csv(SALIDA_PROD, index=False)
    print(f"    Producción sectorial:  {SALIDA_PROD}")

    print("\n✓ Proceso completado.")
    print("\nResumen de la matriz A agregada:")
    print(f"  Dimensión: {A_agr.shape[0]} × {A_agr.shape[1]}")
    print(f"  Sectores: {list(A_agr.index)}")
