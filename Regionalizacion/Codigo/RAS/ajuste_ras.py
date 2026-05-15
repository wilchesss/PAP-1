"""
Ajuste RAS de la MIP regionalizada de Jalisco
Actualiza la estructura de 2018 a las condiciones observadas en Censos 2023.

El RAS (Row and Column Scaling) es un proceso iterativo que ajusta una matriz
de coeficientes para que sus márgenes (sumas de filas y columnas) coincidan
con vectores objetivo observados.

Vectores objetivo (de Censos Económicos Jalisco 2023/2024):
  - u (filas):    consumo intermedio que VENDE cada sector como proveedor
                  = suma de su fila en la matriz de flujos
  - v (columnas): consumo intermedio que COMPRA cada sector
                  = A121A de los Censos (gasto total en insumos)

Para los 3 sectores sin datos de Censos (111, 21-1, 21-2):
  sus filas y columnas se mantienen sin ajuste (coeficientes del FLQ).

Convergencia: diferencia máxima entre iteraciones < tol = 1e-6

Referencia:
  Bacharach, M. (1970). Biproportional Matrices and Input-Output Change.
  Cambridge University Press.
  Miller, R. & Blair, P. (2009). Input-Output Analysis. Cambridge, Cap. 7.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os
warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────────
DIR_IN  = "/mnt/user-data/uploads"
DIR_OUT = "/mnt/user-data/outputs"

ARCHIVO_MIP_REG = f"{DIR_OUT}/mip_regionalizada_optima.csv"
ARCHIVO_CENSO   = f"{DIR_IN}/tr_ce_jal_2024.csv"

SALIDA_MIP_RAS  = f"{DIR_OUT}/mip_jalisco_final.csv"
SALIDA_MULT     = f"{DIR_OUT}/multiplicadores_leontief.csv"
SALIDA_CONV     = f"{DIR_OUT}/convergencia_ras.csv"
SALIDA_GRAF     = f"{DIR_OUT}/grafica_ras_resultados.png"

TOL      = 1e-4
MAX_ITER = 2000

MAPEO_CENSO = {
    "112":"112","113":"113-115","114":"114","115":"113-115",
    "221":"22","236":"23","237":"23","238":"23",
    "311":"311","312":"312","313":"313-314","314":"313-314",
    "315":"315-316","316":"315-316","321":"321",
    "322":"322-323","323":"322-323","324":"324-326","325":"324-326","326":"324-326",
    "327":"327","331":"331-332","332":"331-332","333":"333-336","334":"333-336",
    "335":"333-336","336":"333-336","337":"337","339":"339",
    "431":"43","432":"43","433":"43","434":"43","435":"43","436":"43","437":"43",
    "461":"46","462":"46","463":"46","464":"46","465":"46","466":"46",
    "467":"46","468":"46","469":"46",
    "484":"48-49","485":"48-49","487":"48-49","488":"48-49","492":"48-49","493":"48-49",
    "511":"51","512":"51","515":"51","517":"51","518":"51","519":"51",
    "521":"52","522":"52","523":"52","524":"52",
    "531":"53","532":"53","533":"53","541":"54","551":"55",
    "561":"56","562":"56","611":"61",
    "621":"62","622":"62","623":"62","624":"62",
    "711":"71","712":"71","713":"71","721":"72","722":"72",
    "811":"81","812":"81","813":"81",
}

NOMBRES = {
    "111":"Agricultura","112":"Ganadería","113-115":"Forestal",
    "114":"Pesca","21-1":"Min.petrol.","21-2":"Min.no petrol.",
    "22":"Energía","23":"Construcción","311":"Alimentos",
    "312":"Bebidas","313-314":"Textiles","315-316":"Prendas",
    "321":"Madera","322-323":"Papel","324-326":"Petroquímica",
    "327":"Min.no metal.","331-332":"Metálicas","333-336":"Maquinaria/elect.",
    "337":"Muebles","339":"Otras manuf.","43":"Com.mayor.",
    "46":"Com.menor.","48-49":"Transportes","51":"Información",
    "52":"Financiero","53":"Inmobiliario","54":"Serv.prof.",
    "55":"Corporativos","56":"Serv.apoyo","61":"Educación",
    "62":"Salud","71":"Esparcimiento","72":"Alojamiento","81":"Otros serv.",
}

# ── Cargar datos ───────────────────────────────────────────────────────────────

def cargar_censos():
    censo = pd.read_csv(ARCHIVO_CENSO, encoding='utf-8-sig', low_memory=False)
    censo['COD'] = censo['CODIGO'].astype(str).str.strip()
    sub = censo[censo['COD'].str.len() == 3]
    tot = sub[(sub['E04'].isna()) & (sub['ID_ESTRATO'].isna())].copy()
    tot['A121A'] = pd.to_numeric(tot['A121A'], errors='coerce')
    tot['A111A'] = pd.to_numeric(tot['A111A'], errors='coerce')

    ci_obs, pb_obs = {}, {}
    for _, row in tot.iterrows():
        cod = row['COD']
        if cod in MAPEO_CENSO and pd.notna(row['A121A']):
            g = MAPEO_CENSO[cod]
            ci_obs[g] = ci_obs.get(g, 0) + row['A121A']
            pb_obs[g] = pb_obs.get(g, 0) + (row['A111A'] if pd.notna(row['A111A']) else 0)

    return pd.Series(ci_obs), pd.Series(pb_obs)


# ── Algoritmo RAS ──────────────────────────────────────────────────────────────

def ras(Z0, u_target, v_target, sectores_fijos, tol=1e-4, max_iter=2000):
    """
    RAS parcial: ajusta solo el bloque libre de la matriz.

    Los sectores fijos (sin datos en Censos) mantienen sus flujos intactos.
    Sus contribuciones se restan de los vectores objetivo antes de iterar,
    de modo que el algoritmo opera sobre un sistema perfectamente consistente.

    Referencia: Bacharach (1970), Miller & Blair (2009) Cap. 7.
    """
    Z = Z0.copy().values.astype(float)
    idx = Z0.index.tolist()

    # Separar índices libres y fijos
    pos_libres = [i for i, s in enumerate(idx) if s not in sectores_fijos]
    pos_fijos  = [i for i, s in enumerate(idx) if s in sectores_fijos]

    # ── Vectores objetivo ajustados ────────────────────────────────────────────
    # u_adj[i] = u_target[i] - (lo que ya aportan las columnas fijas a la fila i)
    # v_adj[j] = v_target[j] - (lo que ya aportan las filas fijas a la columna j)
    u_vec = np.array([u_target.get(idx[i], Z0.loc[idx[i],:].sum())
                      for i in pos_libres])
    v_vec = np.array([v_target.get(idx[j], Z0.loc[:,idx[j]].sum())
                      for j in pos_libres])

    # Restar contribución fija de filas a las sumas de columna
    for pf in pos_fijos:
        for k, pl in enumerate(pos_libres):
            v_vec[k] -= Z[pf, pl]   # fila fija contribuye a columna libre

    # Restar contribución fija de columnas a las sumas de fila
    for pf in pos_fijos:
        for k, pl in enumerate(pos_libres):
            u_vec[k] -= Z[pl, pf]   # columna fija contribuye a fila libre

    # Verificar que todos los targets sean positivos
    if np.any(u_vec <= 0) or np.any(v_vec <= 0):
        print("    ADVERTENCIA: algunos targets ajustados son <= 0. Revisa sectores fijos.")

    # ── Submatriz libre ────────────────────────────────────────────────────────
    Z_libre = Z[np.ix_(pos_libres, pos_libres)]
    historial = []

    for iteracion in range(1, max_iter + 1):
        Z_prev = Z_libre.copy()

        # Paso R: escalar filas
        sumas_fila = Z_libre.sum(axis=1)
        for i in range(len(pos_libres)):
            if sumas_fila[i] > 1e-12:
                Z_libre[i, :] *= u_vec[i] / sumas_fila[i]

        # Paso S: escalar columnas
        sumas_col = Z_libre.sum(axis=0)
        for j in range(len(pos_libres)):
            if sumas_col[j] > 1e-12:
                Z_libre[:, j] *= v_vec[j] / sumas_col[j]

        diff = np.max(np.abs(Z_libre - Z_prev))
        historial.append(diff)

        if diff < tol:
            print(f"    Convergió en {iteracion} iteraciones (diff = {diff:.2e})")
            break
    else:
        print(f"    Convergió parcialmente en {max_iter} iter (diff = {diff:.2e})")

    # Reconstruir Z completa con bloque libre ajustado y bloques fijos intactos
    Z[np.ix_(pos_libres, pos_libres)] = Z_libre
    Z_df = pd.DataFrame(Z, index=idx, columns=idx)
    return Z_df, historial, iteracion


# ── Gráficas ───────────────────────────────────────────────────────────────────

def generar_graficas(hist_conv, A_reg, A_ras, mult_reg, mult_ras,
                     ci_pred_antes, ci_pred_despues, ci_obs, sectores_v):

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Ajuste RAS — MIP Regionalizada de Jalisco\n"
        "Actualización de estructura 2018 a condiciones observadas Censos 2023",
        fontsize=13, fontweight='bold', y=0.99
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    C = {'azul':'#2563EB', 'verde':'#059669', 'rojo':'#DC2626',
         'morado':'#7C3AED', 'fondo':'#F8FAFC', 'grid':'#E2E8F0'}

    # ── A. Convergencia del RAS ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(C['fondo'])
    ax1.grid(True, color=C['grid'], linewidth=0.8)
    ax1.semilogy(range(1, len(hist_conv)+1), hist_conv,
                 color=C['azul'], linewidth=2, zorder=4)
    ax1.axhline(TOL, color=C['rojo'], linewidth=1.5, linestyle='--',
                label=f'Umbral convergencia = {TOL:.0e}')
    ax1.set_xlabel('Iteración', fontsize=11)
    ax1.set_ylabel('Diferencia máxima entre iteraciones (escala log)', fontsize=10)
    ax1.set_title('A. Convergencia del algoritmo RAS\n(cuántas iteraciones tomó ajustar la matriz)',
                  fontsize=11)
    ax1.legend(fontsize=9)

    # ── B. CI predicho antes y después del RAS vs observado ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['fondo'])
    ax2.grid(True, color=C['grid'], linewidth=0.8, axis='y')

    x = np.arange(len(sectores_v))
    w = 0.28
    nombres_s = [NOMBRES.get(s, s) for s in sectores_v]

    ax2.bar(x - w, ci_obs[sectores_v].values / 1e3,    w,
            color=C['verde'],  alpha=0.85, label='Observado (Censos)', zorder=3)
    ax2.bar(x,     ci_pred_antes[sectores_v].values / 1e3,  w,
            color=C['azul'],   alpha=0.75, label='Antes RAS (FLQ)', zorder=3)
    ax2.bar(x + w, ci_pred_despues[sectores_v].values / 1e3, w,
            color=C['rojo'],   alpha=0.75, label='Después RAS', zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres_s, rotation=55, ha='right', fontsize=6.5)
    ax2.set_ylabel('Consumo intermedio (miles de MDP)', fontsize=10)
    ax2.set_title('B. CI por sector: antes y después del RAS vs observado\n'
                  '(el RAS acerca las barras azul/roja a la verde)',
                  fontsize=11)
    ax2.legend(fontsize=8)

    # ── C. Cambio en coeficientes técnicos (antes vs después RAS) ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(C['fondo'])
    ax3.grid(True, color=C['grid'], linewidth=0.8)

    # Diferencia relativa promedio por sector (como comprador)
    diff_rel = ((A_ras - A_reg).abs().mean(axis=0) /
                A_reg.replace(0, np.nan).mean(axis=0)).fillna(0) * 100

    colores_bar = [C['rojo'] if v > diff_rel.mean() else C['azul']
                   for v in diff_rel.values]
    ax3.bar([NOMBRES.get(s, s) for s in diff_rel.index],
            diff_rel.values, color=colores_bar, alpha=0.85, zorder=3)
    ax3.axhline(diff_rel.mean(), color='black', linewidth=1.5,
                linestyle='--', label=f'Promedio = {diff_rel.mean():.1f}%')
    ax3.set_xticklabels([NOMBRES.get(s, s) for s in diff_rel.index],
                        rotation=55, ha='right', fontsize=6.5)
    ax3.set_ylabel('Cambio promedio en coeficientes (%)', fontsize=10)
    ax3.set_title('C. Magnitud del ajuste RAS por sector\n'
                  '(rojo = ajuste mayor al promedio)', fontsize=11)
    ax3.legend(fontsize=9)

    # ── D. Multiplicadores antes y después del RAS ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(C['fondo'])
    ax4.grid(True, color=C['grid'], linewidth=0.8, axis='y')

    sectores_plot = [s for s in A_reg.index if s in sectores_v]
    nombres_plot  = [NOMBRES.get(s, s) for s in sectores_plot]
    x2 = np.arange(len(sectores_plot))
    w2 = 0.35

    ax4.bar(x2 - w2/2, mult_reg[sectores_plot].values, w2,
            color=C['azul'],  alpha=0.80, label='Antes RAS (δ=0.25)', zorder=3)
    ax4.bar(x2 + w2/2, mult_ras[sectores_plot].values, w2,
            color=C['rojo'],  alpha=0.80, label='Después RAS', zorder=3)
    ax4.axhline(1, color='black', linewidth=0.8, linestyle=':')

    ax4.set_xticks(x2)
    ax4.set_xticklabels(nombres_plot, rotation=55, ha='right', fontsize=6.5)
    ax4.set_ylabel('Multiplicador de producción de Leontief', fontsize=10)
    ax4.set_title('D. Multiplicadores de Leontief antes y después del RAS\n'
                  '(resultado final del proyecto)', fontsize=11)
    ax4.legend(fontsize=9)

    plt.savefig(SALIDA_GRAF, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Guardada: {SALIDA_GRAF}")


# ── Ejecución principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ajuste RAS — MIP Jalisco 2023")
    print("=" * 60)

    # 1. Cargar inputs
    print("\n[1] Cargando MIP regionalizada y Censos...")
    A_reg = pd.read_csv(ARCHIVO_MIP_REG, index_col=0)
    ci_obs, pb_obs = cargar_censos()

    sectores       = A_reg.index.tolist()
    sectores_fijos = [s for s in sectores if s not in ci_obs.index]
    sectores_v     = [s for s in sectores if s in ci_obs.index]

    print(f"  Sectores totales:          {len(sectores)}")
    print(f"  Con restricción RAS:       {len(sectores_v)}")
    print(f"  Sin restricción (fijos):   {sectores_fijos}")

    # 2. Construir vectores objetivo u y v
    # u[i] = cuánto vende el sector i en total como proveedor de insumos
    # v[j] = cuánto compra el sector j en total como demandante de insumos
    #
    # Ambos los derivamos del CI observado en Censos.
    # El CI observado (A121A) representa las compras de insumos de cada sector,
    # es decir es directamente v (sumas de columna objetivo).
    #
    # Para u (sumas de fila objetivo) usamos la misma información:
    # asumimos que la distribución de ventas como proveedor es proporcional
    # a la participación de cada sector en el CI total observado.
    # Esto es el supuesto estándar cuando no se dispone de una encuesta
    # de uso de insumos regional (ver Flegg & Tohmo, 2014, sección 3.2).

    v_target = ci_obs.copy()   # sumas de columna = compras de cada sector

    # Para u: escalar las sumas de fila de la MIP regionalizada para que
    # el total coincida con el total de CI observado en Censos
    sumas_fila_reg = A_reg.loc[sectores_v, :].sum(axis=1) * pb_obs[sectores_v]
    factor_escala  = ci_obs[sectores_v].sum() / sumas_fila_reg.sum()
    u_target = sumas_fila_reg * factor_escala

    print(f"\n  Total CI observado (Censos): {ci_obs[sectores_v].sum():>12,.1f} MDP")
    print(f"  Total CI predicho (FLQ):    {sumas_fila_reg.sum():>12,.1f} MDP")
    print(f"  Factor de escala u:          {factor_escala:.4f}")

    # Verificación de consistencia: suma(u) debe = suma(v)
    diff_uv = abs(u_target.sum() - v_target[sectores_v].sum())
    print(f"  Diferencia |sum(u) - sum(v)|: {diff_uv:.2f} MDP "
          f"({'✓' if diff_uv < 1 else '⚠ revisar'})")

    # 3. Convertir A_reg a matriz de flujos Z para el RAS
    # Z_ij = a_ij × x_j  (coeficiente × producción del sector comprador)
    print("\n[2] Convirtiendo coeficientes a flujos para el RAS...")
    Z_reg = A_reg.copy()
    for j in sectores_v:
        Z_reg.loc[:, j] = A_reg.loc[:, j] * pb_obs.get(j, 1)

    print(f"  Suma total de flujos Z (FLQ): {Z_reg.loc[sectores_v, sectores_v].sum().sum():>12,.1f} MDP")

    # 4. Ejecutar RAS
    print("\n[3] Ejecutando RAS...")
    Z_ras, historial, n_iter = ras(
        Z_reg, u_target, v_target, sectores_fijos, TOL, MAX_ITER
    )

    # 5. Recalcular coeficientes técnicos desde Z ajustada
    print("\n[4] Recalculando coeficientes técnicos...")
    A_ras = Z_ras.copy()
    for j in sectores_v:
        if pb_obs.get(j, 0) > 0:
            A_ras.loc[:, j] = Z_ras.loc[:, j] / pb_obs[j]
    # Sectores fijos: mantienen coeficientes del FLQ
    for j in sectores_fijos:
        A_ras.loc[:, j] = A_reg.loc[:, j]

    # 6. Verificaciones
    print("\n[5] Verificaciones de la MIP final...")
    neg = (A_ras < 0).sum().sum()
    print(f"  Coeficientes negativos:    {neg}  {'✓' if neg==0 else '✗'}")
    radio = np.max(np.abs(np.linalg.eigvals(A_ras.values)))
    print(f"  Radio espectral:           {radio:.6f}  {'✓ estable' if radio < 1 else '✗'}")
    col_sums = A_ras.sum(axis=0)
    print(f"  Max suma columna:          {col_sums.max():.4f}  {'✓' if col_sums.max()<1 else '✗'}")

    # Error residual del RAS
    ci_pred_despues = pd.Series({
        j: Z_ras.loc[:, j].sum() for j in sectores_v
    })
    wmape_final = ((ci_pred_despues - ci_obs[sectores_v]).abs().sum()
                   / ci_obs[sectores_v].sum())
    print(f"  WMAPE residual post-RAS:   {wmape_final*100:.4f}%")

    # 7. Calcular multiplicadores de Leontief
    print("\n[6] Calculando multiplicadores de Leontief...")
    I = np.eye(len(A_ras))
    L = pd.DataFrame(np.linalg.inv(I - A_ras.values),
                     index=A_ras.index, columns=A_ras.columns)
    mult_ras = L.sum(axis=0)

    # Multiplicadores antes del RAS (para comparación en gráfica)
    L_reg = pd.DataFrame(np.linalg.inv(I - A_reg.values),
                         index=A_reg.index, columns=A_reg.columns)
    mult_reg = L_reg.sum(axis=0)

    print(f"  Multiplicador promedio:    {mult_ras[sectores_v].mean():.4f}")
    print(f"  Sector con mult. mayor:    {mult_ras.idxmax()} "
          f"({mult_ras.max():.4f})")
    print(f"  Sector con mult. menor:    {mult_ras.idxmin()} "
          f"({mult_ras.min():.4f})")

    print(f"\n  Top 5 sectores por multiplicador (MIP final Jalisco):")
    for s, v in mult_ras.sort_values(ascending=False).head(5).items():
        print(f"    {s:<12} ({NOMBRES.get(s,s):<25}) {v:.4f}")

    # 8. Guardar archivos
    print("\n[7] Guardando archivos...")
    A_ras.to_csv(SALIDA_MIP_RAS)
    print(f"  MIP Jalisco final:         {SALIDA_MIP_RAS}")

    # Tabla de multiplicadores con nombres
    mult_df = pd.DataFrame({
        'sector':          mult_ras.index,
        'nombre':          [NOMBRES.get(s, s) for s in mult_ras.index],
        'mult_produccion': mult_ras.values,
        'mult_antes_ras':  mult_reg[mult_ras.index].values,
        'cambio_pct':      ((mult_ras - mult_reg[mult_ras.index])
                            / mult_reg[mult_ras.index] * 100).values,
    }).sort_values('mult_produccion', ascending=False)
    mult_df.to_csv(SALIDA_MULT, index=False)
    print(f"  Multiplicadores Leontief:  {SALIDA_MULT}")

    # Historial de convergencia
    conv_df = pd.DataFrame({
        'iteracion': range(1, len(historial)+1),
        'diff_maxima': historial
    })
    conv_df.to_csv(SALIDA_CONV, index=False)
    print(f"  Historial convergencia:    {SALIDA_CONV}")

    # 9. Gráficas
    print("\n[8] Generando gráficas...")
    ci_pred_antes = pd.Series({
        j: Z_reg.loc[:, j].sum() for j in sectores_v
    })
    generar_graficas(historial, A_reg, A_ras, mult_reg, mult_ras,
                     ci_pred_antes, ci_pred_despues, ci_obs, sectores_v)

    print("\n✓ RAS completado.")
    print(f"  La MIP final de Jalisco está lista para análisis de impacto.")
    print(f"  Archivo principal: mip_jalisco_final.csv")
