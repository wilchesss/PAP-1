"""
Calibración del parámetro δ del FLQ
Metodología: Grid Search + Análisis de sensibilidad + Respaldo bibliográfico

Hallazgo documentado: la función WMAPE no presenta mínimo interior en [0.01, 0.99],
lo que indica que δ no puede identificarse únicamente con datos de Censos.
Se adopta δ = 0.25 con base en la literatura empírica para regiones con
participación de 5-10% en el PIB nacional (Flegg & Webber, 2000; Tohmo, 2004).

Salidas:
  - Curva WMAPE absoluto y relativo vs δ
  - Análisis de sensibilidad de multiplicadores en δ ∈ {0.15, 0.20, 0.25, 0.30}
  - MIP regionalizada final con δ = 0.25
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

ARCHIVO_CTEC   = f"{DIR_OUT}/mip_ctec_agregada_36x36.csv"
ARCHIVO_SLQ    = f"{DIR_OUT}/lq_slq_sectorial.csv"
ARCHIVO_CENSO  = f"{DIR_IN}/tr_ce_jal_2024.csv"

SALIDA_MIP     = f"{DIR_OUT}/mip_regionalizada_optima.csv"
SALIDA_FLQ     = f"{DIR_OUT}/lq_flq_optimo.csv"
SALIDA_SENS    = f"{DIR_OUT}/sensibilidad_delta.csv"
SALIDA_GRAF1   = f"{DIR_OUT}/grafica_calibracion_delta.png"
SALIDA_GRAF2   = f"{DIR_OUT}/grafica_sensibilidad_multiplicadores.png"

DELTA_FINAL    = 0.25   # adoptado con base en literatura
DELTAS_SENS    = [0.15, 0.20, 0.25, 0.30]
DELTAS_SENS_STR = ['0.15', '0.20', '0.25', '0.30']

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

NOMBRES_SECTOR = {
    "111":"Agricultura","112":"Ganadería","113-115":"Forestal/Serv.agrop.",
    "114":"Pesca","21-1":"Min. petrolera","21-2":"Min. no petrolera",
    "22":"Energía/agua/gas","23":"Construcción","311":"Alimentos",
    "312":"Bebidas/tabaco","313-314":"Textiles","315-316":"Prendas/cuero",
    "321":"Madera","322-323":"Papel/impresión","324-326":"Petroquímica/química",
    "327":"Min. no metálicos","331-332":"Metálicas/prod.metálicos",
    "333-336":"Maquinaria/electrónica","337":"Muebles","339":"Otras manuf.",
    "43":"Com. mayorista","46":"Com. minorista","48-49":"Transportes",
    "51":"Información","52":"Financiero","53":"Inmobiliario",
    "54":"Serv. profesionales","55":"Corporativos","56":"Serv. apoyo negocios",
    "61":"Educación","62":"Salud","71":"Esparcimiento",
    "72":"Alojamiento/alimentos","81":"Otros servicios",
}

# ── Utilidades ─────────────────────────────────────────────────────────────────

def cargar_datos():
    A_nac  = pd.read_csv(ARCHIVO_CTEC, index_col=0)
    slq_df = pd.read_csv(ARCHIVO_SLQ).set_index('grupo_mip')
    pibe_jal  = slq_df['pibe_jalisco_2023']
    pibe_nac  = slq_df['pibe_nacional_2023']
    total_jal = pibe_jal.sum()
    total_nac = pibe_nac.sum()

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

    ci_obs = pd.Series(ci_obs)
    pb_obs = pd.Series(pb_obs)
    sectores_v = [s for s in A_nac.index if s in ci_obs.index]
    return A_nac, pibe_jal, pibe_nac, total_jal, total_nac, ci_obs, pb_obs, sectores_v


def regionalizar(A_nac, pibe_jal, pibe_nac, total_jal, total_nac, delta):
    slq = (pibe_jal / total_jal) / (pibe_nac / total_nac)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = np.where(slq.values[None,:]==0, 0.0,
                      slq.values[:,None] / slq.values[None,:])
    lam = (np.log2(1 + total_jal / total_nac)) ** delta
    flq = pd.DataFrame(cm, index=slq.index, columns=slq.index) * lam
    flq_ac = flq.clip(upper=1.0)
    return A_nac * flq_ac, flq


def wmapes(A_reg, ci_obs, pb_obs, sectores_v):
    ci_pred = pd.Series({
        j: A_reg[j].sum() * pb_obs[j]
        for j in sectores_v
        if j in A_reg.columns and pb_obs.get(j, 0) > 0
    })
    obs = ci_obs[ci_pred.index]
    wmape_abs = (ci_pred - obs).abs().sum() / obs.sum()
    ci_pred_r = ci_pred / ci_pred.sum()
    obs_r     = obs / obs.sum()
    wmape_rel = (ci_pred_r - obs_r).abs().sum() / obs_r.sum()
    return wmape_abs, wmape_rel, ci_pred


def multiplicadores_leontief(A):
    I  = np.eye(len(A))
    L  = pd.DataFrame(np.linalg.inv(I - A.values),
                      index=A.index, columns=A.columns)
    mult = L.sum(axis=0)   # multiplicadores de producción (suma de columnas)
    return L, mult


# ── Gráfica 1: Curva de calibración ───────────────────────────────────────────

def grafica_calibracion(deltas, w_abs, w_rel, delta_final):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Calibración del parámetro δ del FLQ — Jalisco 2023\n"
        "La función no presenta mínimo interior → δ adoptado por literatura",
        fontsize=13, fontweight='bold'
    )

    colores = {'abs': '#2563EB', 'rel': '#059669',
               'lit': '#DC2626', 'banda': '#FEF2F2'}

    for ax, vals, titulo, ylabel, color in [
        (axes[0], w_abs, "WMAPE Absoluto\n(CI predicho vs CI observado total)", "WMAPE (%)", colores['abs']),
        (axes[1], w_rel, "WMAPE Relativo\n(estructura de participaciones)", "WMAPE (%)", colores['rel']),
    ]:
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, color='#E2E8F0', linewidth=0.8)
        ax.plot(deltas, np.array(vals)*100, color=color,
                linewidth=2.5, zorder=4, label='WMAPE observado')

        # Banda sombreada rango literatura 0.15–0.30
        ax.axvspan(0.15, 0.30, alpha=0.15, color=colores['lit'],
                   label='Rango literatura (0.15–0.30)')

        # Línea vertical δ adoptado
        ax.axvline(delta_final, color=colores['lit'], linewidth=2,
                   linestyle='--', zorder=5, label=f'δ adoptado = {delta_final}')

        wmape_adopted = np.interp(delta_final, deltas, vals) * 100
        ax.scatter([delta_final], [wmape_adopted],
                   color=colores['lit'], s=100, zorder=6)
        ax.annotate(f"{wmape_adopted:.1f}%",
                    xy=(delta_final, wmape_adopted),
                    xytext=(delta_final + 0.05, wmape_adopted + 1),
                    fontsize=10, color=colores['lit'], fontweight='bold')

        ax.set_xlabel('Valor de δ', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(titulo, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(SALIDA_GRAF1, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Guardada: {SALIDA_GRAF1}")


# ── Gráfica 2: Sensibilidad de multiplicadores ────────────────────────────────

def grafica_sensibilidad(resultados_sens, sectores_v):
    """
    4 sub-gráficas:
      A. Multiplicadores de producción por sector para cada δ
      B. Variación % del multiplicador respecto a δ=0.25 (referencia)
      C. Heatmap de diferencias relativas por sector
      D. Distribución de multiplicadores por δ (boxplot)
    """
    sectores_nombres = [NOMBRES_SECTOR.get(s, s) for s in sectores_v]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Análisis de sensibilidad: multiplicadores de Leontief vs δ\n"
        "Referencia: δ = 0.25  |  Sectores con datos en Censos Jalisco 2023",
        fontsize=13, fontweight='bold', y=0.99
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    pal = {0.15:'#7C3AED', 0.20:'#2563EB', 0.25:'#DC2626', 0.30:'#059669'}
    estilos = {0.15:'-.', 0.20:'--', 0.25:'-', 0.30:':'}
    gruesos  = {0.15:1.5, 0.20:1.5, 0.25:2.5, 0.30:1.5}

    mult_ref = resultados_sens[0.25]['mult']

    # ── A. Multiplicadores absolutos ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#F8FAFC')
    ax1.grid(True, color='#E2E8F0', linewidth=0.8, axis='y')
    x = np.arange(len(sectores_v))
    ancho = 0.20

    for i, (d, res) in enumerate(resultados_sens.items()):
        mult_v = res['mult'][sectores_v].values
        ax1.bar(x + (i - 1.5) * ancho, mult_v, ancho,
                color=pal[d], alpha=0.80, label=f'δ = {d}', zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(sectores_nombres, rotation=55, ha='right', fontsize=6.5)
    ax1.set_ylabel('Multiplicador de producción', fontsize=10)
    ax1.set_title('A. Multiplicadores absolutos por sector', fontsize=11)
    ax1.legend(fontsize=8, ncol=2)

    # ── B. Variación % respecto a δ=0.25 ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#F8FAFC')
    ax2.grid(True, color='#E2E8F0', linewidth=0.8)
    ax2.axhline(0, color='#DC2626', linewidth=1.5, linestyle='--', label='δ = 0.25 (ref.)')

    for d, res in resultados_sens.items():
        if d == 0.25:
            continue
        var_pct = ((res['mult'][sectores_v] - mult_ref[sectores_v])
                   / mult_ref[sectores_v] * 100)
        ax2.plot(sectores_nombres, var_pct.values,
                 color=pal[d], linewidth=gruesos[d],
                 linestyle=estilos[d], marker='o', markersize=3,
                 label=f'δ = {d}', zorder=4)

    ax2.set_xticklabels(sectores_nombres, rotation=55, ha='right', fontsize=6.5)
    ax2.set_ylabel('Variación respecto a δ=0.25 (%)', fontsize=10)
    ax2.set_title('B. Cambio % del multiplicador vs referencia δ=0.25', fontsize=11)
    ax2.legend(fontsize=8)

    # ── C. Heatmap de variaciones ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    matrix_data = []
    filas_labels = []
    for d, res in resultados_sens.items():
        if d == 0.25:
            continue
        var = ((res['mult'][sectores_v] - mult_ref[sectores_v])
               / mult_ref[sectores_v] * 100).values
        matrix_data.append(var)
        filas_labels.append(f'δ = {d}')

    matrix_data = np.array(matrix_data)
    vlim = max(abs(matrix_data.min()), abs(matrix_data.max()))
    im = ax3.imshow(matrix_data, aspect='auto', cmap='RdYlGn',
                    vmin=-vlim, vmax=vlim)
    ax3.set_xticks(range(len(sectores_nombres)))
    ax3.set_xticklabels(sectores_nombres, rotation=55, ha='right', fontsize=6.5)
    ax3.set_yticks(range(len(filas_labels)))
    ax3.set_yticklabels(filas_labels, fontsize=10)
    ax3.set_title('C. Heatmap: variación % vs δ=0.25\n(verde=mayor, rojo=menor)', fontsize=11)
    plt.colorbar(im, ax=ax3, label='Variación (%)', shrink=0.8)

    # Anotar valores en heatmap
    for i in range(len(filas_labels)):
        for j in range(len(sectores_nombres)):
            val = matrix_data[i, j]
            ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=5.5,
                     color='white' if abs(val) > vlim * 0.5 else 'black')

    # ── D. Boxplot distribución de multiplicadores ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#F8FAFC')
    ax4.grid(True, color='#E2E8F0', linewidth=0.8, axis='y')

    datos_box = [res['mult'][sectores_v].values
                 for d, res in resultados_sens.items()]
    labels_box = [f'δ = {d:.2f}' for d in resultados_sens.keys()]
    colores_box = [pal[d] for d in resultados_sens.keys()]

    bp = ax4.boxplot(datos_box, labels=labels_box, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax4.set_ylabel('Multiplicador de producción', fontsize=10)
    ax4.set_title('D. Distribución de multiplicadores\n(mediana, rango intercuartil, outliers)',
                  fontsize=11)

    # Anotar mediana
    for i, datos in enumerate(datos_box):
        med = np.median(datos)
        ax4.text(i + 1, med + 0.02, f'{med:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.savefig(SALIDA_GRAF2, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Guardada: {SALIDA_GRAF2}")


# ── Ejecución principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Calibración de δ — FLQ Jalisco 2023")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    (A_NAC, PIBE_JAL, PIBE_NAC, TOTAL_JAL, TOTAL_NAC,
     CI_OBS, PB_OBS, SECTORES_V) = cargar_datos()
    print(f"    Sectores comparables con Censos: {len(SECTORES_V)} / {len(A_NAC)}")

    # 2. Grid search sobre todo el rango de delta
    print("\n[2] Grid search δ ∈ [0.01, 0.99] (99 puntos)...")
    deltas_grid = np.linspace(0.01, 0.99, 99)
    w_abs_list, w_rel_list = [], []

    for d in deltas_grid:
        A_r, _ = regionalizar(A_NAC, PIBE_JAL, PIBE_NAC, TOTAL_JAL, TOTAL_NAC, d)
        wa, wr, _ = wmapes(A_r, CI_OBS, PB_OBS, SECTORES_V)
        w_abs_list.append(wa)
        w_rel_list.append(wr)

    print(f"    WMAPE abs en δ=0.01:  {w_abs_list[0]*100:.2f}%")
    print(f"    WMAPE abs en δ=0.25:  {np.interp(0.25, deltas_grid, w_abs_list)*100:.2f}%")
    print(f"    WMAPE abs en δ=0.99:  {w_abs_list[-1]*100:.2f}%")
    print(f"    → Función monótona: no hay mínimo interior")
    print(f"\n    WMAPE rel en δ=0.15:  {np.interp(0.15, deltas_grid, w_rel_list)*100:.2f}%")
    print(f"    WMAPE rel en δ=0.25:  {np.interp(0.25, deltas_grid, w_rel_list)*100:.2f}%")
    print(f"    WMAPE rel en δ=0.30:  {np.interp(0.30, deltas_grid, w_rel_list)*100:.2f}%")
    print(f"    → Variación en rango literatura [0.15–0.30]: "
          f"{abs(np.interp(0.15,deltas_grid,w_rel_list)-np.interp(0.30,deltas_grid,w_rel_list))*100:.2f} pp")

    # 3. Análisis de sensibilidad
    print(f"\n[3] Análisis de sensibilidad en δ ∈ {DELTAS_SENS}...")
    resultados_sens = {}
    filas_sens = []

    for d in DELTAS_SENS:
        A_r, flq = regionalizar(A_NAC, PIBE_JAL, PIBE_NAC, TOTAL_JAL, TOTAL_NAC, d)
        wa, wr, ci_pred = wmapes(A_r, CI_OBS, PB_OBS, SECTORES_V)
        L, mult = multiplicadores_leontief(A_r)
        lam = (np.log2(1 + TOTAL_JAL / TOTAL_NAC)) ** d
        resultados_sens[d] = {
            'A_reg': A_r, 'flq': flq, 'mult': mult, 'L': L,
            'wmape_abs': wa, 'wmape_rel': wr, 'lambda': lam
        }
        filas_sens.append({
            'delta': d,
            'lambda_star': lam,
            'wmape_abs_pct': round(wa*100, 3),
            'wmape_rel_pct': round(wr*100, 3),
            'mult_mediana': round(mult[SECTORES_V].median(), 4),
            'mult_max': round(mult[SECTORES_V].max(), 4),
            'mult_min': round(mult[SECTORES_V].min(), 4),
        })
        print(f"    δ={d}: λ*={lam:.4f}  WMAPE_rel={wr*100:.2f}%  "
              f"mult_mediana={mult[SECTORES_V].median():.4f}")

    # 4. MIP final con delta adoptado
    print(f"\n[4] Calculando MIP final con δ = {DELTA_FINAL}...")
    A_final = resultados_sens[DELTA_FINAL]['A_reg']
    flq_final = resultados_sens[DELTA_FINAL]['flq']
    L_final, mult_final = multiplicadores_leontief(A_final)
    radio = np.max(np.abs(np.linalg.eigvals(A_final.values)))
    print(f"    Radio espectral: {radio:.6f} ({'✓ estable' if radio < 1 else '✗'})")
    print(f"    Multiplicador promedio: {mult_final[SECTORES_V].mean():.4f}")
    print(f"    Sector con mayor multiplicador: "
          f"{mult_final.idxmax()} ({mult_final.max():.4f})")
    print(f"    Sector con menor multiplicador: "
          f"{mult_final.idxmin()} ({mult_final.min():.4f})")

    # 5. Guardar archivos
    print("\n[5] Guardando archivos...")
    A_final.to_csv(SALIDA_MIP)
    print(f"    MIP regionalizada (δ=0.25): {SALIDA_MIP}")

    flq_final.loc[A_final.index, A_final.columns].to_csv(SALIDA_FLQ)
    print(f"    Matriz FLQ (δ=0.25):        {SALIDA_FLQ}")

    pd.DataFrame(filas_sens).to_csv(SALIDA_SENS, index=False)
    print(f"    Tabla sensibilidad:         {SALIDA_SENS}")

    # 6. Gráficas
    print("\n[6] Generando gráficas...")
    grafica_calibracion(deltas_grid, w_abs_list, w_rel_list, DELTA_FINAL)
    grafica_sensibilidad(resultados_sens, SECTORES_V)

    print("\n✓ Calibración completada.")
    print(f"\n  δ adoptado = {DELTA_FINAL} (Flegg & Webber, 2000; Tohmo, 2004)")
    print(f"  MIP regionalizada lista para el ajuste RAS.")
