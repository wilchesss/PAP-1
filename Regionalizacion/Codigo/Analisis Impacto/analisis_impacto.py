"""
Análisis de Impacto Económico — MIP Jalisco 2023
Calcula los tres tipos de multiplicadores de Leontief y simula escenarios
de choque de demanda final para sectores estratégicos de Jalisco.

Multiplicadores calculados:
  1. Producción  : pesos de producción total por peso de demanda adicional
  2. Empleo      : empleos generados por millón de pesos de demanda adicional
  3. Valor agr.  : pesos de valor agregado por peso de demanda adicional

Escenarios de simulación (choque de +10,000 MDP en demanda final):
  A. Sector electrónico y maquinaria (333-336) — inversión extranjera directa
  B. Industria alimentaria (311)               — expansión agroindustrial
  C. Construcción (23)                         — obra pública / infraestructura
  D. Servicios profesionales (54)              — economía del conocimiento

Referencia:
  Miller, R. & Blair, P. (2009). Input-Output Analysis. Cambridge, Cap. 6.
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

ARCHIVO_MIP    = f"{DIR_OUT}/mip_jalisco_final.csv"
ARCHIVO_CENSO  = f"{DIR_IN}/tr_ce_jal_2024.csv"

SALIDA_MULT    = f"{DIR_OUT}/multiplicadores_completos.csv"
SALIDA_ESCEN   = f"{DIR_OUT}/escenarios_impacto.csv"
SALIDA_GRAF1   = f"{DIR_OUT}/grafica_multiplicadores.png"
SALIDA_GRAF2   = f"{DIR_OUT}/grafica_escenarios.png"

CHOQUE_MDP     = 10_000   # millones de pesos de choque de demanda

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
    "111":"Agricultura","112":"Ganadería","113-115":"Forestal/Serv.agrop.",
    "114":"Pesca","21-1":"Min. petrolera","21-2":"Min. no petrolera",
    "22":"Energía/agua/gas","23":"Construcción","311":"Alimentos",
    "312":"Bebidas y tabaco","313-314":"Textiles","315-316":"Prendas/cuero",
    "321":"Madera","322-323":"Papel e impresión","324-326":"Petroquímica/química",
    "327":"Min. no metálicos","331-332":"Metálicas/prod. metálicos",
    "333-336":"Maquinaria/electrónica","337":"Muebles","339":"Otras manuf.",
    "43":"Com. mayorista","46":"Com. minorista","48-49":"Transportes/almac.",
    "51":"Información","52":"Financiero y seguros","53":"Inmobiliario/alquiler",
    "54":"Serv. profesionales","55":"Corporativos","56":"Serv. apoyo negocios",
    "61":"Educación","62":"Salud y asistencia","71":"Esparcimiento",
    "72":"Alojamiento/alimentos","81":"Otros servicios",
}

ESCENARIOS = {
    "333-336": ("Maquinaria y electrónica",
                "Simula una inversión extranjera directa en manufactura\n"
                "electrónica (Parque Industrial de Guadalajara)."),
    "311":     ("Industria alimentaria",
                "Simula expansión agroindustrial aprovechando la\n"
                "especialización de Jalisco en alimentos (SLQ = 1.39)."),
    "23":      ("Construcción",
                "Simula un programa de obra pública o infraestructura\n"
                "equivalente a 10,000 MDP en demanda de construcción."),
    "54":      ("Servicios profesionales",
                "Simula el crecimiento del sector de economía del\n"
                "conocimiento y servicios de alto valor agregado."),
}

# ── Funciones ──────────────────────────────────────────────────────────────────

def cargar_censos_intensidades():
    """
    Extrae por sector los totales de:
      - H001A: personal ocupado
      - A131A: valor agregado censal bruto (MDP)
      - A111A: producción bruta total (MDP)
    y calcula coeficientes de intensidad laboral y de valor agregado.
    """
    censo = pd.read_csv(ARCHIVO_CENSO, encoding='utf-8-sig', low_memory=False)
    censo['COD'] = censo['CODIGO'].astype(str).str.strip()

    sub = censo[censo['COD'].str.len() == 3]
    tot = sub[(sub['E04'].isna()) & (sub['ID_ESTRATO'].isna())].copy()

    for col in ['H001A', 'A131A', 'A111A']:
        tot[col] = pd.to_numeric(tot[col], errors='coerce')

    emp_obs, vab_obs, pb_obs = {}, {}, {}
    for _, row in tot.iterrows():
        cod = row['COD']
        if cod in MAPEO_CENSO:
            g = MAPEO_CENSO[cod]
            if pd.notna(row['H001A']):
                emp_obs[g] = emp_obs.get(g, 0) + row['H001A']
            if pd.notna(row['A131A']):
                vab_obs[g] = vab_obs.get(g, 0) + row['A131A']
            if pd.notna(row['A111A']):
                pb_obs[g]  = pb_obs.get(g, 0)  + row['A111A']

    emp = pd.Series(emp_obs)
    vab = pd.Series(vab_obs)
    pb  = pd.Series(pb_obs)

    # Intensidad laboral: empleos por millón de pesos producidos
    # (empleos / MDP = empleos/MDP → multiplicador en empleos por MDP de choque)
    intens_lab = (emp / pb).fillna(0)

    # Coeficiente de valor agregado: VAB por peso producido
    coef_va = (vab / pb).fillna(0).clip(0, 1)

    return emp, vab, pb, intens_lab, coef_va


def calcular_multiplicadores(A, intens_lab, coef_va):
    """
    Calcula la inversa de Leontief y los tres tipos de multiplicadores.

    Multiplicador de producción (tipo II simple):
        mult_prod_j = sum_i L_ij
        = producción total generada por 1 MDP de demanda en j

    Multiplicador de empleo:
        mult_emp_j = sum_i (intens_lab_i × L_ij)
        = empleos generados por 1 MDP de demanda en j

    Multiplicador de valor agregado:
        mult_va_j = sum_i (coef_va_i × L_ij)
        = MDP de valor agregado generados por 1 MDP de demanda en j
    """
    I = np.eye(len(A))
    L = pd.DataFrame(
        np.linalg.inv(I - A.values),
        index=A.index, columns=A.columns
    )

    mult_prod = L.sum(axis=0)

    # Alinear vectores de intensidad con el índice de L
    sectores = L.index.tolist()
    il = intens_lab.reindex(sectores).fillna(0)
    cv = coef_va.reindex(sectores).fillna(0)

    mult_emp = L.mul(il, axis=0).sum(axis=0)
    mult_va  = L.mul(cv, axis=0).sum(axis=0)

    return L, mult_prod, mult_emp, mult_va


def simular_escenario(L, mult_prod, mult_emp, mult_va,
                      sector, choque_mdp, intens_lab, coef_va):
    """
    Dado un choque de demanda final en un sector específico,
    calcula el impacto total sobre producción, empleo y valor agregado.

    Impacto = columna j de L × choque_mdp
    """
    if sector not in L.columns:
        return None

    # Vector de impacto en producción por sector
    impacto_prod_sector = L[sector] * choque_mdp

    total_prod  = mult_prod[sector] * choque_mdp
    total_emp   = mult_emp[sector]  * choque_mdp
    total_va    = mult_va[sector]   * choque_mdp

    # Top 5 sectores más beneficiados
    top5 = impacto_prod_sector.sort_values(ascending=False).head(5)

    return {
        'sector':        sector,
        'choque_mdp':    choque_mdp,
        'prod_total':    total_prod,
        'emp_generados': total_emp,
        'va_generado':   total_va,
        'ratio_prod':    total_prod / choque_mdp,
        'top5_sectores': top5,
        'impacto_completo': impacto_prod_sector,
    }


# ── Gráficas ───────────────────────────────────────────────────────────────────

def grafica_multiplicadores(mult_prod, mult_emp, mult_va, sectores_validos):
    """
    3 paneles horizontales con los multiplicadores por sector,
    ordenados de mayor a menor por multiplicador de producción.
    """
    orden = mult_prod[sectores_validos].sort_values(ascending=False).index
    nombres = [NOMBRES.get(s, s) for s in orden]
    n = len(orden)
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(
        "Multiplicadores de Leontief — MIP Jalisco 2023\n"
        "Efecto total (directo + indirecto) ante un aumento de 1 MDP en la demanda final",
        fontsize=13, fontweight='bold'
    )

    C = ['#2563EB', '#059669', '#7C3AED']
    titulos = [
        ('A. Multiplicador de Producción',
         'Pesos de producción total generados\npor cada peso de demanda adicional'),
        ('B. Multiplicador de Empleo',
         'Empleos generados\npor cada millón de pesos de demanda adicional'),
        ('C. Multiplicador de Valor Agregado',
         'Pesos de valor agregado generados\npor cada peso de demanda adicional'),
    ]
    datos = [mult_prod[orden], mult_emp[orden], mult_va[orden]]

    for ax, dat, col, (tit, ytit) in zip(axes, datos, C, titulos):
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, color='#E2E8F0', linewidth=0.8, axis='y')

        barras = ax.bar(x, dat.values, color=col, alpha=0.85, zorder=3)
        ax.axhline(dat.mean(), color='black', linewidth=1.5,
                   linestyle='--', alpha=0.6,
                   label=f'Promedio = {dat.mean():.3f}')

        # Anotar los 3 mayores
        top3_idx = dat.values.argsort()[::-1][:3]
        for i in top3_idx:
            ax.text(i, dat.values[i] + dat.values.max()*0.01,
                    f'{dat.values[i]:.3f}',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color=col)

        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=7.5)
        ax.set_ylabel(ytit, fontsize=9)
        ax.set_title(tit, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(SALIDA_GRAF1, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Guardada: {SALIDA_GRAF1}")


def grafica_escenarios(resultados_esc, L, sectores_validos):
    """
    Para cada escenario:
      - Barra de impacto total (producción, VA, empleo)
      - Distribución del impacto en producción por sector (top 10)
    """
    n_esc = len(resultados_esc)
    fig = plt.figure(figsize=(18, 5 * n_esc))
    fig.suptitle(
        f"Escenarios de impacto económico — Choque de {CHOQUE_MDP:,} MDP\n"
        "MIP Jalisco 2023 | Análisis de encadenamientos hacia atrás",
        fontsize=13, fontweight='bold', y=1.01
    )

    C_ESC = ['#2563EB', '#059669', '#DC2626', '#7C3AED']
    gs = gridspec.GridSpec(n_esc, 2, figure=fig,
                           hspace=0.55, wspace=0.35)

    for idx, (sector, res) in enumerate(resultados_esc.items()):
        nombre, desc = ESCENARIOS[sector]
        color = C_ESC[idx]

        # ── Panel izquierdo: resumen de impacto ────────────────────────────────
        ax_l = fig.add_subplot(gs[idx, 0])
        ax_l.set_facecolor('#F8FAFC')
        ax_l.grid(True, color='#E2E8F0', linewidth=0.8, axis='x')

        metricas = ['Producción\ntotal (MDP)',
                    'Valor agregado\n(MDP)',
                    'Empleos\ngenerados']
        valores  = [res['prod_total'], res['va_generado'], res['emp_generados']]
        colores_m = [color, '#059669', '#F59E0B']

        barras = ax_l.barh(metricas, valores, color=colores_m, alpha=0.85, zorder=3)
        for bar, val in zip(barras, valores):
            fmt = f'{val:,.0f}' if val > 100 else f'{val:.1f}'
            ax_l.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                      fmt, va='center', fontsize=10, fontweight='bold')

        ax_l.set_title(
            f"Escenario {chr(65+idx)}: {nombre}\n"
            f"Multiplicador de producción = {res['ratio_prod']:.4f}",
            fontsize=10, fontweight='bold'
        )
        ax_l.set_xlabel('Magnitud del impacto', fontsize=9)
        ax_l.text(0.5, -0.18, desc, transform=ax_l.transAxes,
                  ha='center', fontsize=8, color='#475569', style='italic')

        # ── Panel derecho: distribución del impacto por sector (top 10) ────────
        ax_r = fig.add_subplot(gs[idx, 1])
        ax_r.set_facecolor('#F8FAFC')
        ax_r.grid(True, color='#E2E8F0', linewidth=0.8, axis='x')

        top10 = res['impacto_completo'].sort_values(ascending=False).head(10)
        nombres_top = [NOMBRES.get(s, s) for s in top10.index]
        y_pos = np.arange(len(top10))

        colores_bar = [color if s == sector else '#94A3B8' for s in top10.index]
        ax_r.barh(y_pos, top10.values, color=colores_bar, alpha=0.85, zorder=3)

        for i, (s, val) in enumerate(zip(top10.index, top10.values)):
            ax_r.text(val * 1.01, i, f'{val:,.0f}',
                      va='center', fontsize=8)

        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(nombres_top, fontsize=8)
        ax_r.invert_yaxis()
        ax_r.set_title(
            f"Top 10 sectores más beneficiados\n(MDP de producción adicional)",
            fontsize=10, fontweight='bold'
        )
        ax_r.set_xlabel('Producción adicional (MDP)', fontsize=9)

    plt.savefig(SALIDA_GRAF2, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Guardada: {SALIDA_GRAF2}")


# ── Ejecución principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Análisis de Impacto Económico — MIP Jalisco 2023")
    print("=" * 60)

    # 1. Cargar MIP final
    print("\n[1] Cargando MIP final de Jalisco...")
    A = pd.read_csv(ARCHIVO_MIP, index_col=0)
    print(f"  Dimensión: {A.shape}")

    # 2. Cargar intensidades desde Censos
    print("\n[2] Calculando intensidades sectoriales desde Censos...")
    emp, vab, pb, intens_lab, coef_va = cargar_censos_intensidades()

    sectores_validos = [s for s in A.index if s in pb.index and pb.get(s, 0) > 0]
    print(f"  Sectores con datos de empleo y VA: {len(sectores_validos)} / {len(A)}")
    print(f"  Empleo total cubierto:  {emp[sectores_validos].sum():>12,.0f} personas")
    print(f"  VAB total cubierto:     {vab[sectores_validos].sum():>12,.1f} MDP")
    print(f"  PB total cubierta:      {pb[sectores_validos].sum():>12,.1f} MDP")

    # 3. Calcular multiplicadores
    print("\n[3] Calculando multiplicadores de Leontief...")
    L, mult_prod, mult_emp, mult_va = calcular_multiplicadores(
        A, intens_lab, coef_va
    )

    print(f"\n  {'Sector':<14} {'Nombre':<30} {'Prod':>7} {'Empleo':>9} {'VA':>7}")
    print(f"  {'-'*14} {'-'*30} {'-'*7} {'-'*9} {'-'*7}")
    orden_prod = mult_prod[sectores_validos].sort_values(ascending=False)
    for s in orden_prod.index:
        print(f"  {s:<14} {NOMBRES.get(s,s):<30} "
              f"{mult_prod[s]:>7.4f} {mult_emp[s]:>9.4f} {mult_va[s]:>7.4f}")

    # 4. Guardar tabla completa de multiplicadores
    mult_df = pd.DataFrame({
        'sector':              mult_prod.index,
        'nombre':              [NOMBRES.get(s, s) for s in mult_prod.index],
        'mult_produccion':     mult_prod.values,
        'mult_empleo':         mult_emp.values,
        'mult_valor_agregado': mult_va.values,
        'intensidad_laboral':  intens_lab.reindex(mult_prod.index).fillna(0).values,
        'coef_valor_agregado': coef_va.reindex(mult_prod.index).fillna(0).values,
    }).sort_values('mult_produccion', ascending=False)
    mult_df.to_csv(SALIDA_MULT, index=False)
    print(f"\n  Tabla guardada: {SALIDA_MULT}")

    # 5. Simular escenarios
    print(f"\n[4] Simulando escenarios de choque (+{CHOQUE_MDP:,} MDP)...")
    resultados_esc = {}
    filas_escen    = []

    for sector, (nombre, _) in ESCENARIOS.items():
        if sector not in A.index:
            print(f"  ADVERTENCIA: sector {sector} no está en la MIP. Omitido.")
            continue
        res = simular_escenario(
            L, mult_prod, mult_emp, mult_va,
            sector, CHOQUE_MDP, intens_lab, coef_va
        )
        resultados_esc[sector] = res

        print(f"\n  Escenario: {nombre} ({sector})")
        print(f"    Choque:              {CHOQUE_MDP:>10,.0f} MDP")
        print(f"    Producción total:    {res['prod_total']:>10,.1f} MDP "
              f"(mult = {res['ratio_prod']:.4f})")
        print(f"    Empleos generados:   {res['emp_generados']:>10,.0f} empleos")
        print(f"    Valor agregado:      {res['va_generado']:>10,.1f} MDP")
        print(f"    Top 3 sectores beneficiados:")
        for s, v in res['top5_sectores'].head(3).items():
            print(f"      {s:<12} {NOMBRES.get(s,s):<30} {v:>8,.1f} MDP")

        filas_escen.append({
            'sector':           sector,
            'nombre':           nombre,
            'choque_mdp':       CHOQUE_MDP,
            'produccion_total': round(res['prod_total'], 2),
            'mult_produccion':  round(res['ratio_prod'], 4),
            'empleos_gen':      round(res['emp_generados'], 0),
            'va_generado':      round(res['va_generado'], 2),
        })

    pd.DataFrame(filas_escen).to_csv(SALIDA_ESCEN, index=False)
    print(f"\n  Tabla escenarios guardada: {SALIDA_ESCEN}")

    # 6. Gráficas
    print("\n[5] Generando gráficas...")
    grafica_multiplicadores(mult_prod, mult_emp, mult_va, sectores_validos)
    grafica_escenarios(resultados_esc, L, sectores_validos)

    print("\n✓ Análisis de impacto completado.")
    print(f"\n  Resumen ejecutivo:")
    print(f"  {'Sector':<14} {'Mult.Prod':>10} {'Empleos/MDP':>12} {'Mult.VA':>9}")
    print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*9}")
    for row in sorted(filas_escen, key=lambda x: -x['mult_produccion']):
        emp_por_mdp = row['empleos_gen'] / row['choque_mdp']
        print(f"  {row['sector']:<14} {row['mult_produccion']:>10.4f} "
              f"{emp_por_mdp:>12.4f} {row['va_generado']/row['choque_mdp']:>9.4f}")
