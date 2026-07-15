"""
Mollier h-x Diagramm - Web Version
Bauphysik | Feuchteschutz & Schimmelanalyse
Made by Mahmoud Alhayek | Wetzel & von Seht
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import streamlit as st

# ══════════════════════════════════════════════════════
#  SEITEN-KONFIGURATION
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mollier h-x Diagramm | Bauphysik",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  PHYSIK
# ══════════════════════════════════════════════════════
def p_sat(T_C):
    T_C = np.asarray(T_C, dtype=float)
    c = np.where(T_C >= 0, 237.300, 265.500)
    d = np.where(T_C >= 0, 17.269, 21.875)
    return 610.5 * np.exp(d * T_C / (c + T_C))

def x_from_T_RF(T_C, RF, p_pa=101325.0):
    p_ws = float(p_sat(T_C))
    p_w = (RF / 100.0) * p_ws
    d = p_pa - p_w
    return 0.62198 * p_w / d * 1000.0 if d > 0 else 0.0

def enthalpy_kJ(T_C, x_gkg):
    x_k = x_gkg / 1000.0
    return 1.006 * T_C + x_k * (2501.0 + 1.86 * T_C)

def dew_point(T_C, RF):
    c = 237.3 if T_C >= 0 else 265.5
    d = 17.269 if T_C >= 0 else 21.875
    g = np.log(max(RF, 0.01) / 100.0) + d * T_C / (c + T_C)
    return c * g / (d - g)

def critical_surface_temp(T_r, RF_i, p_pa=101325.0, rf_limit=80.0):
    p_w_i = (RF_i / 100.0) * float(p_sat(T_r))
    T_s = float(T_r)
    for _ in range(8000):
        if p_w_i / float(p_sat(T_s)) * 100.0 >= rf_limit:
            break
        T_s -= 0.01
    return T_s

def calc_f_rsi_min(T_surf, T_i, T_e):
    delta = T_i - T_e
    if abs(delta) < 0.01:
        return 1.0
    return (T_surf - T_e) / delta

def max_rf_for_surface_temp(T_surf, T_r, rf_limit=80.0):
    p_sat_surf = float(p_sat(T_surf))
    p_sat_room = float(p_sat(T_r))
    p_w_max = (rf_limit / 100.0) * p_sat_surf
    return min(p_w_max / p_sat_room * 100.0, 100.0)

# ══════════════════════════════════════════════════════
#  NORM
# ══════════════════════════════════════════════════════
NORM_T_I  = 20.0
NORM_T_E  = -5.0
NORM_RF_I = 50.0
NORM_FRSI = 0.70

def norm_abweichung(T_i, T_e, RF_i):
    abw = []
    if abs(T_i - NORM_T_I) > 0.5:
        abw.append(f"T_Innen = {T_i:.1f} °C  (Norm: {NORM_T_I:.0f} °C)")
    if abs(T_e - NORM_T_E) > 0.5:
        abw.append(f"T_Aussen = {T_e:.1f} °C  (Norm: {NORM_T_E:.0f} °C)")
    if abs(RF_i - NORM_RF_I) > 2.0:
        abw.append(f"phi_Innen = {RF_i:.0f} %  (Norm: {NORM_RF_I:.0f} %)")
    return abw

# ══════════════════════════════════════════════════════
#  FARBEN
# ══════════════════════════════════════════════════════
C = {
    "sat":    "#C0392B",
    "aussen": "#1A5276",
    "innen":  "#7B241C",
    "aufhz":  "#1A7A42",
    "p_auf":  "#27AE60",
    "p_feu":  "#6C3483",
    "tkrit":  "#CA6F1E",
    "tau":    "#0E6655",
    "grid":   "#E8E8E8",
    "bg":     "#FAFAFA",
}

# ══════════════════════════════════════════════════════
#  DIAGRAMM
# ══════════════════════════════════════════════════════
def draw_mollier(T_a, RF_a, T_r, RF_i, p_pa=101325.0):
    fig, ax = plt.subplots(figsize=(13, 8), dpi=120)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("#FFFFFF")

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("#AAAAAA")

    x_a    = x_from_T_RF(T_a, RF_a, p_pa)
    x_i    = x_from_T_RF(T_r, RF_i, p_pa)
    h_a    = enthalpy_kJ(T_a, x_a)
    h_i    = enthalpy_kJ(T_r, x_i)
    T_tau  = dew_point(T_r, RF_i)
    T_surf = critical_surface_temp(T_r, RF_i, p_pa)
    p_w_a  = (RF_a / 100.0) * float(p_sat(T_a))
    rf_auf = p_w_a / float(p_sat(T_r)) * 100.0

    x_right = max(max(x_i, x_a) * 1.35 + 1.5, 14.0)
    X_MAX   = min(x_right, 30.0)
    T_plot  = np.linspace(-20, 50, 900)
    x_plot  = np.linspace(0, X_MAX, 900)

    # Enthalpie-Isolinien
    for h_val in np.arange(-20, 130, 10):
        T_h = []
        for xg in x_plot:
            xk = xg / 1000.0
            dh = 1.006 + 1.86 * xk
            T_h.append((h_val - 2501.0 * xk) / dh if dh else np.nan)
        T_h  = np.array(T_h)
        mask = (T_h >= -21) & (T_h <= 51)
        if mask.sum() > 1:
            ax.plot(x_plot[mask], T_h[mask],
                    color="#B2DFDB", lw=0.55, alpha=0.5, zorder=1)

    # RF-Isolinien
    rf_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    blue_cmap = matplotlib.colormaps["Blues"]
    rf_colors = [
        matplotlib.colors.to_hex(
            blue_cmap(0.20 + 0.75 * i / (len(rf_levels) - 1))
        )
        for i in range(len(rf_levels))
    ]
    rf_colors[-1] = C["sat"]

    for rf_val, col in zip(rf_levels, rf_colors):
        x_vals = []
        for T in T_plot:
            pws = float(p_sat(T))
            pw  = (rf_val / 100.0) * pws
            den = p_pa - pw
            x_vals.append(0.62198 * pw / den * 1000.0 if den > 0 else np.nan)
        x_arr = np.array(x_vals)
        vis   = np.isfinite(x_arr) & (x_arr <= X_MAX * 1.02)
        if vis.sum() > 1:
            lw = 2.2 if rf_val == 100 else (1.1 if rf_val % 20 == 0 else 0.6)
            ax.plot(x_arr[vis], T_plot[vis],
                    color=col, lw=lw, alpha=0.80, zorder=2)
            for tgt in [32, 28, 24, 20, 15]:
                idx = np.argmin(np.abs(T_plot - tgt))
                lx, lT = x_arr[idx], T_plot[idx]
                if np.isfinite(lx) and 0.3 < lx < X_MAX - 0.5:
                    ax.text(lx + 0.15, lT, f"{rf_val}%",
                            fontsize=7, color=col,
                            fontweight="bold" if rf_val == 100 else "normal",
                            va="center", ha="left", clip_on=True, zorder=6)
                    break

    # Saettigungslinie
    x_sat = np.array([
        0.62198 * float(p_sat(T)) / (p_pa - float(p_sat(T))) * 1000.0
        for T in T_plot
    ])
    vis = np.isfinite(x_sat) & (x_sat <= X_MAX * 1.05)
    ax.plot(x_sat[vis], T_plot[vis], color=C["sat"], lw=2.5, zorder=4)
    ax.fill_betweenx(T_plot[vis], x_sat[vis], X_MAX,
                     alpha=0.04, color=C["sat"], zorder=1)

    # T_krit Linie
    ax.axhline(T_surf, color=C["tkrit"], ls=(0, (6, 3)),
               lw=2.2, alpha=0.92, zorder=5)
    ax.fill_between([0, X_MAX], [-20, -20], [T_surf, T_surf],
                    alpha=0.04, color=C["tkrit"], zorder=1)

    # Pfeile
    ak = dict(arrowstyle="-|>", mutation_scale=14, lw=2.2)
    ax.annotate("", xy=(x_a, T_r), xytext=(x_a, T_a),
                arrowprops=dict(**ak, color=C["p_auf"]), zorder=8)
    if x_i > x_a + 0.08:
        ax.annotate("", xy=(x_i, T_r), xytext=(x_a, T_r),
                    arrowprops=dict(**ak, color=C["p_feu"]), zorder=8)
    ax.annotate("", xy=(x_i, T_tau), xytext=(x_i, T_r),
                arrowprops=dict(
                    arrowstyle="-|>", mutation_scale=10,
                    lw=1.4, color=C["tau"],
                    linestyle=(0, (4, 2))
                ), zorder=8)

    # Punkte
    pk = dict(zorder=10, clip_on=False)
    ms = 130
    ax.scatter([x_a], [T_a],    color=C["aussen"], s=ms,
               marker="o", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_a], [T_r],    color=C["aufhz"],  s=ms-30,
               marker="D", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_r],    color=C["innen"],  s=ms,
               marker="o", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_tau],  color=C["tau"],    s=ms-30,
               marker="^", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_surf], color=C["tkrit"],  s=ms-30,
               marker="s", edgecolors="white", linewidths=1.5, **pk)

    # Labels
    def _lbl(x, y, txt, col, ha="left", va="bottom", dx=0.0, dy=0.0):
        ax.text(x + dx, y + dy, txt,
                fontsize=8.5, color=col, ha=ha, va=va,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="white", ec=col, alpha=0.93, lw=1.1),
                zorder=12, clip_on=True)

    dx_r, dx_l = 0.35, -0.35
    _lbl(x_a, T_a,
         "① Aussenluft\nT = " + f"{T_a:.1f}" + " °C   phi = " + f"{RF_a:.0f}" + " %\n"
         "x = " + f"{x_a:.2f}" + " g/kg   h = " + f"{h_a:.1f}" + " kJ/kg",
         C["aussen"], ha="right", va="top", dx=dx_l, dy=-0.5)

    _lbl(x_a, T_r,
         "② Aufgeheizt\nT = " + f"{T_r:.1f}" + " °C   phi = " + f"{rf_auf:.1f}" + " %\n"
         "x = " + f"{x_a:.2f}" + " g/kg",
         C["aufhz"], ha="right", va="bottom", dx=dx_l, dy=0.6)

    _lbl(x_i, T_r,
         "③ Innenluft\nT = " + f"{T_r:.1f}" + " °C   phi = " + f"{RF_i:.0f}" + " %\n"
         "x = " + f"{x_i:.2f}" + " g/kg   h = " + f"{h_i:.1f}" + " kJ/kg",
         C["innen"], ha="left", va="bottom", dx=dx_r, dy=0.6)

    if x_i > x_a + 0.08:
        ax.text((x_a + x_i) / 2, T_r + 0.8,
                "Δx = " + f"{x_i - x_a:+.2f}" + " g/kg",
                fontsize=8.5, color=C["p_feu"],
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=C["p_feu"],
                          alpha=0.92, lw=0.9),
                zorder=12, clip_on=True)

    _lbl(x_i, T_tau,
         "④ Taupunkt\nT_tau = " + f"{T_tau:.1f}" + " °C",
         C["tau"], ha="left", va="top", dx=dx_r, dy=-0.5)

    ax.text(0.35, T_surf + 0.7,
            "⑤ T_krit = " + f"{T_surf:.1f}" + " °C   —   phi_Oberflaeche >= 80 % → Schimmelgrenze",
            fontsize=9, color=C["tkrit"],
            ha="left", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec=C["tkrit"],
                      alpha=0.93, lw=1.1),
            zorder=12, clip_on=True)

    # Legende
    handles = [
        mpatches.Patch(color="none", label="--- Zustandspunkte ---"),
        Line2D([0],[0], marker="o", color="w", lw=0,
               markerfacecolor=C["aussen"], markersize=10, label="① Aussenluft"),
        Line2D([0],[0], marker="D", color="w", lw=0,
               markerfacecolor=C["aufhz"],  markersize=9,  label="② Aufgeheizt"),
        Line2D([0],[0], marker="o", color="w", lw=0,
               markerfacecolor=C["innen"],  markersize=10, label="③ Innenluft"),
        Line2D([0],[0], marker="^", color="w", lw=0,
               markerfacecolor=C["tau"],    markersize=9,  label="④ Taupunkt"),
        Line2D([0],[0], marker="s", color="w", lw=0,
               markerfacecolor=C["tkrit"],  markersize=9,  label="⑤ T_krit"),
        mpatches.Patch(color="none", label=" "),
        mpatches.Patch(color="none", label="--- Prozesse ---"),
        Line2D([0],[0], color=C["sat"],   lw=2.5, label="Saettigungslinie"),
        Line2D([0],[0], color=C["p_auf"], lw=2.0, label="① → ② Aufheizung"),
        Line2D([0],[0], color=C["p_feu"], lw=2.0, label="② → ③ Befeuchtung"),
        Line2D([0],[0], color=C["tau"],   lw=1.6, ls="--", label="③ → ④ Abkuehlung"),
        Line2D([0],[0], color=C["tkrit"], lw=2.0, ls="--", label="⑤ Schimmelgrenze"),
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    fontsize=8, framealpha=0.97,
                    ncol=1, title="Legende", title_fontsize=9,
                    edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(1.2)

    # Achsen
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-20, 50)
    ax.set_xlabel("Feuchtegehalt   x   [g/kg trockene Luft]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.set_ylabel("Temperatur   T   [°C]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.tick_params(axis="both", labelsize=9, colors="#555555")
    ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Obere Achse
    ax2 = ax.twiny()
    ax2.set_xlim(0, X_MAX)
    raw_ticks = np.arange(0, 31, 5)
    x_ticks   = raw_ticks[raw_ticks <= X_MAX]
    pd_vals   = [
        xk / 1000.0 / (0.62198 + xk / 1000.0) * p_pa / 100.0
        for xk in x_ticks
    ]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{v:.1f}" for v in pd_vals], fontsize=8.5)
    ax2.set_xlabel("Wasserdampf-Partialdruck   p_D   [hPa]",
                   fontsize=10, labelpad=8, color="#555555")

    ax.set_title(
        "Mollier  h-x-Diagramm  —  Feuchte Luft   (p = 1013 hPa)",
        fontsize=14, fontweight="bold", pad=14, color="#1A252F"
    )
    plt.tight_layout(pad=1.5)

    return fig, dict(
        x_a=x_a, x_i=x_i, h_a=h_a, h_i=h_i,
        T_tau=T_tau, T_surf=T_surf,
        rf_aufgeheizt=rf_auf, delta_x=x_i - x_a
    )

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
.main { background-color: #F0F2F5; }
.block-container { padding-top: 1rem; }
.norm-box {
    background: linear-gradient(135deg,#EBF5FB,#D6EAF8);
    border: 1.5px solid #2471A3; border-radius:10px;
    padding:12px; font-size:0.9em; color:#1A5276;
    text-align:center; margin-bottom:10px;
}
.result-box {
    background: linear-gradient(135deg,#FEF9E7,#FDEBD0);
    border: 2px solid #CA6F1E; border-radius:10px;
    padding:16px; margin:8px 0;
}
.green-box {
    background: linear-gradient(135deg,#F0FFF4,#D5F5E3);
    border: 2px solid #1A7A42; border-radius:10px;
    padding:14px; margin:8px 0;
}
.abw-box {
    background:#FFFBF5; border:2px solid #CA6F1E;
    border-radius:10px; padding:14px; margin:8px 0;
}
.info-box {
    background: linear-gradient(135deg,#EBF5FB,#D6EAF8);
    border:1.5px solid #2471A3; border-radius:10px;
    padding:14px; margin:8px 0;
}
.footer {
    text-align:center; color:#AAA; font-size:0.8em;
    margin-top:24px; padding:12px;
    border-top:1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;color:#1A252F;margin-bottom:4px;'>"
    "🌡️ Mollier h-x Diagramm</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#7F8C8D;margin-top:0;'>"
    "Bauphysikalische Feuchteanalyse  |  "
    "Made by Mahmoud Alhayek  |  Wetzel und von Seht</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='norm-box'>"
    "DIN 4108-2 Beiblatt 2  |  "
    "Schimmelkriterium: phi_Oberflaeche >= 80 %  |  "
    "Mindest-f_Rsi = 0,70"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Eingaben")
    st.markdown("---")
    st.markdown("### Aussenluft")
    T_a  = st.slider("Temperatur T_a [°C]",  -30.0, 20.0,  -5.0, 0.5)
    RF_a = st.slider("Relative Feuchte phi_a [%]", 0, 100, 80, 1)
    st.markdown("---")
    st.markdown("### Innenraum")
    T_r  = st.slider("Temperatur T_r [°C]",  15.0, 35.0, 20.0, 0.5)
    RF_i = st.slider("Relative Feuchte phi_i [%]", 0, 100, 50, 1)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8em;color:#7F8C8D;text-align:center;'>"
        "Wetzel und von Seht<br>Made by Mahmoud Alhayek"
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  BERECHNUNG
# ══════════════════════════════════════════════════════
fig, kv   = draw_mollier(T_a, RF_a, T_r, RF_i)
T_surf    = kv["T_surf"]
T_tau     = kv["T_tau"]
f_rsi_min = calc_f_rsi_min(T_surf, T_r, T_a)
abw_liste = norm_abweichung(T_r, T_a, RF_i)
ist_norm  = len(abw_liste) == 0

# ══════════════════════════════════════════════════════
#  DIAGRAMM ANZEIGEN
# ══════════════════════════════════════════════════════
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.markdown("---")

# ══════════════════════════════════════════════════════
#  METRIKEN
# ══════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("T_krit", f"{T_surf:.1f} °C",
              help="Mindest-Oberflaechentemperatur")
with c2:
    st.metric("Taupunkt", f"{T_tau:.1f} °C",
              help="Kondensationspunkt")
with c3:
    st.metric("f_Rsi min", f"{f_rsi_min:.3f}",
              help="Erforderlicher Temperaturfaktor")
with c4:
    st.metric("Delta x", f"{kv['delta_x']:+.2f} g/kg",
              help="Feuchteproduktion")

st.markdown("---")
st.markdown("## Ergebnis-Analyse")

# ══════════════════════════════════════════════════════
#  T_KRIT BOX
# ══════════════════════════════════════════════════════
tkrit_val  = f"{T_surf:.1f}"
ttau_val   = f"{T_tau:.1f}"

st.markdown(
    "<div class='result-box'>"
    "<b style='color:#CA6F1E;font-size:1.1em;'>T_krit = " + tkrit_val + " °C</b><br><br>"
    "Keine Oberflaeche darf unter <b>" + tkrit_val + " °C</b> abkuehlen.<br>"
    "Darunter: phi_Oberflaeche >= 80 % → <b>Schimmelgefahr!</b><br><br>"
    "<small>Taupunkt: " + ttau_val + " °C</small>"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  NORM ODER ABWEICHUNG
# ══════════════════════════════════════════════════════
frsi_val = f"{f_rsi_min:.3f}"
Ti_val   = f"{NORM_T_I:.0f}"
Te_val   = f"{NORM_T_E:.0f}"
RF_val   = f"{NORM_RF_I:.0f}"
Ts_val   = f"{T_surf:.1f}"

if ist_norm:
    st.markdown(
        "<div class='green-box'>"
        "<b style='color:#1A7A42;'>Norm-Randbedingungen — DIN 4108-2 Beiblatt 2</b><br><br>"
        "T_i = " + Ti_val + " °C  |  "
        "T_e = " + Te_val + " °C  |  "
        "phi_i = " + RF_val + " %<br><br>"
        "T_krit = <b>" + Ts_val + " °C</b>  |  "
        "f_Rsi,min = <b>" + frsi_val + "</b><br><br>"
        "<b>Das Bauteil muss einen f_Rsi >= 0,70 aufweisen.</b>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    abw_html = "<br>".join(["⚠️ " + a for a in abw_liste])
    st.markdown(
        "<div class='abw-box'>"
        "<b style='color:#A04000;'>Abweichende Randbedingungen</b><br><br>"
        + abw_html +
        "<br><br>"
        "T_krit = <b>" + Ts_val + " °C</b>  |  "
        "f_Rsi,min = <b>" + frsi_val + "</b>"
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  DETAIL-TABELLE
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Zustandspunkte")

xa_val    = f"{kv['x_a']:.3f}"
xi_val    = f"{kv['x_i']:.3f}"
ha_val    = f"{kv['h_a']:.1f}"
hi_val    = f"{kv['h_i']:.1f}"
rfauf_val = f"{kv['rf_aufgeheizt']:.1f}"
dx_val    = f"{kv['delta_x']:+.3f}"

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<div class='info-box'>"
        "<b>① Aussenluft</b><br>"
        "T = " + f"{T_a:.1f}" + " °C<br>"
        "phi = " + f"{RF_a:.0f}" + " %<br>"
        "x = " + xa_val + " g/kg<br>"
        "h = " + ha_val + " kJ/kg"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='info-box'>"
        "<b>② Aufgeheizt (x = konst.)</b><br>"
        "T = " + f"{T_r:.1f}" + " °C<br>"
        "phi = " + rfauf_val + " %<br>"
        "x = " + xa_val + " g/kg"
        "</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<div class='info-box'>"
        "<b>③ Innenluft</b><br>"
        "T = " + f"{T_r:.1f}" + " °C<br>"
        "phi = " + f"{RF_i:.0f}" + " %<br>"
        "x = " + xi_val + " g/kg<br>"
        "h = " + hi_val + " kJ/kg"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='info-box'>"
        "<b>④ Taupunkt</b><br>"
        "T_tau = " + ttau_val + " °C<br><br>"
        "<b>⑤ T_krit</b><br>"
        "T_krit = " + tkrit_val + " °C<br>"
        "f_Rsi,min = " + frsi_val
        + "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  FEUCHTE-DELTA
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div class='result-box'>"
    "<b>Feuchteproduktion im Raum</b><br><br>"
    "Delta x = <b>" + dx_val + " g/kg</b><br>"
    "Von Punkt ② nach ③<br><br>"
    "<small>Positiver Wert = Feuchte wird produziert (Kochen, Duschen, Personen...)</small>"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown(
    "<div class='footer'>"
    "Mollier h-x Diagramm  |  Bauphysik  |  "
    "Made by Mahmoud Alhayek  |  Wetzel und von Seht  |  "
    "DIN 4108-2"
    "</div>",
    unsafe_allow_html=True
)
