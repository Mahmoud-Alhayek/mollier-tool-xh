"""
Mollier h-x Diagramm - Web Version v5
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
        abw.append("T_Innen = " + str(round(T_i,1)) + " C  (Norm: " + str(int(NORM_T_I)) + " C)")
    if abs(T_e - NORM_T_E) > 0.5:
        abw.append("T_Aussen = " + str(round(T_e,1)) + " C  (Norm: " + str(int(NORM_T_E)) + " C)")
    if abs(RF_i - NORM_RF_I) > 2.0:
        abw.append("phi_Innen = " + str(int(RF_i)) + " %  (Norm: " + str(int(NORM_RF_I)) + " %)")
    return abw

# ══════════════════════════════════════════════════════
#  FARBEN
# ══════════════════════════════════════════════════════
C = {
    "sat":   "#C0392B",
    "aussen":"#1A5276",
    "innen": "#7B241C",
    "aufhz": "#1A7A42",
    "p_auf": "#27AE60",
    "p_feu": "#6C3483",
    "tkrit": "#CA6F1E",
    "tau":   "#0E6655",
    "grid":  "#E8E8E8",
    "bg":    "#FAFAFA",
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
                    ax.text(lx + 0.15, lT, str(rf_val) + "%",
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
         "① Aussenluft\nT = " + str(round(T_a,1)) + " C   phi = " + str(int(RF_a)) + " %\n"
         + "x = " + str(round(x_a,2)) + " g/kg   h = " + str(round(h_a,1)) + " kJ/kg",
         C["aussen"], ha="right", va="top", dx=dx_l, dy=-0.5)

    _lbl(x_a, T_r,
         "② Aufgeheizt  (x = konst.)\nT = " + str(round(T_r,1)) + " C   phi = " + str(round(rf_auf,1)) + " %\n"
         + "x = " + str(round(x_a,2)) + " g/kg",
         C["aufhz"], ha="right", va="bottom", dx=dx_l, dy=0.6)

    _lbl(x_i, T_r,
         "③ Innenluft\nT = " + str(round(T_r,1)) + " C   phi = " + str(int(RF_i)) + " %\n"
         + "x = " + str(round(x_i,2)) + " g/kg   h = " + str(round(h_i,1)) + " kJ/kg",
         C["innen"], ha="left", va="bottom", dx=dx_r, dy=0.6)

    if x_i > x_a + 0.08:
        delta_x_val = round(x_i - x_a, 2)
        sign = "+" if delta_x_val >= 0 else ""
        ax.text((x_a + x_i) / 2, T_r + 0.8,
                "Dx = " + sign + str(delta_x_val) + " g/kg",
                fontsize=8.5, color=C["p_feu"],
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=C["p_feu"],
                          alpha=0.92, lw=0.9),
                zorder=12, clip_on=True)

    _lbl(x_i, T_tau,
         "④ Taupunkt\nT_tau = " + str(round(T_tau,1)) + " C",
         C["tau"], ha="left", va="top", dx=dx_r, dy=-0.5)

    ax.text(0.35, T_surf + 0.7,
            "⑤ T_krit = " + str(round(T_surf,1)) + " C   —   phi_Oberflaeche >= 80% → Schimmelgrenze",
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
               markerfacecolor=C["aufhz"],  markersize=9,  label="② Aufgeheizt (x=konst.)"),
        Line2D([0],[0], marker="o", color="w", lw=0,
               markerfacecolor=C["innen"],  markersize=10, label="③ Innenluft"),
        Line2D([0],[0], marker="^", color="w", lw=0,
               markerfacecolor=C["tau"],    markersize=9,  label="④ Taupunkt"),
        Line2D([0],[0], marker="s", color="w", lw=0,
               markerfacecolor=C["tkrit"],  markersize=9,  label="⑤ T_krit"),
        mpatches.Patch(color="none", label=" "),
        mpatches.Patch(color="none", label="--- Prozesse ---"),
        Line2D([0],[0], color=C["sat"],   lw=2.5, label="Saettigungslinie (phi=100%)"),
        Line2D([0],[0], color=C["p_auf"], lw=2.0, label="① → ② Aufheizung"),
        Line2D([0],[0], color=C["p_feu"], lw=2.0, label="② → ③ Feuchteproduktion"),
        Line2D([0],[0], color=C["tau"],   lw=1.6, ls="--", label="③ → ④ Abkuehlung"),
        Line2D([0],[0], color=C["tkrit"], lw=2.0, ls="--", label="⑤ Schimmelgrenze"),
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    fontsize=8, framealpha=0.97,
                    ncol=1, title="Legende", title_fontsize=9,
                    edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(1.2)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-20, 50)
    ax.set_xlabel("Feuchtegehalt   x   [g/kg trockene Luft]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.set_ylabel("Temperatur   T   [°C]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.tick_params(axis="both", labelsize=9, colors="#555555")
    ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twiny()
    ax2.set_xlim(0, X_MAX)
    raw_ticks = np.arange(0, 31, 5)
    x_ticks   = raw_ticks[raw_ticks <= X_MAX]
    pd_vals   = [
        xk / 1000.0 / (0.62198 + xk / 1000.0) * p_pa / 100.0
        for xk in x_ticks
    ]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([str(round(v,1)) for v in pd_vals], fontsize=8.5)
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
.green-box {
    background: linear-gradient(135deg,#F0FFF4,#D5F5E3);
    border: 2px solid #1A7A42; border-radius:10px;
    padding:16px; margin:8px 0;
}
.orange-box {
    background: linear-gradient(135deg,#FEF9E7,#FDEBD0);
    border: 2px solid #CA6F1E; border-radius:10px;
    padding:16px; margin:8px 0;
}
.abw-box {
    background:#FFFBF5; border:2px solid #E67E22;
    border-radius:10px; padding:16px; margin:8px 0;
}
.blue-box {
    background: linear-gradient(135deg,#EBF5FB,#D6EAF8);
    border:1.5px solid #2471A3; border-radius:10px;
    padding:14px; margin:8px 0;
}
.purple-box {
    background: linear-gradient(135deg,#F9F3FF,#EDE7F6);
    border:2px solid #8E44AD; border-radius:10px;
    padding:16px; margin:8px 0;
}
.card-aussen {
    background:white; border-radius:10px; padding:16px;
    border-left:5px solid #1A5276;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); margin:6px 0;
}
.card-aufhz {
    background:white; border-radius:10px; padding:16px;
    border-left:5px solid #1A7A42;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); margin:6px 0;
}
.card-innen {
    background:white; border-radius:10px; padding:16px;
    border-left:5px solid #7B241C;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); margin:6px 0;
}
.card-tau {
    background:white; border-radius:10px; padding:16px;
    border-left:5px solid #CA6F1E;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); margin:6px 0;
}
.tbl-row {
    display:flex; justify-content:space-between;
    padding:3px 0; border-bottom:1px solid #F0F0F0;
    font-size:0.9em;
}
.footer {
    text-align:center; color:#AAA; font-size:0.8em;
    margin-top:24px; padding:12px; border-top:1px solid #eee;
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
    "📐 Grundlage: DIN 4108-2 Beiblatt 2  |  "
    "Schimmelkriterium: phi_Oberflaeche >= 80%  |  "
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
    T_r  = st.slider("Temperatur T_r [°C]", 15.0, 35.0, 20.0, 0.5)
    RF_i = st.slider("Relative Feuchte phi_i [%]", 0, 100, 50, 1)
    st.markdown("---")

    # Schimmelgrenz-Tabelle in Sidebar
    st.markdown("### Schimmelgrenze phi_i vs T_krit")
    st.markdown(
        "<small style='color:#7F8C8D;'>"
        "Bauteil darf bis T_krit abkuehlen<br>"
        "ohne Schimmel (phi_surf >= 80%)"
        "</small>",
        unsafe_allow_html=True
    )
    tbl_rows = []
    for rf_t in range(30, 75, 5):
        ts_t = critical_surface_temp(T_r, float(rf_t))
        tbl_rows.append(
            "<div class='tbl-row'>"
            "<span style='color:#555;'>phi_i = " + str(rf_t) + " %</span>"
            "<span style='color:#CA6F1E;font-weight:bold;'>" + str(round(ts_t,1)) + " °C</span>"
            "</div>"
        )
    st.markdown(
        "<div style='background:white;border-radius:8px;padding:10px;"
        "border:1px solid #E0E0E0;'>" + "".join(tbl_rows) + "</div>",
        unsafe_allow_html=True
    )
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
rf_auf    = kv["rf_aufgeheizt"]

# ══════════════════════════════════════════════════════
#  DIAGRAMM
# ══════════════════════════════════════════════════════
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.markdown("---")

# ══════════════════════════════════════════════════════
#  METRIKEN
# ══════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("⑤ T_krit", str(round(T_surf,1)) + " °C",
              help="Mindest-Oberflaechentemperatur — darunter Schimmelgefahr")
with c2:
    st.metric("④ Taupunkt", str(round(T_tau,1)) + " °C",
              help="Kondensationspunkt der Innenluft")
with c3:
    st.metric("f_Rsi,min", str(round(f_rsi_min,3)),
              help="Erforderlicher Temperaturfaktor nach DIN 4108-2")
with c4:
    delta_x = kv["delta_x"]
    sign = "+" if delta_x >= 0 else ""
    st.metric("Dx Feuchte", sign + str(round(delta_x,2)) + " g/kg",
              help="Feuchteproduktion im Raum")

st.markdown("---")

# ══════════════════════════════════════════════════════
#  BLOCK 1 — T_KRIT ERKLARUNG
# ══════════════════════════════════════════════════════
st.markdown("## Ergebnis-Analyse")

st.markdown(
    "<div class='orange-box'>"
    "<b style='color:#CA6F1E;font-size:1.15em;'>⑤  T_krit = " + str(round(T_surf,1)) + " °C</b><br><br>"
    "Bei diesen Randbedingungen darf das Bauteil (Wand, Ecke, Waermebruecke) "
    "bis auf <b>" + str(round(T_surf,1)) + " °C</b> abkuehlen — "
    "<b>darunter wird phi_Oberflaeche >= 80% erreicht → Schimmelgefahr!</b><br><br>"
    "<span style='color:#7F8C8D;font-size:0.88em;'>"
    "Taupunkt: " + str(round(T_tau,1)) + " °C  |  "
    "T_krit liegt immer ueber dem Taupunkt, da Schimmel schon vor dem Kondensieren entsteht."
    "</span>"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  BLOCK 2 — NORM ODER ABWEICHUNG
# ══════════════════════════════════════════════════════
if ist_norm:
    st.markdown(
        "<div class='green-box'>"
        "<b style='color:#1A7A42;font-size:1.05em;'>✅  Norm-Randbedingungen — DIN 4108-2 Beiblatt 2</b><br><br>"
        "Die eingegebenen Randbedingungen entsprechen den Norm-Randbedingungen.<br><br>"
        "<b>Norm-Randbedingungen DIN 4108-2 Beiblatt 2:</b><br>"
        "&nbsp;&nbsp;T_i = " + str(int(NORM_T_I)) + " °C  |  "
        "T_e = " + str(int(NORM_T_E)) + " °C  |  "
        "phi_i = " + str(int(NORM_RF_I)) + " %<br><br>"
        "<b>Der Norm-Nachweis ergibt:</b><br>"
        "&nbsp;&nbsp;T_krit = <b>" + str(round(T_surf,1)) + " °C</b><br>"
        "&nbsp;&nbsp;Erforderlicher f_Rsi,min = <b>" + str(round(f_rsi_min,3)) + "</b><br><br>"
        "<b>Das Bauteil muss einen f_Rsi >= 0,70 aufweisen.</b><br>"
        "<span style='color:#555;font-size:0.88em;'>"
        "Den tatsaechlichen f_Rsi aus dem Waermebruecken-Detailnachweis ermitteln."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    abw_html = ""
    for a in abw_liste:
        abw_html += "&nbsp;&nbsp;⚠️  " + a + "<br>"

    st.markdown(
        "<div class='abw-box'>"
        "<b style='color:#A04000;font-size:1.05em;'>⚠️  Abweichende Randbedingungen — kein Norm-Nachweis</b><br><br>"
        "<b>Abweichung von DIN 4108-2 Beiblatt 2:</b><br>"
        + abw_html +
        "<br><span style='color:#555;font-size:0.88em;'>"
        "Norm: T_i = " + str(int(NORM_T_I)) + " °C  |  "
        "T_e = " + str(int(NORM_T_E)) + " °C  |  "
        "phi_i = " + str(int(NORM_RF_I)) + " %"
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='blue-box'>"
        "<b style='color:#1A5276;'>Was bedeutet das?</b><br><br>"
        "Die berechnete T_krit = <b>" + str(round(T_surf,1)) + " °C</b> gilt fuer "
        "<b>diese spezifischen Randbedingungen</b> "
        "(T_i = " + str(round(T_r,1)) + " °C, "
        "phi_i = " + str(int(RF_i)) + " %, "
        "T_e = " + str(round(T_a,1)) + " °C).<br><br>"
        "Diese Berechnung ist kein Norm-Nachweis nach DIN 4108-2, "
        "sondern eine <b>individuelle Analyse</b> fuer die tatsaechlichen Randbedingungen.<br><br>"
        "<b>Anwendungsbeispiele:</b><br>"
        "&nbsp;&nbsp;• Bestandsgebaeude mit reduzierter Raumfeuchte (z.B. phi_i &le; 35 %)<br>"
        "&nbsp;&nbsp;• Denkmalgeschuetzte Gebaeude, bei denen Daemmung nicht moeglich ist<br>"
        "&nbsp;&nbsp;• Pruefung: Ab welcher Wandtemperatur wird es bei der tatsaechlichen phi_i kritisch?<br>"
        "&nbsp;&nbsp;• Waermebruecken-Analyse mit abweichenden Klimabedingungen"
        "</div>",
        unsafe_allow_html=True
    )

    f_rsi_str = str(round(f_rsi_min,3))
    st.markdown(
        "<div class='purple-box'>"
        "<b style='color:#6C3483;font-size:1.05em;'>📌  Praktische Aussage:</b><br><br>"
        "Mit T_i = " + str(round(T_r,1)) + " °C und phi_i = " + str(int(RF_i)) + " % gilt:<br>"
        "→ Das Bauteil muss waermer als <b>" + str(round(T_surf,1)) + " °C</b> bleiben.<br><br>"
        "<b>Erforderlicher Mindest-f_Rsi:</b><br>"
        "<span style='font-size:1.4em;color:#6C3483;'><b>f_Rsi,min = " + f_rsi_str + "</b></span><br><br>"
        "<span style='color:#555;font-size:0.85em;'>"
        "Formel: f_Rsi,min = (T_krit - T_e) / (T_i - T_e) = "
        "(" + str(round(T_surf,1)) + " - " + str(round(T_a,1)) + ") / "
        "(" + str(round(T_r,1)) + " - " + str(round(T_a,1)) + ") = " + f_rsi_str + "<br><br>"
        "Hinweis: Der Norm-Mindest-f_Rsi von 0,70 gilt ausschliesslich fuer die Norm-Randbedingungen "
        "(T_i = " + str(int(NORM_T_I)) + " °C, T_e = " + str(int(NORM_T_E)) + " °C, "
        "phi_i = " + str(int(NORM_RF_I)) + " %). "
        "Bei abweichenden Randbedingungen ergibt sich ein anderer erforderlicher f_Rsi-Wert."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  BLOCK 3 — ZUSTANDSPUNKTE
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Zustandspunkte im Detail")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<div class='card-aussen'>"
        "<b style='color:#1A5276;font-size:1.0em;'>🔵  ① Aussenluft</b><br><br>"
        "<div class='tbl-row'><span style='color:#555;'>Temperatur T_e</span>"
        "<span><b>" + str(round(T_a,1)) + " °C</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Relative Feuchte phi_e</span>"
        "<span><b>" + str(int(RF_a)) + " %</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Feuchtegehalt x</span>"
        "<span><b>" + str(round(kv["x_a"],3)) + " g/kg</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Enthalpie h</span>"
        "<span><b>" + str(round(kv["h_a"],1)) + " kJ/kg</b></span></div>"
        "<br><span style='color:#888;font-size:0.85em;font-style:italic;'>"
        "Kalte Luft enthaelt wenig absoluten Wasserdampf trotz hoher relativer Feuchte."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='card-aufhz'>"
        "<b style='color:#1A7A42;font-size:1.0em;'>🟢  ② Aufgeheizte Luft (x = konst.)</b><br><br>"
        "<div class='tbl-row'><span style='color:#555;'>Temperatur T_r</span>"
        "<span><b>" + str(round(T_r,1)) + " °C</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Relative Feuchte phi</span>"
        "<span><b>" + str(round(rf_auf,1)) + " %</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Feuchtegehalt x</span>"
        "<span><b>" + str(round(kv["x_a"],3)) + " g/kg</b></span></div>"
        "<br><span style='color:#888;font-size:0.85em;font-style:italic;'>"
        "Beim Aufheizen sinkt phi von " + str(int(RF_a)) + " % auf " + str(round(rf_auf,1)) + " % — x bleibt konstant."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<div class='card-innen'>"
        "<b style='color:#7B241C;font-size:1.0em;'>🔴  ③ Innenluft</b><br><br>"
        "<div class='tbl-row'><span style='color:#555;'>Temperatur T_r</span>"
        "<span><b>" + str(round(T_r,1)) + " °C</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Relative Feuchte phi_i</span>"
        "<span><b>" + str(int(RF_i)) + " %</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Feuchtegehalt x</span>"
        "<span><b>" + str(round(kv["x_i"],3)) + " g/kg</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Enthalpie h</span>"
        "<span><b>" + str(round(kv["h_i"],1)) + " kJ/kg</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Dx Feuchteproduktion</span>"
        "<span><b>" + ("+") + str(round(kv["delta_x"],2)) + " g/kg</b></span></div>"
        "<br><span style='color:#888;font-size:0.85em;font-style:italic;'>"
        "Dx = durch Nutzung hinzugefuegte Feuchtigkeit (Kochen, Duschen, Personen)."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='card-tau'>"
        "<b style='color:#CA6F1E;font-size:1.0em;'>🌡️  ④ Taupunkt  &  ⑤ T_krit</b><br><br>"
        "<div class='tbl-row'><span style='color:#555;'>Taupunkt T_tau</span>"
        "<span><b>" + str(round(T_tau,1)) + " °C</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>T_krit (Schimmelgrenze)</span>"
        "<span><b>" + str(round(T_surf,1)) + " °C</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Erforderl. f_Rsi,min</span>"
        "<span><b>" + str(round(f_rsi_min,3)) + "</b></span></div>"
        "<div class='tbl-row'><span style='color:#555;'>Norm-Mindest-f_Rsi</span>"
        "<span><b>0,70 (DIN 4108-2)</b></span></div>"
        "<br><span style='color:#888;font-size:0.85em;font-style:italic;'>"
        "T_krit (" + str(round(T_surf,1)) + " °C) > Taupunkt (" + str(round(T_tau,1)) + " °C): "
        "Schimmel entsteht vor dem sichtbaren Kondensieren."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  BLOCK 4 — UMKEHR-ANALYSE
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Umkehr-Analyse: Bei bekannter Bauteil-Oberflachentemperatur")

st.markdown(
    "<div class='blue-box'>"
    "Wenn Sie die <b>tatsaechliche Oberflaechentemperatur</b> Ihres Bauteils "
    "aus einem Waermebruecken-Nachweis kennen, koennen Sie hier die "
    "<b>maximale zulaessige Innenraumfeuchte phi_i</b> ablesen:<br>"
    "<span style='font-size:0.85em;color:#666;'>"
    "phi_i,max = p_sat(T_Oberflaeche) x 80% / p_sat(T_i) x 100"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)

T_wb = st.slider(
    "Oberflaechentemperatur aus Waermebruecken-Nachweis [°C]",
    min_value=-20.0, max_value=35.0,
    value=round(T_surf, 1), step=0.1
)

rf_max = max_rf_for_surface_temp(T_wb, T_r)
f_rsi_wb = calc_f_rsi_min(T_wb, T_r, T_a)

if rf_max >= 60:
    box_cls = "green-box"
    ico = "✅"
    col_txt = "#1A7A42"
elif rf_max >= 45:
    box_cls = "orange-box"
    ico = "⚠️"
    col_txt = "#9A7D0A"
else:
    box_cls = "abw-box"
    ico = "🔴"
    col_txt = "#C0392B"

st.markdown(
    "<div class='" + box_cls + "'>"
    "<b style='color:" + col_txt + ";font-size:1.1em;'>"
    + ico + "  T_Oberflaeche = " + str(round(T_wb,1)) + " °C</b><br><br>"
    "Maximale zulaessige Innenraumfeuchte:<br>"
    "<span style='font-size:1.5em;color:" + col_txt + ";'>"
    "<b>phi_i,max = " + str(round(rf_max,1)) + " %</b></span><br><br>"
    "Solange phi_i unter <b>" + str(round(rf_max,1)) + " %</b> bleibt, "
    "wird an dieser Oberflaeche (" + str(round(T_wb,1)) + " °C) "
    "keine phi_Oberflaeche >= 80% erreicht → kein Schimmelrisiko.<br><br>"
    "<span style='color:#666;font-size:0.85em;'>"
    "Zugehoeriger f_Rsi: " + str(round(f_rsi_wb,3)) + "<br>"
    "Formel: phi_i,max = p_sat(" + str(round(T_wb,1)) + " °C) x 80% "
    "/ p_sat(" + str(round(T_r,1)) + " °C) x 100"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown(
    "<div class='footer'>"
    "Mollier h-x Diagramm  |  Bauphysik  |  "
    "Made by Mahmoud Alhayek  |  Wetzel und von Seht  |  DIN 4108-2"
    "</div>",
    unsafe_allow_html=True
)
