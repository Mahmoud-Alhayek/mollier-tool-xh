"""
Mollier h-x Diagramm — Web Version v5
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
    c   = np.where(T_C >= 0, 237.300, 265.500)
    d   = np.where(T_C >= 0,  17.269,  21.875)
    return 610.5 * np.exp(d * T_C / (c + T_C))

def x_from_T_RF(T_C, RF, p_pa=101325.0):
    p_ws = float(p_sat(T_C))
    p_w  = (RF / 100.0) * p_ws
    d    = p_pa - p_w
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
    T_s   = float(T_r)
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
    p_w_max    = (rf_limit / 100.0) * p_sat_surf
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
        abw.append(f"T_Außen = {T_e:.1f} °C  (Norm: {NORM_T_E:.0f} °C)")
    if abs(RF_i - NORM_RF_I) > 2.0:
        abw.append(f"φ_Innen = {RF_i:.0f} %  (Norm: {NORM_RF_I:.0f} %)")
    return abw

# ══════════════════════════════════════════════════════
#  FARBEN
# ══════════════════════════════════════════════════════
C = {
    "sat":    "#C0392B",
    "außen":  "#1A5276",
    "innen":  "#7B241C",
    "aufhz":  "#1A7A42",
    "p_auf":  "#27AE60",
    "p_feu":  "#6C3483",
    "t_krit": "#CA6F1E",
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

    # ── Enthalpie-Isolinien ──────────────────────────
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
                    color="#B2DFDB", lw=0.55,
                    alpha=0.5, zorder=1)

    # ── RF-Isolinien ────────────────────────────────
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
            x_vals.append(
                0.62198 * pw / den * 1000.0 if den > 0 else np.nan
            )
        x_arr = np.array(x_vals)
        vis   = np.isfinite(x_arr) & (x_arr <= X_MAX * 1.02)
        if vis.sum() > 1:
            lw = 2.2 if rf_val == 100 else (
                1.1 if rf_val % 20 == 0 else 0.6
            )
            ax.plot(x_arr[vis], T_plot[vis],
                    color=col, lw=lw, alpha=0.80, zorder=2)
            for tgt in [32, 28, 24, 20, 15]:
                idx = np.argmin(np.abs(T_plot - tgt))
                lx, lT = x_arr[idx], T_plot[idx]
                if np.isfinite(lx) and 0.3 < lx < X_MAX - 0.5:
                    ax.text(lx + 0.15, lT, f"{rf_val}%",
                            fontsize=7, color=col,
                            fontweight="bold" if rf_val == 100
                            else "normal",
                            va="center", ha="left",
                            clip_on=True, zorder=6)
                    break

    # ── Sättigungslinie ──────────────────────────────
    x_sat = np.array([
        0.62198 * float(p_sat(T))
        / (p_pa - float(p_sat(T))) * 1000.0
        for T in T_plot
    ])
    vis = np.isfinite(x_sat) & (x_sat <= X_MAX * 1.05)
    ax.plot(x_sat[vis], T_plot[vis],
            color=C["sat"], lw=2.5, zorder=4)
    ax.fill_betweenx(T_plot[vis], x_sat[vis], X_MAX,
                     alpha=0.04, color=C["sat"], zorder=1)

    # ── T_krit Linie ─────────────────────────────────
    ax.axhline(T_surf, color=C["t_krit"], ls=(0, (6, 3)),
               lw=2.2, alpha=0.92, zorder=5)
    ax.fill_between([0, X_MAX], [-20, -20], [T_surf, T_surf],
                    alpha=0.04, color=C["t_krit"], zorder=1)

    # ── Prozess-Pfeile ───────────────────────────────
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

    # ── Punkte ───────────────────────────────────────
    pk = dict(zorder=10, clip_on=False)
    ms = 130
    ax.scatter([x_a], [T_a],    color=C["außen"],  s=ms,
               marker="o", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_a], [T_r],    color=C["aufhz"],  s=ms - 30,
               marker="D", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_r],    color=C["innen"],  s=ms,
               marker="o", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_tau],  color=C["tau"],    s=ms - 30,
               marker="^", edgecolors="white", linewidths=1.5, **pk)
    ax.scatter([x_i], [T_surf], color=C["t_krit"], s=ms - 30,
               marker="s", edgecolors="white", linewidths=1.5, **pk)

    # ── Labels ───────────────────────────────────────
    def _lbl(x, y, txt, col, ha="left", va="bottom", dx=0.0, dy=0.0):
        ax.text(x + dx, y + dy, txt,
                fontsize=8.5, color=col, ha=ha, va=va,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="white", ec=col,
                          alpha=0.93, lw=1.1),
                zorder=12, clip_on=True)

    dx_r, dx_l = 0.35, -0.35

    _lbl(x_a, T_a,
         f"① Außenluft\nT = {T_a:.1f} °C   φ = {RF_a:.0f} %\n"
         f"x = {x_a:.2f} g/kg\nh = {h_a:.1f} kJ/kg",
         C["außen"], ha="right", va="top", dx=dx_l, dy=-0.5)

    _lbl(x_a, T_r,
         f"② Aufgeheizt\nT = {T_r:.1f} °C   φ = {rf_auf:.1f} %\n"
         f"x = {x_a:.2f} g/kg",
         C["aufhz"], ha="right", va="bottom", dx=dx_l, dy=0.6)

    _lbl(x_i, T_r,
         f"③ Innenluft\nT = {T_r:.1f} °C   φ = {RF_i:.0f} %\n"
         f"x = {x_i:.2f} g/kg\nh = {h_i:.1f} kJ/kg",
         C["innen"], ha="left", va="bottom", dx=dx_r, dy=0.6)

    if x_i > x_a + 0.08:
        ax.text((x_a + x_i) / 2, T_r + 0.8,
                f"Δx = {x_i - x_a:+.2f} g/kg",
                fontsize=8.5, color=C["p_feu"],
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=C["p_feu"],
                          alpha=0.92, lw=0.9),
                zorder=12, clip_on=True)

    _lbl(x_i, T_tau,
         f"④ Taupunkt\nT_τ = {T_tau:.1f} °C",
         C["tau"], ha="left", va="top", dx=dx_r, dy=-0.5)

    ax.text(0.35, T_surf + 0.7,
            f"⑤ T_krit = {T_surf:.1f} °C   —   "
            f"φ_Oberfläche ≥ 80 % → Schimmelgrenze",
            fontsize=9, color=C["t_krit"],
            ha="left", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec=C["t_krit"],
                      alpha=0.93, lw=1.1),
            zorder=12, clip_on=True)

    # ── Legende ─────────────────────────────────────
    handles = [
        mpatches.Patch(color="none",
                       label="── Zustands-Punkte ──"),
        Line2D([0],[0], marker="o", color="w", lw=0,
               markerfacecolor=C["außen"], markersize=10,
               label="① Außenluft"),
        Line2D([0],[0], marker="D", color="w", lw=0,
               markerfacecolor=C["aufhz"], markersize=9,
               label="② Aufgeheizt (x = konst.)"),
        Line2D([0],[0], marker="o", color="w", lw=0,
               markerfacecolor=C["innen"], markersize=10,
               label="③ Innenluft"),
        Line2D([0],[0], marker="^", color="w", lw=0,
               markerfacecolor=C["tau"], markersize=9,
               label="④ Taupunkt"),
        Line2D([0],[0], marker="s", color="w", lw=0,
               markerfacecolor=C["t_krit"], markersize=9,
               label="⑤ Krit. Oberflächentemp."),
        mpatches.Patch(color="none", label=" "),
        mpatches.Patch(color="none",
                       label="── Prozesse & Linien ──"),
        Line2D([0],[0], color=C["sat"],    lw=2.5,
               label="Sättigungslinie (φ=100%)"),
        Line2D([0],[0], color=C["p_auf"],  lw=2.0,
               label="① → ② Aufheizung"),
        Line2D([0],[0], color=C["p_feu"],  lw=2.0,
               label="② → ③ Feuchteproduktion"),
        Line2D([0],[0], color=C["tau"],    lw=1.6, ls="--",
               label="③ → ④ Abkühlung"),
        Line2D([0],[0], color=C["t_krit"], lw=2.0, ls="--",
               label="⑤ Schimmelgrenztemp."),
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    fontsize=8, framealpha=0.97,
                    ncol=1, title="Legende", title_fontsize=9,
                    edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(1.2)

    # ── Achsen ──────────────────────────────────────
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-20, 50)
    ax.set_xlabel("Feuchtegehalt   x   [g/kg trockene Luft]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.set_ylabel("Temperatur   T   [°C]",
                  fontsize=12, labelpad=10, color="#333333")
    ax.tick_params(axis="both", labelsize=9, colors="#555555")
    ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Obere Achse Partialdruck ─────────────────────
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
.card {
    background:white; border-radius:12px;
    padding:16px; margin:6px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.umkehr-box {
    background: linear-gradient(135deg,#F9F3FF,#EDE7F6);
    border:2px solid #8E44AD; border-radius:10px;
    padding:16px; margin:8px 0;
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
    "Made by Mahmoud Alhayek  |  Wetzel & von Seht</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='norm-box'>"
    "📐 Grundlage: DIN 4108-2 Beiblatt 2  |  "
    "Schimmelkriterium: φ_Oberfläche ≥ 80 %  |  "
    "Mindest-f_Rsi = 0,70"
    "</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Eingaben")
    st.markdown("---")
    st.markdown("### 🌬️ ① Außenluft")
    T_a  = st.slider("Temperatur T_a [°C]",  -30.0, 20.0,  -5.0, 0.5)
    RF_a = st.slider("Relative Feuchte φ_a [%]", 0, 100, 80, 1)
    st.markdown("---")
    st.markdown("### 🏠 ③ Innenraum")
    T_r  = st.slider("Temperatur T_r [°C]",  15.0, 35.0, 20.0, 0.5)
    RF_i = st.slider("Relative Feuchte φ_i [%]", 0, 100, 50, 1)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8em;color:#7F8C8D;text-align:center;'>"
        "🏢 Wetzel & von Seht<br>Made by Mahmoud Alhayek"
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
rf_auf    = kv["rf_aufgeheizt"]
abw_liste = norm_abweichung(T_r, T_a, RF_i)
ist_norm  = len(abw_liste) == 0

# ── Diagramm ────────────────────────────────────────
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.markdown("---")

# ── Metriken ────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("⑤ T_krit",   f"{T_surf:.1f} °C",
              help="Mindest-Oberflächentemperatur gegen Schimmel")
with c2:
    st.metric("④ Taupunkt", f"{T_tau:.1f} °C",
              help="Kondensationspunkt der Innenluft")
with c3:
    st.metric("f_Rsi,min",  f"{f_rsi_min:.3f}",
              help="Erforderlicher Temperaturfaktor")
with c4:
    st.metric("Δx Feuchte", f"{kv['delta_x']:+.2f} g/kg",
              help="Feuchteproduktion im Innenraum")

st.markdown("---")
st.markdown("## 📊 Ergebnis-Analyse")

# ══════════════════════════════════════════════════════
#  BLOCK 1 — T_krit
# ══════════════════════════════════════════════════════
st.markdown(
    f"<div class='result-box'>"
    f"<b style='color:#CA6F1E;font-size:1.15em;'>"
    f"⑤  T_krit = {T_surf:.1f} °C</b><br><br>"
    f"Bei diesen Randbedingungen darf das Bauteil "
    f"(Wand, Ecke, Wärmebrücke) bis auf "
    f"<b>{T_surf:.1f} °C</b> abkühlen — "
    f"darunter wird φ_Oberfläche ≥ 80 % erreicht "
    f"→ <b>Schimmelgefahr!</b><br><br>"
    f"<small style='color:#888;'>"
    f"Taupunkt: {T_tau:.1f} °C  |  "
    f"T_krit liegt immer über dem Taupunkt, da Schimmel "
    f"schon vor dem Kondensieren entsteht."
    f"</small>"
    f"</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  BLOCK 2 — Norm oder Abweichung
# ══════════════════════════════════════════════════════
if ist_norm:
    st.markdown(
        f"<div class='green-box'>"
        f"<b style='color:#1A7A42;font-size:1.05em;'>"
        f"✅  Norm-Randbedingungen — DIN 4108-2 Beiblatt 2</b>"
        f"<br><br>"
        f"T_i = {NORM_T_I:.0f} °C  |  "
        f"T_e = {NORM_T_E:.0f} °C  |  "
        f"φ_i = {NORM_RF_I:.0f} %<br><br>"
        f"T_krit = <b>{T_surf:.1f} °C</b>  |  "
        f"f_Rsi,min = <b>{f_rsi_min:.3f}</b><br><br>"
        f"<b>Das Bauteil muss einen f_Rsi ≥ 0,70 aufweisen.</b><br>"
        f"<small style='color:#555;'>Den tatsächlichen f_Rsi aus "
        f"dem Wärmebrücken-Detailnachweis ermitteln.</small>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    abw_html = "<br>".join([f"⚠️ {a}" for a in abw_liste])
    st.markdown(
        f"<div class='abw-box'>"
        f"<b style='color:#A04000;font-size:1.05em;'>"
        f"⚠️  Abweichende Randbedingungen</b><br><br>"
        f"{abw_html}<br><br>"
        f"<small style='color:#666;'>"
        f"Norm: T_i = {NORM_T_I:.0f} °C  |  "
        f"T_e = {NORM_T_E:.0f} °C  |  "
        f"φ_i = {NORM_RF_I:.0f} %</small><br><br>"
        f"<b style='color:#1A5276;'>Was bedeutet das?</b><br>"
        f"Die berechnete T_krit = <b>{T_surf:.1f} °C</b> gilt für "
        f"diese spezifischen Randbedingungen "
        f"(T_i = {T_r:.1f} °C, φ_i = {RF_i:.0f} %, "
        f"T_e = {T_a:.1f} °C).<br>"
        f"Dies ist kein Norm-Nachweis, sondern eine "
        f"<b>individuelle Analyse</b>.<br><br>"
        f"f_Rsi,min = <b>{f_rsi_min:.3f}</b><br>"
        f"<small style='color:#666;'>"
        f"Formel: ({T_surf:.1f} − {T_a:.1f}) / "
        f"({T_r:.1f} − {T_a:.1f}) = {f_rsi_min:.3f}</small>"
        f"</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  BLOCK 3 — Zustandspunkte Detail
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔍 Zustandspunkte im Detail")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"<div class='card' style='border-left:4px solid {C[\"außen\"]};'>"
        f"<b style='color:{C[\"außen\"]};'>① Außenluft</b><br><br>"
        f"🌡️ Temperatur T_e = <b>{T_a:.1f} °C</b><br>"
        f"💧 Rel. Feuchte φ_e = <b>{RF_a:.0f} %</b><br>"
        f"📊 Feuchtegehalt x = <b>{kv['x_a']:.2f} g/kg</b><br>"
        f"⚡ Enthalpie h = <b>{kv['h_a']:.1f} kJ/kg</b><br><br>"
        f"<small style='color:#777;'>Kalte Luft enthält wenig "
        f"absoluten Wasserdampf trotz hoher relativer Feuchte.</small>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='card' style='border-left:4px solid {C[\"innen\"]};'>"
        f"<b style='color:{C[\"innen\"]};'>③ Innenluft</b><br><br>"
        f"🌡️ Temperatur T_i = <b>{T_r:.1f} °C</b><br>"
        f"💧 Rel. Feuchte φ_i = <b>{RF_i:.0f} %</b><br>"
        f"📊 Feuchtegehalt x = <b>{kv['x_i']:.2f} g/kg</b><br>"
        f"⚡ Enthalpie h = <b>{kv['h_i']:.1f} kJ/kg</b><br>"
        f"➕ Δx Feuchteproduktion = <b>{kv['delta_x']:+.2f} g/kg</b><br><br>"
        f"<small style='color:#777;'>Δx = durch Nutzung "
        f"hinzugefügte Feuchtigkeit.</small>"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div class='card' style='border-left:4px solid {C[\"aufhz\"]};'>"
        f"<b style='color:{C[\"aufhz\"]};'>② Aufgeheizte Luft "
        f"(x = konst.)</b><br><br>"
        f"🌡️ Temperatur T_r = <b>{T_r:.1f} °C</b><br>"
        f"💧 Rel. Feuchte φ = <b>{rf_auf:.1f} %</b><br>"
        f"📊 Feuchtegehalt x = <b>{kv['x_a']:.2f} g/kg</b><br><br>"
        f"<small style='color:#777;'>Beim Aufheizen sinkt φ von "
        f"{RF_a:.0f} % auf {rf_auf:.1f} % — x bleibt konstant.</small>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='card' style='border-left:4px solid {C[\"t_krit\"]};'>"
        f"<b style='color:{C[\"t_krit\"]};'>④ Taupunkt & "
        f"⑤ T_krit</b><br><br>"
        f"🌡️ Taupunkt T_τ = <b>{T_tau:.1f} °C</b><br>"
        f"⚠️ T_krit (Schimmelgrenze) = <b>{T_surf:.1f} °C</b><br>"
        f"📐 f_Rsi,min = <b>{f_rsi_min:.3f}</b><br>"
        f"📏 Norm-Mindest-f_Rsi = <b>0,70</b> (DIN 4108-2)<br><br>"
        f"<small style='color:#777;'>T_krit ({T_surf:.1f} °C) > "
        f"Taupunkt ({T_tau:.1f} °C): Schimmel entsteht vor "
        f"dem sichtbaren Kondensieren.</small>"
        f"</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  BLOCK 4 — Schimmelgrenz-Tabelle
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Schimmelgrenze  φ_i → T_krit")
st.markdown(
    f"<small style='color:#666;'>Bei T_r = {T_r:.1f} °C: "
    f"Ab welcher φ_i wird T_krit kritisch?</small>",
    unsafe_allow_html=True
)

tbl_data = []
for rf in range(30, 75, 5):
    ts    = critical_surface_temp(T_r, float(rf))
    f_rsi = calc_f_rsi_min(ts, T_r, T_a)
    tbl_data.append({
        "φ_i [%]":       rf,
        "T_krit [°C]":   f"{ts:.1f}",
        "f_Rsi,min":     f"{f_rsi:.3f}",
        "Bewertung":     "🔴 kritisch" if ts > T_r - 2
                         else ("⚠️ knapp" if ts > T_r - 5
                               else "✅ unkritisch"),
    })

import pandas as pd
st.dataframe(pd.DataFrame(tbl_data),
             use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
#  BLOCK 5 — Umkehr-Analyse
# ══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔄 Umkehr-Analyse")
st.markdown(
    "Bei bekannter Bauteil-Oberflächentemperatur aus einem "
    "Wärmebrücken-Nachweis: **Wie hoch darf φ_i maximal sein?**"
)

T_wb = st.number_input(
    "T_Oberfläche aus Wärmebrücken-Nachweis [°C]",
    min_value=-20.0, max_value=35.0,
    value=float(round(T_surf, 1)),
    step=0.1, format="%.1f"
)

rf_max = max_rf_for_surface_temp(T_wb, T_r)
f_rsi_wb = calc_f_rsi_min(T_wb, T_r, T_a)

if rf_max >= 60:
    box_col, ico = "#1A7A42", "✅"
    bg_col = "linear-gradient(135deg,#F0FFF4,#D5F5E3)"
    bc_col = "#1A7A42"
elif rf_max >= 45:
    box_col, ico = "#9A7D0A", "⚠️"
    bg_col = "linear-gradient(135deg,#FEF9E7,#FDEBD0)"
    bc_col = "#CA6F1E"
else:
    box_col, ico = "#C0392B", "🔴"
    bg_col = "linear-gradient(135deg,#FDEDEC,#F9CCCA)"
    bc_col = "#C0392B"

st.markdown(
    f"<div style='background:{bg_col};border:2px solid {bc_col};"
    f"border-radius:10px;padding:16px;margin:8px 0;'>"
    f"<b style='color:{box_col};font-size:1.1em;'>"
    f"{ico}  T_Oberfläche = {T_wb:.1f} °C</b><br><br>"
    f"Maximale zulässige Innenraumfeuchte:<br>"
    f"<span style='font-size:1.4em;color:{box_col};'>"
    f"<b>φ_i,max = {rf_max:.1f} %</b></span><br><br>"
    f"Solange φ_i unter <b>{rf_max:.1f} %</b> bleibt, "
    f"wird an dieser Oberfläche ({T_wb:.1f} °C) "
    f"keine φ_Oberfläche ≥ 80 % erreicht → kein Schimmelrisiko.<br><br>"
    f"<small style='color:#666;'>"
    f"Zugehöriger f_Rsi: {f_rsi_wb:.3f}<br>"
    f"Formel: φ_i,max = p_sat({T_wb:.1f} °C) × 80% "
    f"/ p_sat({T_r:.1f} °C) × 100"
    f"</small>"
    f"</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown(
    "<div class='footer'>"
    "🌡️ Mollier h-x Diagramm  |  Bauphysik  |  "
    "Made by Mahmoud Alhayek  |  Wetzel & von Seht  |  "
    "DIN 4108-2 Beiblatt 2"
    "</div>",
    unsafe_allow_html=True
)
