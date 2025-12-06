import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@dataclass
class PointLoad:
    P: float
    x: float

@dataclass
class PointMoment:
    M0: float
    x: float

@dataclass
class UDL:
    w: float
    a: float
    b: float

@dataclass
class LinearLoad:
    w1: float
    w2: float
    a: float
    b: float

class BeamModel:
    def __init__(self, L: float, beam_type: str, xA: float = 0.0, xB: float = None):
        self.L = float(L)
        self.beam_type = beam_type
        self.xA = float(xA)
        self.xB = float(xB) if xB is not None else None
        self.point_loads: List[PointLoad] = []
        self.point_moments: List[PointMoment] = []
        self.udls: List[UDL] = []
        self.linear_loads: List[LinearLoad] = []
        self.VA: Optional[float] = None
        self.VB: Optional[float] = None
        self.V_fixed: Optional[float] = None
        self.M_fixed: Optional[float] = None

    @staticmethod
    def _udl_resultant(udl: UDL) -> Tuple[float, float]:
        R = udl.w * (udl.b - udl.a)
        xbar = udl.a + 0.5 * (udl.b - udl.a)
        return R, xbar

    @staticmethod
    def _linear_resultant(ll: LinearLoad) -> Tuple[float, float]:
        a, b, w1, w2 = ll.a, ll.b, ll.w1, ll.w2
        Ls = b - a
        if Ls == 0:
            return 0.0, a
        k = (w2 - w1) / Ls
        R = Ls * (w1 + w2) / 2.0
        x_moment = (
            a * w1 * Ls + a * k * (Ls**2) / 2.0 + w1 * (Ls**2) / 2.0 + k * (Ls**3) / 3.0
        )
        xbar = x_moment / R if R != 0 else (a + Ls/2)
        return R, xbar

    def add_point_load(self, P: float, x: float):
        self.point_loads.append(PointLoad(P, x))

    def add_point_moment(self, M0: float, x: float):
        self.point_moments.append(PointMoment(M0, x))

    def add_udl(self, w: float, a: float, b: float):
        self.udls.append(UDL(w, a, b))

    def add_linear_load(self, w1: float, w2: float, a: float, b: float):
        self.linear_loads.append(LinearLoad(w1, w2, a, b))

    def compute_reactions(self):
        W_point = sum(pl.P for pl in self.point_loads)
        R_udl = 0.0; M_udl_A = 0.0
        for udl in self.udls:
            R, xb = self._udl_resultant(udl)
            R_udl += R
            if self.beam_type != "Cantilever":
                M_udl_A += R * (xb - self.xA)
        R_lin = 0.0; M_lin_A = 0.0
        for ll in self.linear_loads:
            R, xb = self._linear_resultant(ll)
            R_lin += R
            if self.beam_type != "Cantilever":
                M_lin_A += R * (xb - self.xA)
        W_total = W_point + R_udl + R_lin
        M_points_total = sum(pm.M0 for pm in self.point_moments)

        if self.beam_type in ("Simply Supported", "Overhanging"):
            M_point_P_about_A = sum(pl.P * (pl.x - self.xA) for pl in self.point_loads)
            RHS = M_point_P_about_A + M_udl_A + M_lin_A + M_points_total
            span = (self.xB - self.xA)
            self.VB = RHS / span
            self.VA = W_total - self.VB
            self.V_fixed = None; self.M_fixed = None
        elif self.beam_type == "Cantilever":
            self.V_fixed = W_total
            M_point_P_fixed = sum(pl.P * pl.x for pl in self.point_loads)
            M_udl_fixed = sum(self._udl_resultant(udl)[0] * self._udl_resultant(udl)[1] for udl in self.udls)
            M_lin_fixed = sum(self._linear_resultant(ll)[0] * self._linear_resultant(ll)[1] for ll in self.linear_loads)
            self.M_fixed = M_point_P_fixed + M_udl_fixed + M_lin_fixed + M_points_total
            self.VA = None; self.VB = None
        else:
            raise ValueError("Unsupported beam type.")

    def shear_and_moment(self, n_points: int = 2001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0.0, self.L, n_points)
        self.compute_reactions()
        V = np.zeros_like(x)
        if self.beam_type in ("Simply Supported", "Overhanging"):
            V += (x >= self.xA) * self.VA
            V += (x >= self.xB) * self.VB
        elif self.beam_type == "Cantilever":
            V += (x >= 0.0) * self.V_fixed
        for pl in self.point_loads:
            V -= (x >= pl.x) * pl.P
        for udl in self.udls:
            a, b, w = udl.a, udl.b, udl.w
            left = (x >= a)
            Lx = np.clip(x - a, 0.0, b - a)
            V -= left * w * Lx
        for ll in self.linear_loads:
            a, b, w1, w2 = ll.a, ll.b, ll.w1, ll.w2
            Ls = b - a
            if Ls <= 0:
                continue
            k = (w2 - w1) / Ls
            left = (x >= a)
            Lx = np.clip(x - a, 0.0, Ls)
            V -= left * (w1 * Lx + 0.5 * k * Lx**2)
        M = np.zeros_like(x)
        dx = x[1] - x[0]
        for i in range(1, len(x)):
            M[i] = M[i-1] + 0.5 * (V[i-1] + V[i]) * dx
        for pm in self.point_moments:
            idx = np.searchsorted(x, pm.x, side='left')
            if idx < len(x):
                M[idx:] += pm.M0
        if self.beam_type in ("Simply Supported", "Overhanging"):
            idxA = np.searchsorted(x, self.xA, side='left')
            M -= M[idxA]
        if self.beam_type == "Cantilever":
            M += self.M_fixed
        return x, V, M

import numpy as np

def compute_extrema(x: np.ndarray, V: np.ndarray, M: np.ndarray):
    idxV = int(np.argmax(np.abs(V)))
    maxV = V[idxV]; x_maxV = x[idxV]
    idxM = int(np.argmax(np.abs(M)))
    maxM = M[idxM]; x_maxM = x[idxM]
    x_zero = None
    for i in range(len(x)-1):
        if V[i] == 0.0:
            x_zero = x[i]; break
        if V[i] * V[i+1] < 0.0:
            dx = x[i+1] - x[i]
            x_zero = x[i] - V[i] * dx / (V[i+1] - V[i])
            break
    return maxV, x_maxV, maxM, x_maxM, x_zero

st.set_page_config(page_title="Determinate Beam – Shear & Moment (Web)", layout="wide")
st.title("Determinate Beam – Shear & Moment Calculator (Web)")

with st.sidebar:
    st.header("Beam Setup")
    beam_type = st.selectbox("Beam Type", ["Simply Supported", "Overhanging", "Cantilever"], index=0)
    L = st.number_input("Total Length L [m]", min_value=0.1, value=12.0, step=0.1)
    if beam_type == "Cantilever":
        xA = 0.0
        xB = None
        st.info("Cantilever: fixed at x=0. Roller not applicable.")
    else:
        xA = st.number_input("xA [m]", min_value=0.0, max_value=L, value=0.0, step=0.1)
        xB = st.number_input("xB [m]", min_value=xA + 0.1, max_value=L, value=L, step=0.1)
    n_points = st.number_input("Samples (>=501)", min_value=501, value=2001, step=100)

beam = BeamModel(L, beam_type, xA, xB)

st.sidebar.markdown("---")
st.sidebar.header("Add Loads")
col1, col2 = st.sidebar.columns(2)
with col1:
    pl_P = st.number_input("Point Load P [kN] (down)", value=30.0, step=1.0)
with col2:
    pl_x = st.number_input("Point Load x [m]", min_value=0.0, max_value=L, value=5.0, step=0.1)
if st.sidebar.button("Add Point Load"):
    beam.add_point_load(pl_P, pl_x)

col3, col4 = st.sidebar.columns(2)
with col3:
    pm_M = st.number_input("Point Moment M0 [kN·m] (+jump)", value=20.0, step=1.0)
with col4:
    pm_x = st.number_input("Point Moment x [m]", min_value=0.0, max_value=L, value=9.0, step=0.1)
if st.sidebar.button("Add Point Moment"):
    beam.add_point_moment(pm_M, pm_x)

st.sidebar.markdown("—")
col5, col6, col7 = st.sidebar.columns(3)
with col5:
    udl_w = st.number_input("UDL w [kN/m]", value=6.0, step=0.5)
with col6:
    udl_a = st.number_input("UDL a [m]", min_value=0.0, max_value=L, value=2.0, step=0.1)
with col7:
    udl_b = st.number_input("UDL b [m]", min_value=0.0, max_value=L, value=10.0, step=0.1)
if st.sidebar.button("Add UDL"):
    if udl_b <= udl_a:
        st.sidebar.error("Require b > a for UDL")
    else:
        beam.add_udl(udl_w, udl_a, udl_b)

st.sidebar.markdown("—")
col8, col9, col10, col11 = st.sidebar.columns(4)
with col8:
    ll_w1 = st.number_input("Linear w1 [kN/m]", value=0.0, step=0.5)
with col9:
    ll_w2 = st.number_input("Linear w2 [kN/m]", value=12.0, step=0.5)
with col10:
    ll_a = st.number_input("Linear a [m]", min_value=0.0, max_value=L, value=0.0, step=0.1)
with col11:
    ll_b = st.number_input("Linear b [m]", min_value=0.0, max_value=L, value=4.0, step=0.1)
if st.sidebar.button("Add Linear Load"):
    if ll_b <= ll_a:
        st.sidebar.error("Require b > a for Linear Load")
    else:
        beam.add_linear_load(ll_w1, ll_w2, ll_a, ll_b)

if "loads_state" not in st.session_state:
    st.session_state.loads_state = {"PL": [], "PM": [], "UDL": [], "LIN": []}

for pl in beam.point_loads:
    st.session_state.loads_state["PL"].append(pl)
for pm in beam.point_moments:
    st.session_state.loads_state["PM"].append(pm)
for ud in beam.udls:
    st.session_state.loads_state["UDL"].append(ud)
for ll in beam.linear_loads:
    st.session_state.loads_state["LIN"].append(ll)

beam.point_loads = st.session_state.loads_state["PL"]
beam.point_moments = st.session_state.loads_state["PM"]
beam.udls = st.session_state.loads_state["UDL"]
beam.linear_loads = st.session_state.loads_state["LIN"]

st.subheader("Current Loads")
if len(beam.point_loads)+len(beam.point_moments)+len(beam.udls)+len(beam.linear_loads) == 0:
    st.info("No loads added yet. Use the sidebar to add loads.")
else:
    table_rows = []
    for pl in beam.point_loads:
        table_rows.append(("Point Load", f"P={pl.P:.2f} kN", f"x={pl.x:.2f} m"))
    for pm in beam.point_moments:
        table_rows.append(("Point Moment", f"M0={pm.M0:.2f} kN·m", f"x={pm.x:.2f} m"))
    for ud in beam.udls:
        table_rows.append(("UDL", f"w={ud.w:.2f} kN/m", f"a={ud.a:.2f} m", f"b={ud.b:.2f} m"))
    for ll in beam.linear_loads:
        table_rows.append(("Linear", f"w1={ll.w1:.2f} kN/m", f"w2={ll.w2:.2f} kN/m", f"a={ll.a:.2f} m", f"b={ll.b:.2f} m"))
    st.table(table_rows)

colA, colB = st.columns([1,1])
with colA:
    if st.button("Clear All Loads"):
        st.session_state.loads_state = {"PL": [], "PM": [], "UDL": [], "LIN": []}
        st.experimental_rerun()

with colB:
    compute_clicked = st.button("Compute & Plot")

if compute_clicked:
    try:
        x, V, M = beam.shear_and_moment(n_points=int(n_points))
    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.stop()

    if beam.beam_type in ("Simply Supported", "Overhanging"):
        st.success(f"Reactions: VA={beam.VA:.3f} kN at xA={beam.xA:.2f} m; VB={beam.VB:.3f} kN at xB={beam.xB:.2f} m")
    else:
        st.success(f"Reactions: V_fixed={beam.V_fixed:.3f} kN at x=0; M_fixed={beam.M_fixed:.3f} kN·m")

    maxV, x_maxV, maxM, x_maxM, x_zero = compute_extrema(x, V, M)
    ztxt = f"{x_zero:.3f} m (from left)" if x_zero is not None else "— (no zero-shear in span)"
    st.info(
        f"Extrema: max|V|={abs(maxV):.3f} kN @ x={x_maxV:.3f} m (V={maxV:.3f} kN); "
        f"max|M|={abs(maxM):.3f} kN·m @ x={x_maxM:.3f} m (M={maxM:.3f} kN·m); "
        f"Zero-shear at x={ztxt}"
    )

    fig, (axBeam, axV, axM) = plt.subplots(3, 1, figsize=(10.5, 8.0))
    L = beam.L
    axBeam.plot([0, L], [0, 0], color="black", lw=6, solid_capstyle='round')
    if beam.beam_type in ("Simply Supported", "Overhanging"):
        axBeam.plot([beam.xA], [-0.15], marker="^", markersize=12, color="#444")
        axBeam.text(beam.xA, 0.28, "A (Pin)", ha='center', va='bottom', fontsize=9, color="#333")
        axBeam.plot([beam.xB], [-0.18], marker="o", markersize=10, color="#444")
        axBeam.text(beam.xB, 0.28, "B (Roller)", ha='center', va='bottom', fontsize=9, color="#333")
    elif beam.beam_type == "Cantilever":
        axBeam.add_patch(plt.Rectangle((-0.11, -0.22), 0.22, 0.22, facecolor="#444", edgecolor="#222"))
        axBeam.text(0.0, 0.28, "Fixed", ha='center', va='bottom', fontsize=9, color="#333")

    max_intensity = max(([ud.w for ud in beam.udls] + [abs(ll.w1) for ll in beam.linear_loads] + [abs(ll.w2) for ll in beam.linear_loads] + [1.0]))
    k_udl = 0.15 / max_intensity

    for pl in beam.point_loads:
        y_top = 0.45; y_bot = y_top - 0.25
        axBeam.annotate("", xy=(pl.x, y_bot), xytext=(pl.x, y_top), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
        axBeam.text(pl.x, y_top + 0.05, f"{pl.P:.1f} kN", ha='center', va='bottom', fontsize=8, color="tab:blue")

    for pm in beam.point_moments:
        axBeam.text(pm.x, 0.85, f"{pm.M0:.1f} kN·m", ha='center', va='bottom', fontsize=8, color="tab:red")
        axBeam.plot([pm.x-0.1, pm.x+0.1], [0.55, 0.55], color="tab:red", lw=2)

    for udl in beam.udls:
        axBeam.plot([udl.a, udl.b], [0.6, 0.6], color="#2a9d8f", lw=6, alpha=0.25, solid_capstyle='butt')
        n = max(3, int((udl.b - udl.a) * 6))
        xs = np.linspace(udl.a, udl.b, n)
        for xi in xs:
            length = max(0.10, udl.w * k_udl)
            axBeam.annotate("", xy=(xi, 0.55 - length), xytext=(xi, 0.55), arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.5))
        axBeam.text((udl.a+udl.b)/2, 0.68, f"UDL {udl.w:.1f} kN/m", ha='center', va='bottom', fontsize=8, color="#2a9d8f")

    for ll in beam.linear_loads:
        n = max(3, int((ll.b - ll.a) * 6))
        xs = np.linspace(ll.a, ll.b, n)
        for xi in xs:
            wi = ll.w1 + (ll.w2 - ll.w1) * ((xi - ll.a)/(ll.b - ll.a))
            length = max(0.08, wi * k_udl)
            axBeam.annotate("", xy=(xi, 0.50 - length), xytext=(xi, 0.50), arrowprops=dict(arrowstyle="->", color="#e76f51", lw=1.4))
        y0 = 0.62
        h1 = max(0.05, ll.w1 * k_udl * 1.2); h2 = max(0.05, ll.w2 * k_udl * 1.2)
        axBeam.add_patch(plt.Polygon([[ll.a, y0], [ll.a, y0 + h1], [ll.b, y0 + h2], [ll.b, y0]], closed=True, facecolor="#e76f51", alpha=0.15, edgecolor="#e76f51"))
        axBeam.text((ll.a+ll.b)/2, y0 + max(h1, h2) + 0.02, f"Linear {ll.w1:.1f}→{ll.w2:.1f} kN/m", ha='center', va='bottom', fontsize=8, color="#e76f51")

    axBeam.set_xlim(-0.5, L + 0.5); axBeam.set_ylim(-0.8, 1.0)
    axBeam.set_xticks(np.linspace(0, L, 9)); axBeam.set_yticks([])
    axBeam.grid(alpha=0.15); axBeam.set_ylabel("schematic")
    axBeam.set_title("Beam View (supports & loadings)")

    axV.plot(x, V, color="tab:blue", lw=1.8)
    axV.set_ylabel("V [kN]"); axV.grid(alpha=0.3)
    axV.set_title("Shear Force V(x) [kN]")
    axV.plot([x_maxV], [maxV], marker="o", color="k")

    if x_zero is not None:
        axV.axvline(x_zero, ls="--", color="#777")
        axV.text(x_zero, 0.02, "V=0", rotation=90, va="bottom", ha="right", fontsize=8)

    axM.plot(x, M, color="tab:red", lw=1.8)
    axM.set_ylabel("M [kN·m]"); axM.set_xlabel("x [m]"); axM.grid(alpha=0.3)
    axM.set_title("Bending Moment M(x) [kN·m]")
    axM.plot([x_maxM], [maxM], marker="o", color="k")

    fig.tight_layout()
    st.pyplot(fig)

    csv_enable = st.checkbox("Enable CSV export")
    if csv_enable:
        import pandas as pd
        df = pd.DataFrame({"x [m]": x, "V [kN]": V, "M [kN·m]": M})
        st.download_button("Download x,V,M CSV", df.to_csv(index=False), file_name="beam_V_M.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Notes**  ")
st.markdown("- Determinate beams use ΣM=0, ΣFx=0, ΣFy=0.")
st.markdown("- Downward loads positive; V upward positive; positive point moment causes upward jump in M.")
