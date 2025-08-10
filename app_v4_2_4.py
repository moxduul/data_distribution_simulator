#!/usr/bin/env python3
# streamlit run app_v4_2_4.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from scipy.stats import anderson
import ast, textwrap, re

st.set_page_config(page_title="Data Distribution Simulator 2.4.2", layout="wide")

# ---------- App header ----------
st.title("Data Distribution Simulator")
st.caption("Generate, transform, and combine distributions. Inspect normality (Shapiro/D’Agostino), "
           "Anderson–Darling, histograms with bell-curve overlays, and Q–Q plots.")

# ---------- Constants ----------
DIST_TYPES = ["Normal", "Uniform", "Lognormal", "Weibull", "Student's t"]
LIGHT_RED = "#f8d7da"   # for boxes and table cells (p<0.05)
LIGHT_RED_BG = "#fdecea"  # subtle axes background when p<0.05

ALLOWED_FUNCS = {
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "power": np.power,
    "clip": np.clip,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "where": np.where,
}
ALLOWED_CONSTS = {"pi": np.pi, "e": np.e}

# ---------- Helpers ----------
def normality_pvalue(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return ("insufficient_n", np.nan)
    if 3 <= n <= 5000:
        stat, p = stats.shapiro(x)
        return ("shapiro", float(p))
    else:
        stat, p = stats.normaltest(x)
        return ("dagostino_k2", float(p))

def compute_stats(data: np.ndarray):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = data.size
    mu = float(np.mean(data)) if n else np.nan
    sigma = float(np.std(data, ddof=1)) if n > 1 else np.nan

    # Anderson-Darling
    if n >= 3:
        ad = anderson(data, dist="norm")
        ad_stat = float(ad.statistic)
    else:
        ad_stat = np.nan

    test_name, p_val = normality_pvalue(data)
    return {
        "n": n,
        "mean": mu,
        "std": sigma,
        "ad_stat": ad_stat,
        "p_test": test_name,
        "p_value": p_val,
    }

def truncated_sample(draw_fn, n, low, high, max_iter=50):
    """Rejection sampling to truncate to [low, high]. Returns (array, warning_or_none)."""
    if low is None and high is None:
        return draw_fn(n), None
    if (low is not None) and (high is not None) and not (high > low):
        return np.array([], dtype=float), "Invalid bounds: high must be > low."

    need = n
    accepted = []
    it = 0
    chunk = max(1000, n)
    while need > 0 and it < max_iter:
        x = draw_fn(chunk)
        if low is not None:
            x = x[x >= low]
        if high is not None:
            x = x[x <= high]
        if x.size:
            take = int(min(need, x.size))
            accepted.append(x[:take])
            need -= take
        it += 1

    data = np.concatenate(accepted) if accepted else np.array([], dtype=float)
    warn = None
    if data.size < n:
        warn = f"Only generated {data.size} / {n} samples within bounds after {it} iterations. Consider loosening bounds."
    return data, warn

def generate_data(rng, dist_type: str, params: dict, n: int, low=None, high=None):
    """Generate data and apply optional truncation. Returns (array, warning_or_none)."""
    if n < 1:
        return np.array([], dtype=float), None

    if dist_type == "Normal":
        mu = params.get("mu", 0.0)
        sd = params.get("sd", 1.0)
        if sd <= 0:
            return np.array([], dtype=float), "σ must be > 0."
        draw = lambda m: rng.normal(loc=mu, scale=sd, size=m)

    elif dist_type == "Uniform":
        lo = params.get("low", 0.0)
        hi = params.get("high", 1.0)
        if not (hi > lo):
            return np.array([], dtype=float), "Uniform requires high > low."
        draw = lambda m: rng.uniform(lo, hi, size=m)

    elif dist_type == "Lognormal":
        mu_log = params.get("mu_log", 0.0)
        sd_log = params.get("sd_log", 0.5)
        if sd_log <= 0:
            return np.array([], dtype=float), "σ_log must be > 0."
        draw = lambda m: rng.lognormal(mean=mu_log, sigma=sd_log, size=m)

    elif dist_type == "Weibull":
        k = params.get("k", 1.5)      # shape
        lam = params.get("lam", 1.0)  # scale
        if k <= 0 or lam <= 0:
            return np.array([], dtype=float), "Weibull requires k>0 and λ>0."
        draw = lambda m: rng.weibull(a=k, size=m) * lam

    elif dist_type == "Student's t":
        df = params.get("df", 5.0)
        mu = params.get("mu", 0.0)
        sd = params.get("sd", 1.0)
        if df <= 0 or sd <= 0:
            return np.array([], dtype=float), "Student's t requires ν>0 and σ>0."
        draw = lambda m: mu + sd * rng.standard_t(df=df, size=m)

    else:
        return np.array([], dtype=float), None

    return truncated_sample(draw, n, low, high)

def plot_row(ax_hist, ax_qq, label, data, mu, sigma, p_val):
    # Conditional light background if p<0.05
    highlight = (pd.notna(p_val) and p_val < 0.05)
    if highlight:
        ax_hist.set_facecolor(LIGHT_RED_BG)
        ax_qq.set_facecolor(LIGHT_RED_BG)

    # Histogram + bell curve
    if data.size:
        ax_hist.hist(data, bins="auto", density=True, alpha=0.6, edgecolor="black")
        if not np.isnan(mu) and not np.isnan(sigma) and sigma > 0:
            lo = np.nanmin(data); hi = np.nanmax(data)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                xs = np.linspace(lo, hi, 200)
                pdf = stats.norm.pdf(xs, loc=mu, scale=sigma)
                ax_hist.plot(xs, pdf, linestyle="--", linewidth=2)
    ax_hist.set_title(f"{label} — Histogram + Bell Curve")
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Density")

    # Annotation with conditional light-red background if p<0.05
    p_text = "NA" if np.isnan(p_val) else f"{p_val:.3g}"
    mu_text = "NA" if np.isnan(mu) else f"{mu:.6g}"
    sd_text = "NA" if np.isnan(sigma) else f"{sigma:.6g}"
    ann = f"μ={mu_text}\nσ={sd_text}\np={p_text}"
    bbox_kwargs = dict(boxstyle="round", alpha=0.2)
    if highlight:
        bbox_kwargs["facecolor"] = LIGHT_RED
    ax_hist.text(0.97, 0.97, ann, ha="right", va="top", transform=ax_hist.transAxes, bbox=bbox_kwargs)

    # Q-Q Plot
    if data.size:
        stats.probplot(data, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"{label} — Q-Q Plot")

def parse_optional_float(txt):
    txt = (txt or "").strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None

def op_descriptor(op_key: str, params: dict):
    """Create a short descriptor string for file names/labels."""
    def fmt(v):
        if v is None: return "NA"
        try:
            f = float(v)
            return f"{f:.4g}"
        except Exception:
            return str(v)
    if op_key == "power":
        return f"power_p{fmt(params.get('p', 2.0))}"
    if op_key == "affine":
        return f"affine_a{fmt(params.get('a',1.0))}_b{fmt(params.get('b',0.0))}"
    if op_key == "clip":
        return f"clip_{fmt(params.get('low', None))}_{fmt(params.get('high', None))}"
    return op_key

# ---------- Safe Expression Parser ----------
def normalize_expr(expr: str) -> str:
    # Replace non-breaking spaces & unicode minus/multiply/divide, caret power
    expr = expr.replace("\u00A0", " ")
    expr = expr.replace("−", "-").replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    # Remove accidental newlines/indentation
    expr = textwrap.dedent(expr).strip()
    # Collapse multiple spaces
    expr = re.sub(r"\s+", " ", expr)
    # If "y=" or "Y=" prefix, strip it out (tolerate spaces)
    if "=" in expr:
        left, right = expr.split("=", 1)
        if left.strip().lower() in ("y", "y()"):
            expr = right.strip()
    return expr

class SafeEval(ast.NodeVisitor):
    def __init__(self, variables, funcs, consts):
        self.variables = variables
        self.funcs = funcs
        self.consts = consts
        self.used_names = set()

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Num):  # py<3.8 legacy
            return float(node.n)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")
        elif isinstance(node, ast.Name):
            name = node.id
            self.used_names.add(name)
            if name in self.variables:
                return self.variables[name]
            if name in self.consts:
                return self.consts[name]
            if name in self.funcs:
                return self.funcs[name]
            raise ValueError(f"Unknown symbol: {name}")
        elif isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")
        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return np.power(left, right)
            raise ValueError("Unsupported binary operator.")
        elif isinstance(node, ast.Call):
            func = self.visit(node.func)
            if func not in self.funcs.values():
                raise ValueError("Only whitelisted functions are allowed.")
            args = [self.visit(a) for a in node.args]
            kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        else:
            raise ValueError("Unsupported expression element.")

def eval_expression(expr_text, variables, funcs, consts):
    expr_text = normalize_expr(expr_text)
    if not expr_text:
        raise ValueError("Empty expression.")
    try:
        tree = ast.parse(expr_text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e.msg}")
    se = SafeEval(variables, funcs, consts)
    try:
        result = se.visit(tree)
    except ZeroDivisionError:
        # Shape it like the first variable if division by zero occurs
        first_arr = next(iter(variables.values()))
        result = np.full_like(first_arr, np.nan, dtype=float)
    return result, se.used_names, expr_text

# ---------- Single tab selector ----------
tab_show, tab_transform, tab_combine = st.tabs(["Show", "Transform", "Combine"])

# ===== Show Tab =====
with tab_show:
    st.header("Show")
    st.caption("Create A–D, then view summary stats and plots.")

    with st.sidebar:
        st.header("Create distributions")
        st.header("Global Settings")
        seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
        # Keep integer type; no float format to avoid Streamlit warning
        n_global = st.number_input("Sample size n", min_value=1, value=500, step=1)

        st.header("Distributions (A–D)")

        def dist_controls(label, defaults):
            st.subheader(f"Distribution {label}")
            dist = st.selectbox(f"Type ({label})", DIST_TYPES, index=DIST_TYPES.index(defaults["type"]), key=f"{label}_type")
            params = {}

            if dist == "Normal":
                c1, c2 = st.columns(2)
                with c1:
                    params["mu"] = st.number_input(f"μ ({label})", value=float(defaults.get("mu", 0.0)), key=f"{label}_mu", format="%.9g")
                with c2:
                    params["sd"] = st.number_input(f"σ ({label})", min_value=0.0, value=float(defaults.get("sd", 1.0)), key=f"{label}_sd", format="%.9g")

            elif dist == "Uniform":
                c1, c2 = st.columns(2)
                with c1:
                    params["low"] = st.number_input(f"low ({label})", value=float(defaults.get("low", 0.0)), key=f"{label}_low", format="%.9g")
                with c2:
                    params["high"] = st.number_input(f"high ({label})", value=float(defaults.get("high", 1.0)), key=f"{label}_high", format="%.9g")

            elif dist == "Lognormal":
                c1, c2 = st.columns(2)
                with c1:
                    params["mu_log"] = st.number_input(f"μ_log ({label})", value=float(defaults.get("mu_log", 0.0)), key=f"{label}_mu_log", format="%.9g")
                with c2:
                    params["sd_log"] = st.number_input(f"σ_log ({label})", min_value=0.0, value=float(defaults.get("sd_log", 0.5)), key=f"{label}_sd_log", format="%.9g")

            elif dist == "Weibull":
                c1, c2 = st.columns(2)
                with c1:
                    params["k"] = st.number_input(f"k shape ({label})", min_value=0.0, value=float(defaults.get("k", 1.5)), key=f"{label}_k", format="%.9g")
                with c2:
                    params["lam"] = st.number_input(f"λ scale ({label})", min_value=0.0, value=float(defaults.get("lam", 1.0)), key=f"{label}_lam", format="%.9g")

            elif dist == "Student's t":
                c1, c2, c3 = st.columns(3)
                with c1:
                    params["df"] = st.number_input(f"ν df ({label})", min_value=0.1, value=float(defaults.get("df", 5.0)), key=f"{label}_df", format="%.9g")
                with c2:
                    params["mu"] = st.number_input(f"μ ({label})", value=float(defaults.get("mu", 0.0)), key=f"{label}_t_mu", format="%.9g")
                with c3:
                    params["sd"] = st.number_input(f"σ ({label})", min_value=0.0, value=float(defaults.get("sd", 1.0)), key=f"{label}_t_sd", format="%.9g")

            with st.expander(f"Optional truncation bounds ({label})"):
                lb_txt = st.text_input(f"Lower bound (optional, {label})", value="", key=f"{label}_lb")
                ub_txt = st.text_input(f"Upper bound (optional, {label})", value="", key=f"{label}_ub")
            low = parse_optional_float(lb_txt)
            high = parse_optional_float(ub_txt)

            return dist, params, low, high

        # Default all distributions to Normal(0,1)
        defaults = {
            "A": {"type": "Normal", "mu": 0.0, "sd": 1.0},
            "B": {"type": "Normal", "mu": 0.0, "sd": 1.0},
            "C": {"type": "Normal", "mu": 0.0, "sd": 1.0},
            "D": {"type": "Normal", "mu": 0.0, "sd": 1.0},
        }

        choices = {}
        for label in ["A", "B", "C", "D"]:
            dist, params, low, high = dist_controls(label, defaults[label])
            choices[label] = {"type": dist, "params": params, "low": low, "high": high}

        run = st.button("Generate / Update")

    if run:
        rng = np.random.default_rng(int(seed))

        data_dict = {}
        meta = {}
        warnings = []
        for label in ["A", "B", "C", "D"]:
            dtype = choices[label]["type"]
            params = choices[label]["params"]
            low = choices[label]["low"]
            high = choices[label]["high"]
            data, warn = generate_data(rng, dtype, params, int(n_global), low=low, high=high)
            data_dict[label] = data
            meta[label] = {"type": dtype, **params, "low": low, "high": high}
            if warn:
                warnings.append(f"{label}: {warn}")

        st.session_state["data_dict"] = data_dict
        st.session_state["meta"] = meta
        st.session_state["n_global"] = int(n_global)

        if warnings:
            st.warning(" • ".join(warnings))

    # Results area
    if "data_dict" in st.session_state:
        st.divider()
        st.subheader("Results")

        show_tbl = st.checkbox("Show summary table", value=True)
        show_plots = st.checkbox("Show combined plots", value=True)

        data_dict = st.session_state["data_dict"]
        meta = st.session_state["meta"]

        # Table
        if show_tbl:
            rows = []
            for label, data in data_dict.items():
                s = compute_stats(data)
                rows.append({
                    "Distribution": label,
                    "Type": meta[label]["type"],
                    "n": s["n"],
                    "mean": s["mean"],
                    "std": s["std"],
                    "p_test": s["p_test"],
                    "p_value": s["p_value"],
                    "AD_stat": s["ad_stat"],
                    "Low": meta[label]["low"],
                    "High": meta[label]["high"],
                })
            stats_df = pd.DataFrame(rows).set_index("Distribution")

            def highlight_p(series):
                return ['background-color: %s' % LIGHT_RED if (pd.notna(v) and v < 0.05) else '' for v in series]

            styled = (stats_df
                      .style
                      .format({"mean": "{:.6g}", "std": "{:.6g}", "p_value": "{:.3g}", "AD_stat": "{:.3g}"})
                      .apply(highlight_p, subset=["p_value"]))
            st.dataframe(styled, use_container_width=True)

        if show_plots:
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
            labels = ["A", "B", "C", "D"]
            for i, label in enumerate(labels):
                data = data_dict[label]
                s = compute_stats(data)
                ax_hist = axes[i, 0]
                ax_qq = axes[i, 1]

                trunc = ""
                low = meta[label]["low"]
                high = meta[label]["high"]
                if low is not None or high is not None:
                    lo_txt = "-∞" if low is None else f"{low:g}"
                    hi_txt = "+∞" if high is None else f"{high:g}"
                    trunc = f" | trunc[{lo_txt},{hi_txt}]"
                pretty_label = f"{label} ({meta[label]['type']}){trunc}"

                plot_row(ax_hist, ax_qq, pretty_label, data, s["mean"], s["std"], s["p_value"])

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # Download all
        labels = ["A", "B", "C", "D"]
        df_out = pd.DataFrame({lbl: data_dict.get(lbl, np.array([], dtype=float)) for lbl in labels})
        st.download_button("Download ALL generated data (CSV)", data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="generated_distributions_AD.csv", mime="text/csv")

# ===== Transform Tab (Single | All) =====
def transform_data(x: np.ndarray, op_key: str, params: dict, invalid_policy: str, epsilon: float):
    """Return (x_original, y_transformed, info). 'Drop invalid' keeps x intact; y gets NaNs at invalid positions."""
    x = np.asarray(x, dtype=float)
    info = {"dropped": 0, "replaced": 0, "warnings": []}

    def nan_like(ref):
        arr = np.empty_like(ref, dtype=float)
        arr[:] = np.nan
        return arr

    if op_key == "reciprocal":
        mask = np.abs(x) < epsilon
        if invalid_policy == "Drop invalid":
            y = nan_like(x)
            valid = ~mask
            y[valid] = 1.0 / x[valid]
            cnt = int(np.sum(mask))
            if cnt:
                info["dropped"] += cnt
                info["warnings"].append(f"Reciprocal: {cnt} values with |x|<ε set to NaN in transformed output (original kept).")
            return x, y, info
        else:
            safe_x = np.where(mask, np.sign(x) * epsilon, x)
            y = 1.0 / safe_x
            cnt = int(np.sum(mask))
            if cnt:
                info["replaced"] += cnt
                info["warnings"].append(f"Reciprocal: {cnt} near-zero values replaced using ε in transformed output (original kept).")
            return x, y, info

    elif op_key == "square":
        y = x ** 2
        return x, y, info

    elif op_key == "abs":
        y = np.abs(x)
        return x, y, info

    elif op_key == "log":
        mask = x <= 0
        if invalid_policy == "Drop invalid":
            y = nan_like(x)
            valid = ~mask
            y[valid] = np.log(x[valid])
            cnt = int(np.sum(mask))
            if cnt:
                info["dropped"] += cnt
                info["warnings"].append(f"log: {cnt} values with x<=0 set to NaN in transformed output (original kept).")
            return x, y, info
        else:
            safe_x = np.where(mask, epsilon, x)
            y = np.log(safe_x)
            cnt = int(np.sum(mask))
            if cnt:
                info["replaced"] += cnt
                info["warnings"].append(f"log: {cnt} values with x<=0 replaced using ε in transformed output (original kept).")
            return x, y, info

    elif op_key == "log10":
        mask = x <= 0
        if invalid_policy == "Drop invalid":
            y = nan_like(x)
            valid = ~mask
            y[valid] = np.log10(x[valid])
            cnt = int(np.sum(mask))
            if cnt:
                info["dropped"] += cnt
                info["warnings"].append(f"log10: {cnt} values with x<=0 set to NaN in transformed output (original kept).")
            return x, y, info
        else:
            safe_x = np.where(mask, epsilon, x)
            y = np.log10(safe_x)
            cnt = int(np.sum(mask))
            if cnt:
                info["replaced"] += cnt
                info["warnings"].append(f"log10: {cnt} values with x<=0 replaced using ε in transformed output (original kept).")
            return x, y, info

    elif op_key == "exp":
        y = np.exp(x)
        mask = ~np.isfinite(y)
        if np.any(mask):
            cnt = int(np.sum(mask))
            if invalid_policy == "Drop invalid":
                y[mask] = np.nan
                info["dropped"] += cnt
                info["warnings"].append(f"exp: {cnt} overflow values set to NaN in transformed output (original kept).")
            else:
                maxf = np.finfo(float).max
                y[mask] = np.sign(y[mask]) * maxf
                info["replaced"] += cnt
                info["warnings"].append(f"exp: {cnt} overflow values clamped in transformed output (original kept).")
        return x, y, info

    elif op_key == "power":
        p = float(params.get("p", 2.0))
        is_int = np.isclose(p, round(p))
        if is_int:
            y = x ** int(round(p))
            return x, y, info
        else:
            mask = x < 0
            y = nan_like(x)
            if np.any(~mask):
                y[~mask] = np.power(x[~mask], p)
            cnt = int(np.sum(mask))
            if cnt:
                info["dropped"] += cnt
                info["warnings"].append(f"power p={p:g}: {cnt} negatives set to NaN (non-integer power). Original kept.")
            return x, y, info

    elif op_key == "zscore":
        mu = np.mean(x) if x.size else 0.0
        sd = np.std(x, ddof=1) if x.size > 1 else 0.0
        if sd <= 0:
            y = np.zeros_like(x)
            info["warnings"].append("z-score: standard deviation is 0; returning zeros.")
            return x, y, info
        y = (x - mu) / sd
        return x, y, info

    elif op_key == "affine":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        y = a * x + b
        return x, y, info

    elif op_key == "clip":
        lo = params.get("low", None)
        hi = params.get("high", None)
        lo_v = -np.inf if lo is None else float(lo)
        hi_v = np.inf if hi is None else float(hi)
        if (lo is not None) and (hi is not None) and not (hi > lo):
            y = x.copy()
            info["warnings"].append("clip: high must be > low; ignoring clip.")
        else:
            y = np.clip(x, lo_v, hi_v)
        return x, y, info

    else:
        return x, x.copy(), {"warnings": ["Unknown operation."], "dropped": 0, "replaced": 0}

# ===== Transform Tab =====
with tab_transform:
    st.header("Transform")
    st.caption("Apply the same operation to one or all distributions. 'Drop invalid' keeps originals intact; transformed output has NaNs where invalid.")

    if "data_dict" not in st.session_state:
        st.info("Please create A–D in the Show tab first.")
        st.stop()

    data_dict = st.session_state["data_dict"]

    # Apply to toggle
    mode = st.radio("Apply to:", ["Single", "All"], horizontal=True)

    # Shared operation selection
    st.subheader("Operation & parameters")
    ops = {
        "reciprocal": "Reciprocal (1/x)",
        "square": "Square (x^2)",
        "abs": "Absolute value |x|",
        "log": "Natural log ln(x)",
        "log10": "Log base 10 log10(x)",
        "exp": "Exponential exp(x)",
        "power": "Power x^p",
        "zscore": "Z-score (x-μ)/σ",
        "affine": "Affine a*x + b",
        "clip": "Clip to [low, high]",
    }
    op_key = st.selectbox("Operation", list(ops.keys()), format_func=lambda k: ops[k], index=0)

    params = {}
    if op_key == "power":
        params["p"] = st.number_input("Exponent p", value=2.0, step=0.1)
    elif op_key == "affine":
        c1, c2 = st.columns(2)
        with c1:
            params["a"] = st.number_input("a (scale)", value=1.0, step=0.1)
        with c2:
            params["b"] = st.number_input("b (offset)", value=0.0, step=0.1)
    elif op_key == "clip":
        c1, c2 = st.columns(2)
        with c1:
            lo_txt = st.text_input("Clip low (optional)", value="")
        with c2:
            hi_txt = st.text_input("Clip high (optional)", value="")
        params["low"] = parse_optional_float(lo_txt)
        params["high"] = parse_optional_float(hi_txt)

    st.markdown("**Invalid handling**")
    colA, colB = st.columns([1,1])
    with colA:
        invalid_policy = st.radio("When operation is invalid (e.g., 1/0, log≤0):",
                                  ["Drop invalid", "Replace (epsilon/clamp)"], index=0)
    with colB:
        epsilon = st.number_input("Epsilon ε (for near-zero or lower-bound replacements)",
                                  min_value=0.0, value=1e-8, format="%.1e")

    if mode == "Single":
        # Pick one
        labels = ["A", "B", "C", "D"]
        label = st.selectbox("Source distribution", labels, index=0)
        run_tf = st.button("Run transform")
        if run_tf:
            x = data_dict.get(label, np.array([], dtype=float))
            x_orig, y, info = transform_data(x, op_key, params, invalid_policy, float(epsilon))

            s_before = compute_stats(x_orig)
            s_after = compute_stats(y)

            # Warnings
            msgs = []
            if info.get("dropped", 0) > 0:
                msgs.append(f"{label}: {info['dropped']} values marked NaN in transformed output")
            if info.get("replaced", 0) > 0:
                msgs.append(f"{label}: Replaced {info['replaced']} values")
            if info.get("warnings"):
                msgs.extend(info["warnings"])
            if msgs:
                st.warning(" • ".join(msgs))

            # Table
            rows = [
                {"Stage": f"{label} (before)", "n": s_before["n"], "mean": s_before["mean"], "std": s_before["std"],
                 "p_test": s_before["p_test"], "p_value": s_before["p_value"], "AD_stat": s_before["ad_stat"]},
                {"Stage": f"{label} → {ops[op_key]} (after)", "n": s_after["n"], "mean": s_after["mean"], "std": s_after["std"],
                 "p_test": s_after["p_test"], "p_value": s_after["p_value"], "AD_stat": s_after["ad_stat"]},
            ]
            tbl = pd.DataFrame(rows).set_index("Stage")

            def highlight_p(series):
                return ['background-color: %s' % LIGHT_RED if (pd.notna(v) and v < 0.05) else '' for v in series]

            styled = (tbl
                      .style
                      .format({"mean": "{:.6g}", "std": "{:.6g}", "p_value": "{:.3g}", "AD_stat": "{:.3g}"})
                      .apply(highlight_p, subset=["p_value"]))

            st.subheader("Summary Statistics (Before vs After)")
            st.dataframe(styled, use_container_width=True)

            # Plots
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            plot_row(axes[0,0], axes[0,1], f"{label} (before)", x_orig,
                     s_before["mean"], s_before["std"], s_before["p_value"])
            plot_row(axes[1,0], axes[1,1], f"{label} → {ops[op_key]} (after)", y,
                     s_after["mean"], s_after["std"], s_after["p_value"])
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            # Download CSV
            out_name = f"{label}_{op_key}"
            df_out = pd.DataFrame({label: x_orig, out_name: y})
            st.download_button("Download (before & after) CSV",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name=f"{label}_transform_{op_descriptor(op_key, params)}.csv",
                               mime="text/csv")

    else:  # mode == "All"
        run_all = st.button("Run transform for ALL (A–D)")
        if run_all:
            labels = ["A", "B", "C", "D"]
            results = {}
            warnings = []
            for lbl in labels:
                x = data_dict.get(lbl, np.array([], dtype=float))
                x_orig, y, info = transform_data(x, op_key, params, invalid_policy, float(epsilon))
                s_before = compute_stats(x_orig)
                s_after = compute_stats(y)
                results[lbl] = {
                    "x_before": x_orig,
                    "y_after": y,
                    "s_before": s_before,
                    "s_after": s_after,
                    "info": info
                }
                msg = []
                if info.get("dropped", 0) > 0:
                    msg.append(f"{lbl}: {info['dropped']} values marked NaN in transformed output")
                if info.get("replaced", 0) > 0:
                    msg.append(f"{lbl}: Replaced {info['replaced']} values")
                if info.get("warnings"):
                    msg.extend([f"{lbl}: {w}" for w in info["warnings"]])
                if msg:
                    warnings.append(" • ".join(msg))

            st.session_state["all_results"] = {
                "labels": labels,
                "results": results,
                "op_key": op_key,
                "params": params,
                "invalid_policy": invalid_policy,
                "epsilon": float(epsilon),
                "warnings": warnings,
            }

        # Use persisted results if available
        has_results = "all_results" in st.session_state
        if not has_results:
            st.info("Run the transform for ALL to see results.")
        else:
            AR = st.session_state["all_results"]
            labels = AR["labels"]; results = AR["results"]; warnings = AR["warnings"]
            if warnings:
                st.warning(" | ".join(warnings))

            # Aggregated summary table (per distribution, before & after)
            rows = []
            for lbl in labels:
                sb = results[lbl]["s_before"]
                sa = results[lbl]["s_after"]
                rows.append({
                    "Distribution": lbl,
                    "n_before": sb["n"], "n_after": sa["n"],
                    "mean_before": sb["mean"], "mean_after": sa["mean"],
                    "std_before": sb["std"], "std_after": sa["std"],
                    "p_before": sb["p_value"], "p_after": sa["p_value"],
                    "AD_before": sb["ad_stat"], "AD_after": sa["ad_stat"],
                })
            agg_df = pd.DataFrame(rows).set_index("Distribution")

            def highlight_p_cols(df_):
                styles = pd.DataFrame('', index=df_.index, columns=df_.columns)
                for col in ["p_before", "p_after"]:
                    if col in df_.columns:
                        styles.loc[df_[col] < 0.05, col] = f'background-color: {LIGHT_RED}'
                return styles

            styled_agg = (agg_df
                          .style
                          .format({
                              "mean_before": "{:.6g}", "mean_after": "{:.6g}",
                              "std_before": "{:.6g}", "std_after": "{:.6g}",
                              "p_before": "{:.3g}", "p_after": "{:.3g}",
                              "AD_before": "{:.3g}", "AD_after": "{:.3g}"
                          })
                          .apply(highlight_p_cols, axis=None))

            st.subheader("Summary (All distributions)")
            st.dataframe(styled_agg, use_container_width=True)

            # Plots
            mode_plots = st.radio("Plot view:", ["After only (grid)", "Before & after"], horizontal=True, index=1)
            if mode_plots == "After only (grid)":
                fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
                for i, lbl in enumerate(labels):
                    sa = results[lbl]["s_after"]
                    y = results[lbl]["y_after"]
                    plot_row(axes[i,0], axes[i,1], f"{lbl} → {ops[AR['op_key']]} (after)",
                             y, sa["mean"], sa["std"], sa["p_value"])
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                # Before & after: choose one or render all
                col1, col2 = st.columns([1,1])
                with col1:
                    which = st.selectbox("Choose a distribution to preview", labels, index=0)
                with col2:
                    render_all = st.checkbox("Render ALL before/after (may be slow)", value=False)

                if render_all:
                    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(12, 32))
                    row = 0
                    for lbl in labels:
                        sb = results[lbl]["s_before"]; sa = results[lbl]["s_after"]
                        xb = results[lbl]["x_before"]; ya = results[lbl]["y_after"]
                        plot_row(axes[row,0], axes[row,1], f"{lbl} (before)",
                                 xb, sb["mean"], sb["std"], sb["p_value"])
                        plot_row(axes[row+1,0], axes[row+1,1], f"{lbl} → {ops[AR['op_key']]} (after)",
                                 ya, sa["mean"], sa["std"], sa["p_value"])
                        row += 2
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                else:
                    sb = results[which]["s_before"]; sa = results[which]["s_after"]
                    xb = results[which]["x_before"]; ya = results[which]["y_after"]
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
                    plot_row(axes[0,0], axes[0,1], f"{which} (before)",
                             xb, sb["mean"], sb["std"], sb["p_value"])
                    plot_row(axes[1,0], axes[1,1], f"{which} → {ops[AR['op_key']]} (after)",
                             ya, sa["mean"], sa["std"], sa["p_value"])
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)

            # Downloads
            # Build wide CSV with padded columns (lengths may differ across distributions)
            max_len = 0
            cols = {}
            for lbl in labels:
                xb = results[lbl]["x_before"]
                ya = results[lbl]["y_after"]
                max_len = max(max_len, len(xb), len(ya))
            idx = pd.RangeIndex(max_len)
            for lbl in labels:
                xb = pd.Series(results[lbl]["x_before"], index=pd.RangeIndex(len(results[lbl]["x_before"]))).reindex(idx)
                ya = pd.Series(results[lbl]["y_after"], index=pd.RangeIndex(len(results[lbl]["y_after"]))).reindex(idx)
                cols[lbl] = xb.values
                cols[f"{lbl}_{AR['op_key']}"] = ya.values
            df_wide = pd.DataFrame(cols)
            st.download_button("Download ALL (wide) CSV",
                               data=df_wide.to_csv(index=False).encode("utf-8"),
                               file_name=f"all_transform_{op_descriptor(AR['op_key'], AR['params'])}.csv",
                               mime="text/csv")

            # Individual CSVs
            st.markdown("**Individual downloads**")
            for lbl in labels:
                pair = pd.DataFrame({
                    lbl: results[lbl]["x_before"],
                    f"{lbl}_{AR['op_key']}": results[lbl]["y_after"]
                })
                st.download_button(f"Download {lbl} pair CSV",
                                   data=pair.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{lbl}_transform_{op_descriptor(AR['op_key'], AR['params'])}.csv",
                                   mime="text/csv")

# ===== Combine Tab =====
with tab_combine:
    st.header("Combine")
    st.caption("Create a new series from A–D using a safe custom expression. Operators: + - * / ** ( ), functions: abs, exp, log, log10, sqrt, power, clip, minimum, maximum, sin, cos, tan, where. Constants: pi, e, and 'eps'. '^' is accepted as power.")

    if "data_dict" not in st.session_state:
        st.info("Please create A–D in the Show tab first.")
        st.stop()

    data_dict = st.session_state["data_dict"]

    col1, col2 = st.columns([2,1])
    with col1:
        expr_default = "(8*A*B*0.0000001/ (pi*C**4))*pi*D**2"
        expr_text = st.text_input("Expression", value=expr_default)
        st.caption("Example: " + expr_default)
    with col2:
        out_name = st.text_input("Output name", value="Y")

    st.markdown("**Advanced options**")
    c1, c2, c3 = st.columns(3)
    with c1:
        align_policy = st.selectbox("If input lengths differ:", ["Error", "Truncate to min length", "Pad with NaN"], index=0)
    with c2:
        invalid_result_policy = st.selectbox("When result is NaN/Inf:", ["Mark as NaN (recommended)", "Drop invalid rows"], index=0)
    with c3:
        eps_val = st.number_input("eps (ε) for your formulas", min_value=0.0, value=1e-8, format="%.3e")

    run_expr = st.button("Run custom function")

    if run_expr:
        # Prepare variables and constants
        vars_available = {k: np.asarray(v, dtype=float) for k, v in data_dict.items()}
        consts = dict(ALLOWED_CONSTS)
        consts["eps"] = float(eps_val)

        # Evaluate expression safely
        try:
            y_raw, used_names, norm_expr = eval_expression(expr_text, vars_available, ALLOWED_FUNCS, consts)
        except Exception as e:
            st.error(f"Parse/eval error: {e}")
            st.stop()

        # Determine used elements
        used_vars = [name for name in used_names if name in vars_available]
        used_funcs = [name for name in used_names if name in ALLOWED_FUNCS]
        used_consts = [name for name in used_names if name in consts]

        if not used_vars:
            st.error("Your expression did not reference any of A, B, C, or D.")
            st.stop()

        # Align inputs if lengths differ
        lengths = [len(vars_available[v]) for v in used_vars]
        all_equal = all(L == lengths[0] for L in lengths)
        idx = None
        if not all_equal:
            if align_policy == "Error":
                st.error(f"Input lengths differ for {used_vars}: {lengths}. Choose a different alignment policy.")
                st.stop()
            elif align_policy == "Truncate to min length":
                idx = slice(0, min(lengths))
            elif align_policy == "Pad with NaN":
                max_len = max(lengths)
                for v in used_vars:
                    arr = vars_available[v]
                    if len(arr) < max_len:
                        pad = np.full(max_len - len(arr), np.nan)
                        vars_available[v] = np.concatenate([arr, pad])
                idx = slice(0, max_len)

            # re-evaluate with aligned inputs
            try:
                y_raw, used_names, norm_expr = eval_expression(expr_text, vars_available, ALLOWED_FUNCS, consts)
            except Exception as e:
                st.error(f"Re-eval error after alignment: {e}")
                st.stop()

        # If sliced, also slice inputs
        if isinstance(idx, slice):
            for v in used_vars:
                vars_available[v] = vars_available[v][idx]

        # Ensure array output
        y_raw = np.asarray(y_raw, dtype=float).astype(float)

        # Handle invalid results
        finite_mask = np.isfinite(y_raw)
        n_invalid = int(np.sum(~finite_mask))
        if invalid_result_policy == "Mark as NaN (recommended)":
            y = y_raw.copy()
            y[~finite_mask] = np.nan
            if n_invalid > 0:
                st.warning(f"{n_invalid} non-finite results marked as NaN (inputs kept unchanged).")
            y_plot = y
            df_cols = {v: vars_available[v] for v in used_vars}
            df_cols[out_name] = y
            df = pd.DataFrame(df_cols)
        else:
            keep = finite_mask
            if not np.any(keep):
                st.error("All results are non-finite; nothing to show.")
                st.stop()
            if n_invalid > 0:
                st.warning(f"Dropped {n_invalid} rows with non-finite results.")
            y_plot = y_raw[keep]
            df_cols = {v: vars_available[v][keep] for v in used_vars}
            df_cols[out_name] = y_plot
            df = pd.DataFrame(df_cols)
            y = y_plot

        # Stats
        s = compute_stats(y_plot)

        # Summary
        rows = [{
            "Series": out_name, "n": s["n"], "mean": s["mean"], "std": s["std"],
            "p_test": s["p_test"], "p_value": s["p_value"], "AD_stat": s["ad_stat"],
            "Used vars": ", ".join(used_vars),
            "Funcs": ", ".join(used_funcs) if used_funcs else "-",
            "Expr": norm_expr
        }]
        tbl = pd.DataFrame(rows).set_index("Series")

        def highlight_p(series):
            return ['background-color: %s' % LIGHT_RED if (pd.notna(v) and v < 0.05) else '' for v in series]

        styled = (tbl
                  .style
                  .format({"mean": "{:.6g}", "std": "{:.6g}", "p_value": "{:.3g}", "AD_stat": "{:.3g}"})
                  .apply(highlight_p, subset=["p_value"]))

        st.subheader("Result summary")
        st.dataframe(styled, use_container_width=True)

        # Plots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
        plot_row(axes[0], axes[1], out_name, y_plot, s["mean"], s["std"], s["p_value"])
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Download
        st.download_button("Download result CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"custom_{out_name}.csv", mime="text/csv")

        st.caption(f"Variables used: {', '.join(used_vars)} | Functions: {', '.join(used_funcs) if used_funcs else 'none'} | Constants: {', '.join(used_consts) if used_consts else 'none'}")
