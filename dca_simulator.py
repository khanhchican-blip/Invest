# ============================================================
# CÀI ĐẶT THƯ VIỆN (chạy lệnh này trong terminal / Colab):
# pip install streamlit plotly pandas numpy yfinance requests
# pip install vnstock3   (tuỳ chọn – lấy dữ liệu chứng khoán VN)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CẤU HÌNH TRANG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mô phỏng Đầu tư Định kỳ (DCA)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
h1 { font-weight: 800 !important; font-size: 1.9rem !important; color: #0F172A !important; }
h2 { font-weight: 700 !important; color: #1E3A5F !important; }
h3 { font-weight: 600 !important; color: #1E3A5F !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600; font-size: 0.88rem;
}
.metric-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    border-radius: 14px; padding: 16px 20px;
    color: white; text-align: center; margin-bottom: 8px;
}
.metric-card .val { font-size: 1.65rem; font-weight: 800; }
.metric-card .lbl { font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }
.warn-box {
    background: #FEF3C7; border-left: 4px solid #F59E0B;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.85rem; color: #92400E; margin: 8px 0;
}
.info-box {
    background: #EFF6FF; border-left: 4px solid #3B82F6;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.85rem; color: #1E40AF; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HẰNG SỐ & CẤU HÌNH KÊNH ĐẦU TƯ
# ════════════════════════════════════════════════════════════
CHANNELS = {
    "🏦 Tiết kiệm ngân hàng": {
        "mu": 0.063,   # lợi nhuận kỳ vọng/năm
        "sigma": 0.005, # độ lệch chuẩn/năm
        "color": "#10B981",
        "ticker_yf": None,
        "ticker_vn": None,
        "desc": "Lãi suất ổn định ~5.5–7%/năm, rủi ro gần bằng 0",
    },
    "📊 Chứng khoán VN (ETF E1VFVN30)": {
        "mu": 0.135,
        "sigma": 0.22,
        "color": "#EF4444",
        "ticker_yf": "E1VFVN30.BK",
        "ticker_vn": "E1VFVN30",
        "desc": "VN-Index / VN30 ETF, kỳ vọng ~12–15%/năm",
    },
    "🇺🇸 Chứng khoán Mỹ (S&P 500 – VOO)": {
        "mu": 0.11,
        "sigma": 0.17,
        "color": "#3B82F6",
        "ticker_yf": "VOO",
        "ticker_vn": None,
        "desc": "S&P 500 ETF, kỳ vọng ~10–12%/năm (USD)",
    },
    "🥇 Vàng (GLD)": {
        "mu": 0.08,
        "sigma": 0.14,
        "color": "#F59E0B",
        "ticker_yf": "GLD",
        "ticker_vn": None,
        "desc": "Vàng thế giới quy đổi, kỳ vọng ~7–9%/năm",
    },
    "₿ Tiền mã hóa (Bitcoin)": {
        "mu": 0.60,
        "sigma": 0.80,
        "color": "#8B5CF6",
        "ticker_yf": "BTC-USD",
        "ticker_vn": None,
        "desc": "Bitcoin, kỳ vọng cao nhưng biến động rất lớn",
    },
}

RISK_FREE_RATE = 0.05  # lãi suất phi rủi ro (5%/năm)

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Chuyển mã màu HEX sang chuỗi rgba() mà Plotly chấp nhận."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ════════════════════════════════════════════════════════════
# CLASS: DataLoader
# ════════════════════════════════════════════════════════════
class DataLoader:
    """Tải dữ liệu giá lịch sử từ yfinance hoặc fallback sang dữ liệu mẫu."""

    @staticmethod
    def load_yfinance(ticker: str, start: str, end: str) -> pd.Series | None:
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end,
                             interval="1mo", progress=False, auto_adjust=True)
            if df.empty:
                return None
            close = df["Close"].squeeze()
            close.index = pd.to_datetime(close.index).to_period("M").to_timestamp()
            return close.dropna()
        except Exception:
            return None

    @staticmethod
    def load_vnstock(ticker: str, start: str, end: str) -> pd.Series | None:
        try:
            from vnstock3 import Vnstock
            stk = Vnstock().stock(symbol=ticker, source="VCI")
            df = stk.quote.history(start=start, end=end, interval="1M")
            if df is None or df.empty:
                return None
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            return df["close"].dropna()
        except Exception:
            return None

    @staticmethod
    def generate_fallback(mu: float, sigma: float,
                          n_months: int, seed: int = 42) -> pd.Series:
        """Sinh chuỗi giá giả lập khi không có dữ liệu thực."""
        rng = np.random.default_rng(seed)
        monthly_r = mu / 12
        monthly_s = sigma / np.sqrt(12)
        returns = rng.normal(monthly_r, monthly_s, n_months)
        price = 100 * np.cumprod(1 + returns)
        idx = pd.date_range(
            end=pd.Timestamp.today().replace(day=1),
            periods=n_months, freq="MS"
        )
        return pd.Series(price, index=idx)

    def get_price_series(self, channel_key: str, start: str, end: str,
                         n_months: int) -> tuple[pd.Series, bool]:
        """Trả về (series giá, is_real_data)."""
        cfg = CHANNELS[channel_key]

        # Thử vnstock
        if cfg["ticker_vn"]:
            series = self.load_vnstock(cfg["ticker_vn"], start, end)
            if series is not None and len(series) >= 3:
                return series, True

        # Thử yfinance
        if cfg["ticker_yf"]:
            series = self.load_yfinance(cfg["ticker_yf"], start, end)
            if series is not None and len(series) >= 3:
                return series, True

        # Fallback
        return self.generate_fallback(cfg["mu"], cfg["sigma"], n_months), False


# ════════════════════════════════════════════════════════════
# CLASS: DCASimulator
# ════════════════════════════════════════════════════════════
class DCASimulator:
    """Mô phỏng chiến lược DCA theo 2 chế độ: Backtesting & Monte Carlo."""

    def __init__(self, monthly_invest: float):
        self.monthly_invest = monthly_invest  # VND

    # ── Backtesting ────────────────────────────────────────
    def backtest(self, price_series: pd.Series,
                 alloc_frac: float = 1.0) -> pd.DataFrame:
        """DCA với dữ liệu lịch sử thực."""
        prices = price_series.values
        n = len(prices)
        portfolio_val = np.zeros(n)
        units = 0.0
        invested = 0.0

        for i in range(n):
            monthly = self.monthly_invest * alloc_frac
            units += monthly / prices[i]
            invested += monthly
            portfolio_val[i] = units * prices[i]

        df = pd.DataFrame({
            "date": price_series.index[:n],
            "portfolio": portfolio_val,
            "invested": np.cumsum([self.monthly_invest * alloc_frac] * n),
        })
        return df.set_index("date")

    # ── Monte Carlo ────────────────────────────────────────
    def monte_carlo(self, mu: float, sigma: float,
                    n_months: int, n_sims: int = 500,
                    alloc_frac: float = 1.0,
                    black_swan_prob: float = 0.04,
                    black_swan_range: tuple = (-0.35, -0.15),
                    black_swan_duration: int = 2) -> dict:
        """
        Geometric Brownian Motion + Jump Diffusion (Black Swan).
        Trả về dict chứa ma trận portfolio và các phân vị.
        """
        monthly_mu    = mu / 12
        monthly_sigma = sigma / np.sqrt(12)
        rng = np.random.default_rng(None)

        all_portfolios = np.zeros((n_sims, n_months))

        for s in range(n_sims):
            units, portfolio_val = 0.0, np.zeros(n_months)
            price = 100.0  # giá chuẩn hoá
            swan_countdown = 0

            for t in range(n_months):
                # Black Swan event
                if swan_countdown > 0:
                    shock = rng.uniform(black_swan_range[0], black_swan_range[1]) / black_swan_duration
                    price *= (1 + shock)
                    swan_countdown -= 1
                elif rng.random() < black_swan_prob / 12:
                    shock = rng.uniform(black_swan_range[0], black_swan_range[1]) / black_swan_duration
                    price *= (1 + shock)
                    swan_countdown = black_swan_duration - 1
                else:
                    # GBM
                    ret = rng.normal(monthly_mu - 0.5 * monthly_sigma**2,
                                     monthly_sigma)
                    price *= np.exp(ret)

                price = max(price, 0.01)
                monthly = self.monthly_invest * alloc_frac
                units += monthly / price
                portfolio_val[t] = units * price

            all_portfolios[s] = portfolio_val

        invested = np.cumsum([self.monthly_invest * alloc_frac] * n_months)
        return {
            "p10":    np.percentile(all_portfolios, 10, axis=0),
            "p25":    np.percentile(all_portfolios, 25, axis=0),
            "p50":    np.percentile(all_portfolios, 50, axis=0),
            "p75":    np.percentile(all_portfolios, 75, axis=0),
            "p90":    np.percentile(all_portfolios, 90, axis=0),
            "invested": invested,
            "all":    all_portfolios,
        }


# ════════════════════════════════════════════════════════════
# CLASS: MetricsVisualizer
# ════════════════════════════════════════════════════════════
class MetricsVisualizer:
    """Tính toán chỉ số hiệu suất và vẽ biểu đồ Plotly."""

    # ── Chỉ số hiệu suất ──────────────────────────────────
    @staticmethod
    def calc_metrics(portfolio: np.ndarray,
                     invested: np.ndarray,
                     n_months: int) -> dict:
        final_val  = portfolio[-1]
        total_inv  = invested[-1]
        abs_return = final_val - total_inv
        cagr = (final_val / total_inv) ** (12 / n_months) - 1 if total_inv > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(portfolio)
        dd   = (portfolio - peak) / np.where(peak > 0, peak, 1)
        mdd  = dd.min()

        # Sharpe (monthly returns)
        monthly_ret = np.diff(portfolio) / np.where(portfolio[:-1] > 0,
                                                      portfolio[:-1], 1)
        rf_monthly  = RISK_FREE_RATE / 12
        excess      = monthly_ret - rf_monthly
        sharpe = (excess.mean() / excess.std() * np.sqrt(12)
                  if excess.std() > 0 else 0)

        return {
            "final_val":  final_val,
            "total_inv":  total_inv,
            "abs_return": abs_return,
            "cagr":       cagr,
            "mdd":        mdd,
            "sharpe":     sharpe,
            "drawdown":   dd,
        }

    # ── Biểu đồ tăng trưởng ───────────────────────────────
    @staticmethod
    def plot_growth(results: dict, channel_names: list,
                    dates, mode: str = "backtest") -> go.Figure:
        fig = go.Figure()

        # Vốn gốc (đường đứt)
        fig.add_trace(go.Scatter(
            x=dates,
            y=results[channel_names[0]]["invested"] / 1e6,
            name="Vốn gốc nạp vào",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            fill=None,
        ))

        for ch in channel_names:
            r   = results[ch]
            cfg = CHANNELS[ch]
            if mode == "backtest":
                fig.add_trace(go.Scatter(
                    x=dates, y=r["portfolio"] / 1e6,
                    name=ch.split(" ")[0] + " " + ch.split(" ")[1] if len(ch.split(" ")) > 1 else ch,
                    line=dict(color=cfg["color"], width=2.5),
                    hovertemplate="%{y:.2f} tr VND<extra>" + ch + "</extra>",
                ))
            else:  # monte carlo
                x_fwd  = dates
                name_s = ch.split(" ")[0]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_fwd, x_fwd[::-1]]),
                    y=np.concatenate([r["p90"] / 1e6, r["p10"][::-1] / 1e6]),
                    fill="toself",
                    fillcolor=hex_to_rgba(cfg["color"], 0.13),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{name_s} (P10–P90)",
                    showlegend=True,
                ))
                fig.add_trace(go.Scatter(
                    x=x_fwd, y=r["p50"] / 1e6,
                    name=f"{name_s} trung vị",
                    line=dict(color=cfg["color"], width=2.5),
                    hovertemplate="%{y:.2f} tr VND<extra>" + ch + "</extra>",
                ))

        fig.update_layout(
            title=dict(text="📈 Tăng trưởng tài sản theo thời gian (triệu VND)",
                       font=dict(size=16, color="#0F172A")),
            xaxis_title="Thời gian",
            yaxis_title="Giá trị (triệu VND)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            hovermode="x unified",
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Inter", size=12),
            margin=dict(t=80, b=40, l=60, r=20),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
        fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        return fig

    # ── Biểu đồ Drawdown ──────────────────────────────────
    @staticmethod
    def plot_drawdown(results: dict, channel_names: list, dates) -> go.Figure:
        fig = go.Figure()
        for ch in channel_names:
            cfg = CHANNELS[ch]
            dd  = results[ch]["drawdown"]
            name_s = " ".join(ch.split(" ")[:2])
            fig.add_trace(go.Scatter(
                x=dates[1:], y=dd * 100,
                name=name_s,
                line=dict(color=cfg["color"], width=2),
                fill="tozeroy",
                fillcolor=hex_to_rgba(cfg["color"], 0.20),
                hovertemplate="%{y:.1f}%<extra>" + ch + "</extra>",
            ))

        fig.update_layout(
            title=dict(text="📉 Mức sụt giảm từ đỉnh (Drawdown %)",
                       font=dict(size=16, color="#0F172A")),
            xaxis_title="Thời gian",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Inter", size=12),
            margin=dict(t=80, b=40, l=60, r=20),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
        fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0",
                         ticksuffix="%")
        return fig

    # ── Bảng so sánh ──────────────────────────────────────
    @staticmethod
    def comparison_table(results: dict, channel_names: list,
                         n_months: int) -> pd.DataFrame:
        rows = []
        for ch in channel_names:
            r = results[ch]
            m = MetricsVisualizer.calc_metrics(
                r["portfolio"], r["invested"], n_months)
            rows.append({
                "Kênh đầu tư": ch,
                "Vốn gốc (tr VND)": f"{m['total_inv']/1e6:,.1f}",
                "Giá trị cuối (tr VND)": f"{m['final_val']/1e6:,.1f}",
                "Lợi nhuận (tr VND)": f"{m['abs_return']/1e6:,.1f}",
                "CAGR (%)": f"{m['cagr']*100:.1f}%",
                "Max Drawdown (%)": f"{m['mdd']*100:.1f}%",
                "Sharpe Ratio": f"{m['sharpe']:.2f}",
            })
        return pd.DataFrame(rows).set_index("Kênh đầu tư")


# ════════════════════════════════════════════════════════════
# SIDEBAR – Cấu hình đầu vào
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình mô phỏng")
    st.divider()

    st.markdown("### 💰 Ngân sách tháng")
    monthly_total = st.number_input(
        "Tổng tiền mỗi tháng (triệu VND)",
        min_value=0.5, max_value=500.0, value=10.0, step=0.5,
        format="%.1f",
        help="Toàn bộ số tiền có sẵn mỗi tháng để đầu tư"
    )
    monthly_vnd = monthly_total * 1_000_000

    st.markdown("### ⏱️ Thời gian mô phỏng")
    n_months = st.number_input(
        "Số tháng mô phỏng",
        min_value=6, max_value=360, value=60, step=6
    )
    n_years = n_months / 12

    st.markdown("### 📂 Kênh đầu tư & Phân bổ")
    selected_channels = st.multiselect(
        "Chọn kênh muốn mô phỏng",
        options=list(CHANNELS.keys()),
        default=list(CHANNELS.keys()),
    )

    st.markdown("**Tỷ lệ phân bổ (tổng = 100%)**")
    allocs = {}
    remaining = 100
    for i, ch in enumerate(selected_channels):
        default_val = round(100 / len(selected_channels)) if selected_channels else 0
        if i == len(selected_channels) - 1:
            val = remaining
            st.slider(ch.split(" ")[0] + " " + ch.split(" ")[1] if len(ch.split()) > 1 else ch,
                      0, 100, val, disabled=True,
                      key=f"alloc_{i}")
        else:
            val = st.slider(
                ch.split(" ")[0] + " " + (ch.split(" ")[1] if len(ch.split()) > 1 else ""),
                0, remaining, min(default_val, remaining),
                key=f"alloc_{i}"
            )
            remaining -= val
            remaining = max(0, remaining)
        allocs[ch] = val / 100

    total_alloc = sum(allocs.values())

    st.divider()
    st.markdown("### 🎛️ Chế độ mô phỏng")
    mode = st.radio(
        "Chọn chế độ",
        ["📜 Backtesting (dữ liệu lịch sử)", "🔮 Monte Carlo (mô phỏng tương lai)"],
        index=0,
    )
    is_backtest = "Backtesting" in mode

    if is_backtest:
        st.markdown("**Khoảng thời gian lịch sử**")
        end_date   = datetime.date.today().replace(day=1)
        start_date = end_date - datetime.timedelta(days=int(n_months * 30.5))
        st.info(f"Từ **{start_date.strftime('%m/%Y')}** đến **{end_date.strftime('%m/%Y')}**")
    else:
        st.markdown("**Tham số Monte Carlo**")
        n_sims        = st.slider("Số lần mô phỏng", 100, 1000, 300, step=50)
        swan_prob     = st.slider("Xác suất Thiên nga đen (%/năm)", 1, 10, 4) / 100
        swan_min      = st.slider("Mức sụt giảm tối thiểu (%)", -50, -10, -35) / 100
        swan_max      = st.slider("Mức sụt giảm tối đa (%)", -30, -5, -15) / 100
        swan_duration = st.slider("Thời gian sốc (tháng)", 1, 6, 2)

    st.divider()
    run_btn = st.button("▶ Chạy mô phỏng", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════
# TIÊU ĐỀ CHÍNH
# ════════════════════════════════════════════════════════════
st.title("📈 Mô phỏng Đầu tư Định kỳ (DCA)")
st.markdown(
    "Công cụ mô phỏng chiến lược **Dollar-Cost Averaging** qua nhiều kênh đầu tư, "
    "so sánh hiệu suất lịch sử và dự báo xác suất tương lai với mô hình Monte Carlo."
)

if not selected_channels:
    st.warning("⚠️ Vui lòng chọn ít nhất một kênh đầu tư ở thanh bên trái.")
    st.stop()

if abs(total_alloc - 1.0) > 0.02:
    st.markdown(
        f"<div class='warn-box'>⚠️ Tổng phân bổ hiện tại là <b>{total_alloc*100:.0f}%</b>. "
        "Điều chỉnh để tổng đạt 100%.</div>",
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════
# CHẠY MÔ PHỎNG
# ════════════════════════════════════════════════════════════
if run_btn:
    loader = DataLoader()
    simulator = DCASimulator(monthly_vnd)
    viz = MetricsVisualizer()
    results_bt  = {}   # backtest
    results_mc  = {}   # monte carlo
    data_source = {}   # is_real?

    progress_bar = st.progress(0, text="Đang tải dữ liệu và chạy mô phỏng…")

    for idx, ch in enumerate(selected_channels):
        cfg   = CHANNELS[ch]
        frac  = allocs.get(ch, 0)
        progress_bar.progress(
            (idx + 0.5) / len(selected_channels),
            text=f"Đang xử lý: {ch}…"
        )

        # ── BACKTEST ────────────────────────────────────────
        if is_backtest:
            series, is_real = loader.get_price_series(
                ch,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                n_months=n_months,
            )
            data_source[ch] = is_real
            n_actual = min(len(series), n_months)
            series   = series.iloc[:n_actual]
            df_bt    = simulator.backtest(series, alloc_frac=frac)
            m        = viz.calc_metrics(
                df_bt["portfolio"].values,
                df_bt["invested"].values,
                n_actual,
            )
            results_bt[ch] = {
                "portfolio": df_bt["portfolio"].values,
                "invested":  df_bt["invested"].values,
                "drawdown":  m["drawdown"],
                "dates":     df_bt.index,
                "metrics":   m,
            }

        # ── MONTE CARLO ─────────────────────────────────────
        else:
            mc = simulator.monte_carlo(
                mu=cfg["mu"], sigma=cfg["sigma"],
                n_months=n_months, n_sims=n_sims,
                alloc_frac=frac,
                black_swan_prob=swan_prob,
                black_swan_range=(swan_min, swan_max),
                black_swan_duration=swan_duration,
            )
            m = viz.calc_metrics(mc["p50"], mc["invested"], n_months)
            mc["drawdown"] = m["drawdown"]
            mc["metrics"]  = m
            mc["portfolio"] = mc["p50"]
            results_mc[ch]  = mc

        progress_bar.progress(
            (idx + 1) / len(selected_channels),
            text=f"✅ Xong: {ch}"
        )

    progress_bar.empty()
    results = results_bt if is_backtest else results_mc

    # ── Ngày chung (ngắn nhất) ──────────────────────────────
    if is_backtest:
        min_len = min(len(r["dates"]) for r in results.values())
        dates_common = list(results[selected_channels[0]]["dates"])[:min_len]
        for ch in results:
            results[ch]["portfolio"] = results[ch]["portfolio"][:min_len]
            results[ch]["invested"]  = results[ch]["invested"][:min_len]
            results[ch]["drawdown"]  = results[ch]["drawdown"][:min_len-1]
    else:
        dates_common = pd.date_range(
            start=pd.Timestamp.today().replace(day=1),
            periods=n_months, freq="MS"
        )
        for ch in results:
            results[ch]["dates"] = dates_common

    # ════════════════════════════════════════════════════════
    # HIỂN THỊ – Thông báo nguồn dữ liệu
    # ════════════════════════════════════════════════════════
    if is_backtest:
        fake_chs = [ch for ch, real in data_source.items() if not real]
        if fake_chs:
            st.markdown(
                f"<div class='info-box'>ℹ️ Không lấy được dữ liệu thực của: "
                f"<b>{', '.join(fake_chs)}</b>. "
                "Đã dùng dữ liệu mô phỏng theo tham số thống kê lịch sử.</div>",
                unsafe_allow_html=True
            )

    # ════════════════════════════════════════════════════════
    # METRIC CARDS – Tổng quan
    # ════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 📊 Tổng quan kết quả")
    total_invested = monthly_vnd * n_months / 1e6

    c0, c1, c2, c3 = st.columns(4)
    c0.markdown(f"""<div class='metric-card'>
        <div class='val'>{total_invested:,.1f} tr</div>
        <div class='lbl'>Tổng vốn gốc nạp vào (VND)</div>
    </div>""", unsafe_allow_html=True)
    c1.markdown(f"""<div class='metric-card'>
        <div class='val'>{monthly_total:,.1f} tr</div>
        <div class='lbl'>Đầu tư mỗi tháng (VND)</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-card'>
        <div class='val'>{n_months} tháng</div>
        <div class='lbl'>Thời gian mô phỏng ({n_years:.1f} năm)</div>
    </div>""", unsafe_allow_html=True)
    mode_label = "Backtesting" if is_backtest else "Monte Carlo"
    c3.markdown(f"""<div class='metric-card'>
        <div class='val'>{mode_label}</div>
        <div class='lbl'>Chế độ mô phỏng</div>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TABS – Biểu đồ & Bảng
    # ════════════════════════════════════════════════════════
    st.divider()
    tab1, tab2, tab3 = st.tabs([
        "📈 Tăng trưởng tài sản",
        "📉 Biểu đồ Drawdown",
        "📋 Bảng so sánh chi tiết",
    ])

    with tab1:
        fig_growth = viz.plot_growth(
            results, selected_channels, dates_common,
            mode="backtest" if is_backtest else "mc"
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        if not is_backtest:
            st.markdown(
                "<div class='info-box'>📌 Vùng bóng = khoảng P10–P90 của tất cả mô phỏng. "
                "Đường đậm = trung vị (P50). Bao gồm các sự kiện thiên nga đen.</div>",
                unsafe_allow_html=True
            )

    with tab2:
        fig_dd = viz.plot_drawdown(results, selected_channels, dates_common)
        st.plotly_chart(fig_dd, use_container_width=True)
        st.markdown(
            "<div class='info-box'>📌 Drawdown = % sụt giảm so với đỉnh trước đó. "
            "Càng âm càng sâu, thể hiện áp lực tâm lý khi đầu tư.</div>",
            unsafe_allow_html=True
        )

    with tab3:
        df_table = viz.comparison_table(results, selected_channels, n_months)
        st.dataframe(
            df_table.style
            .set_table_styles([
                {"selector": "thead th",
                 "props": [("background-color", "#1E3A5F"),
                           ("color", "white"),
                           ("font-weight", "700")]},
                {"selector": "tbody tr:nth-child(even)",
                 "props": [("background-color", "#F1F5F9")]},
            ]),
            use_container_width=True,
        )

        # Mini cards từng kênh
        st.markdown("### 📌 Chi tiết từng kênh")
        cols = st.columns(min(len(selected_channels), 3))
        for i, ch in enumerate(selected_channels):
            m   = results[ch]["metrics"]
            cfg = CHANNELS[ch]
            with cols[i % 3]:
                cagr_str = f"{m['cagr']*100:.1f}%"
                final_str = f"{m['final_val']/1e6:,.1f} tr"
                mdd_str  = f"{m['mdd']*100:.1f}%"
                sharpe_str = f"{m['sharpe']:.2f}"
                icon = ch.split(" ")[0]
                st.markdown(f"""
                <div style='background:white;border:1px solid #E2E8F0;
                border-radius:12px;padding:14px 16px;margin-bottom:10px;
                border-top:4px solid {cfg["color"]}'>
                  <div style='font-weight:700;font-size:0.9rem;
                  color:#0F172A;margin-bottom:8px'>{icon} {" ".join(ch.split(" ")[1:])}</div>
                  <div style='font-size:0.8rem;color:#475569;line-height:1.8'>
                    💰 Giá trị cuối: <b style='color:{cfg["color"]}'>{final_str} VND</b><br>
                    📈 CAGR: <b>{cagr_str}</b><br>
                    📉 Max Drawdown: <b style='color:#EF4444'>{mdd_str}</b><br>
                    ⚖️ Sharpe Ratio: <b>{sharpe_str}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # PHÂN BỔ DANH MỤC – Biểu đồ tròn
    # ════════════════════════════════════════════════════════
    if len(selected_channels) > 1:
        st.divider()
        st.markdown("## 🥧 Phân bổ danh mục")
        fig_pie = go.Figure(go.Pie(
            labels=[" ".join(ch.split(" ")[:2]) for ch in selected_channels],
            values=[allocs[ch] * 100 for ch in selected_channels],
            marker_colors=[CHANNELS[ch]["color"] for ch in selected_channels],
            hole=0.42,
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        fig_pie.update_layout(
            title="Tỷ lệ phân bổ ngân sách đầu tư",
            paper_bgcolor="#F8FAFC",
            font=dict(family="Inter"),
            margin=dict(t=60, b=20, l=20, r=20),
            height=350,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    # ── Màn hình chờ ──────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#64748B'>
        <div style='font-size:4rem;margin-bottom:16px'>📊</div>
        <h3 style='color:#1E3A5F'>Sẵn sàng mô phỏng</h3>
        <p>Cấu hình tham số ở thanh bên trái, sau đó nhấn <b>▶ Chạy mô phỏng</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 Hướng dẫn nhanh")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""**1️⃣ Thiết lập ngân sách**
- Nhập tổng tiền đầu tư/tháng
- Chọn thời gian mô phỏng (tháng)""")
    with c2:
        st.markdown("""**2️⃣ Chọn kênh & phân bổ**
- Chọn kênh đầu tư muốn so sánh
- Điều chỉnh tỷ lệ % cho từng kênh""")
    with c3:
        st.markdown("""**3️⃣ Chọn chế độ mô phỏng**
- **Backtesting:** dùng dữ liệu lịch sử thực
- **Monte Carlo:** dự báo tương lai xác suất""")

    st.divider()
    st.markdown("### 📚 Các kênh đầu tư được hỗ trợ")
    for ch, cfg in CHANNELS.items():
        st.markdown(
            f"**{ch}** — {cfg['desc']} "
            f"| Kỳ vọng: **{cfg['mu']*100:.0f}%/năm** "
            f"| Biến động (σ): **{cfg['sigma']*100:.0f}%**"
        )

# ── Miễn trừ trách nhiệm ───────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Tuyên bố miễn trừ trách nhiệm:** Công cụ này chỉ mang mục đích giáo dục và tham khảo. "
    "Kết quả mô phỏng không đảm bảo lợi nhuận thực tế trong tương lai. "
    "Hiệu suất quá khứ không phản ánh kết quả tương lai. "
    "Mọi quyết định đầu tư cần được tư vấn bởi chuyên gia tài chính có chuyên môn."
)
