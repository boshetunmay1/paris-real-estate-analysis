import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# НАСТРОЙКИ СТРАНИЦЫ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Paris Housing Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1rem; opacity: 0.75; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card .label { font-size: 0.75rem; color: #8a9bb0; text-transform: uppercase;
                          letter-spacing: 0.8px; font-weight: 600; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #1a1a2e; margin-top: 4px; }
    .metric-card .sub   { font-size: 0.8rem; color: #6b7a8d; margin-top: 2px; }

    .result-box {
        background: linear-gradient(135deg, #0f3460, #533483);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .result-box .price { font-size: 2.8rem; font-weight: 700; letter-spacing: -1px; }
    .result-box .label { font-size: 0.9rem; opacity: 0.75; margin-bottom: 0.5rem; }

    .info-tag {
        display: inline-block;
        background: #f0f4ff;
        color: #3b5bdb;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }

    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/ParisHousing_clean.csv")
    ts = pd.read_csv("data/time_series.csv")
    return df, ts

@st.cache_resource
def load_lgbm():
    model       = joblib.load("models/lightgbm_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, feature_cols

@st.cache_resource
def fit_arima(_ts):
    m = ARIMA(_ts["avg_price"], order=(0, 0, 1)).fit()
    return m

try:
    df, ts = load_data()
    lgbm_model, feature_cols = load_lgbm()
    arima_model = fit_arima(ts)
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Ошибка загрузки файлов: {e}\n\nПроверь структуру папок: data/ и models/")
    st.stop()


# ─────────────────────────────────────────────
# ШАПКА
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 Paris Housing Analytics</h1>
    <p>Анализ рынка недвижимости · LightGBM · ARIMA · 101 объект · 17 признаков</p>
</div>
""", unsafe_allow_html=True)

# Метрики верхнего уровня
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Объектов в выборке", "101",         "наблюдений"),
    (c2, "Средняя цена",       "4.43M €",     "по выборке"),
    (c3, "Мин. цена",          "22 671 €",    "объект"),
    (c4, "Макс. цена",         "9.94M €",     "объект"),
    (c5, "R² модели",          "≈ 1.000",     "LightGBM"),
]
for col, label, value, sub in metrics:
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ВКЛАДКИ
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Анализ данных",
    "🏠  Оценить квартиру",
    "📈  Прогноз рынка",
])

sns.set_style("whitegrid")
FMT = mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 1 — EDA
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Исследовательский анализ данных")

    # ── Распределение price ──────────────────────────────────
    st.markdown("#### Распределение целевой переменной — цена (€)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(df["price"], bins=25, color="#0f3460", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["price"].mean(),   color="#e94560", linestyle="--", lw=2, label="Среднее")
    axes[0].axvline(df["price"].median(), color="#f5a623", linestyle=":",  lw=2, label="Медиана")
    axes[0].set_title("Гистограмма price", fontweight="bold")
    axes[0].set_xlabel("Цена (€)"); axes[0].set_ylabel("Частота")
    axes[0].xaxis.set_major_formatter(FMT); axes[0].legend()

    axes[1].boxplot(df["price"], patch_artist=True,
                    boxprops=dict(facecolor="#0f3460", alpha=0.6),
                    medianprops=dict(color="#e94560", linewidth=2))
    axes[1].set_title("Ящик с усами — price", fontweight="bold")
    axes[1].set_ylabel("Цена (€)"); axes[1].yaxis.set_major_formatter(FMT)
    axes[1].tick_params(axis="x", bottom=False, labelbottom=False)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Корреляция ───────────────────────────────────────────
    st.markdown("#### Матрица корреляций (Пирсон)")
    quant_cols = ["squareMeters","price","numberOfRooms","floors",
                  "basement","attic","garage","numPrevOwners"]
    corr = df[quant_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Матрица корреляций", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Средняя цена по бинарным признакам ──────────────────
    st.markdown("#### Средняя цена по характеристикам объекта")
    binary_cols = ["hasPool","hasYard","isNewBuilt","hasStormProtector","hasStorageRoom"]
    labels_ru   = ["Бассейн","Двор","Новостройка","Защита от непогоды","Кладовая"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col, lbl in zip(axes, binary_cols, labels_ru):
        means = df.groupby(col)["price"].mean()
        bars  = ax.bar(["Нет","Да"], means.values,
                       color=["#8da9c4","#0f3460"], edgecolor="white",
                       alpha=0.9, width=0.5)
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height()*1.02,
                    f"{v/1e6:.2f}M", ha="center", va="bottom", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_ylabel("Средняя цена (€)" if col == binary_cols[0] else "")
        ax.yaxis.set_major_formatter(FMT)

    plt.suptitle("Средняя цена по бинарным признакам", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Диаграмма рассеяния squareMeters vs price ────────────
    st.markdown("#### Ключевая зависимость: площадь vs цена")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["squareMeters"], df["price"], alpha=0.6,
               color="#0f3460", edgecolors="white", linewidths=0.4, s=70)
    m, b = np.polyfit(df["squareMeters"], df["price"], 1)
    x_l  = np.linspace(df["squareMeters"].min(), df["squareMeters"].max(), 200)
    ax.plot(x_l, m*x_l+b, color="#e94560", linewidth=2.5, label=f"Тренд  r = 1.000")
    ax.set_xlabel("Площадь (м²)"); ax.set_ylabel("Цена (€)")
    ax.yaxis.set_major_formatter(FMT)
    ax.set_title("squareMeters vs price  (r = 1.000)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 2 — ПРЕДСКАЗАНИЕ ЦЕНЫ
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏠 Оценка стоимости объекта")
    st.markdown(
        "Заполни параметры квартиры — модель **LightGBM** рассчитает "
        "прогнозную рыночную цену на основе обучающей выборки."
    )
    st.info(
        "💡 **Ключевой фактор:** площадь объекта определяет цену "
        "практически линейно (r = 1.000). Остальные параметры дают "
        "тонкую корректировку.",
        icon=None
    )

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("**Основные характеристики**")
        sq  = st.number_input("Площадь (м²)",  min_value=50,   max_value=100_000, value=3_000,  step=100)
        ro  = st.number_input("Количество комнат (индекс 0–100)", min_value=0, max_value=100, value=10, step=1)
        fl  = st.number_input("Этажность (0–100)",  min_value=0, max_value=100, value=5,   step=1)
        yr  = st.number_input("Год постройки",  min_value=1900, max_value=2024, value=2005, step=1)
        own = st.number_input("Прошлых владельцев", min_value=0, max_value=20,  value=2,    step=1)
        cc  = st.number_input("Код города",     min_value=0,    max_value=99999, value=75001, step=1)
        cpr = st.number_input("Рейтинг района (1–10)", min_value=1, max_value=10, value=5, step=1)

        st.markdown("**Дополнительные помещения**")
        c1, c2, c3 = st.columns(3)
        bas = c1.number_input("Подвал (м²)",   min_value=0, max_value=10_000, value=500,  step=50)
        att = c2.number_input("Мансарда (м²)", min_value=0, max_value=10_000, value=500,  step=50)
        gar = c3.number_input("Гараж (м²)",    min_value=0, max_value=1_000,  value=200,  step=10)
        gst = st.number_input("Гостевых комнат", min_value=0, max_value=10, value=0, step=1)

    with col_r:
        st.markdown("**Дополнительные опции**")
        pool   = st.checkbox("🏊 Бассейн")
        yard   = st.checkbox("🌿 Двор")
        new_b  = st.checkbox("🏗️ Новостройка")
        storm  = st.checkbox("⛈️ Защита от непогоды")
        stor   = st.checkbox("📦 Кладовая")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("💰 Рассчитать стоимость",
                                use_container_width=True, type="primary")

        if predict_btn:
            inp = {
                "squareMeters":       sq,
                "numberOfRooms":      ro,
                "hasYard":            int(yard),
                "hasPool":            int(pool),
                "floors":             fl,
                "cityCode":           cc,
                "cityPartRange":      cpr,
                "numPrevOwners":      own,
                "made":               yr,
                "isNewBuilt":         int(new_b),
                "hasStormProtector":  int(storm),
                "basement":           bas,
                "attic":              att,
                "garage":             gar,
                "hasStorageRoom":     int(stor),
                "hasGuestRoom":       gst,
            }
            X_in  = pd.DataFrame([inp])[feature_cols]
            price = lgbm_model.predict(X_in)[0]
            margin = price * 0.10

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Прогнозная рыночная стоимость</div>
                <div class="price">{price:,.0f} €</div>
                <div style="opacity:0.7; margin-top:8px; font-size:0.85rem;">
                    Диапазон ±10%:&nbsp;
                    {(price-margin)/1e6:.2f}M € — {(price+margin)/1e6:.2f}M €
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Цена за м²", f"{price/sq:,.0f} €")
            m2.metric("В миллионах", f"{price/1e6:.3f} M €")

        else:
            st.markdown("""
            <div style="background:#f8f9ff; border:2px dashed #c5cff5;
                        border-radius:12px; padding:3rem 2rem;
                        text-align:center; color:#8a9bb0; margin-top:1rem;">
                <div style="font-size:2.5rem">🏷️</div>
                <div style="font-weight:600; margin-top:0.5rem;">
                    Заполни параметры и нажми кнопку
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ВКЛАДКА 3 — ARIMA ПРОГНОЗ
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Прогноз рынка — временной ряд ARIMA(0,0,1)")
    st.markdown(
        "Временной ряд построен на основе **средней цены объектов "
        "по году их постройки** (1990–2021). Модель ARIMA(0,0,1) "
        "даёт прогноз на 3 года вперёд."
    )

    # Метрики модели
    m1, m2, m3, m4 = st.columns(4)
    arima_metrics = [
        (m1, "Модель",      "ARIMA(0,0,1)", "MA(1)-процесс"),
        (m2, "AIC",         "990.59",        "критерий качества"),
        (m3, "MAE",         "1.51M €",       "средняя ошибка"),
        (m4, "R²",          "0.027",         "объяснённая дисперсия"),
    ]
    for col, lbl, val, sub in arima_metrics:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value" style="font-size:1.3rem">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Прогноз
    n_steps = st.slider("Горизонт прогноза (лет)", 1, 10, 3)

    fc          = arima_model.get_forecast(steps=n_steps)
    fc_mean     = fc.predicted_mean.values
    fc_ci       = fc.conf_int(alpha=0.05)
    last_year   = int(ts["year"].max())
    fc_years    = list(range(last_year + 1, last_year + 1 + n_steps))

    # График
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts["year"], ts["avg_price"],
            marker="o", linewidth=2.5, color="#0f3460",
            markersize=6, markerfacecolor="white", markeredgewidth=2,
            label="Исторические данные")
    ax.fill_between(ts["year"], ts["avg_price"], alpha=0.07, color="#0f3460")

    ax.plot(fc_years, fc_mean,
            marker="s", linewidth=2.5, color="#e94560",
            markersize=8, linestyle="--", label="Прогноз")
    ax.fill_between(fc_years,
                    fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values,
                    alpha=0.2, color="#e94560", label="95% доверительный интервал")

    ax.axvline(x=last_year, color="#aaa", linestyle=":", linewidth=1.5)
    for yr, val in zip(fc_years, fc_mean):
        ax.annotate(f"{val/1e6:.2f}M €",
                    xy=(yr, val), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=10, color="#e94560", fontweight="bold")

    ax.set_title(f"Прогноз средней цены жилья — ARIMA(0,0,1) · горизонт {n_steps} лет",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Год постройки"); ax.set_ylabel("Средняя цена (€)")
    ax.yaxis.set_major_formatter(FMT)
    ax.set_xticks(list(ts["year"]) + fc_years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Таблица прогноза
    st.markdown("#### Таблица прогноза")
    fc_df = pd.DataFrame({
        "Год":             fc_years,
        "Прогноз (€)":    [f"{v:,.0f}" for v in fc_mean],
        "Нижняя граница": [f"{v:,.0f}" for v in fc_ci.iloc[:, 0].values],
        "Верхняя граница":[f"{v:,.0f}" for v in fc_ci.iloc[:, 1].values],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Интерпретация результатов:**
    - Прогнозируемый уровень цен ~4.4–4.5M € соответствует историческому среднему ряда
    - Широкий доверительный интервал отражает высокую волатильность рынка
    - R² = 0.027 закономерен для MA(1) на стационарном ряду без тренда:
      модель предсказывает возврат к среднему, а не конкретные скачки цен
    """)

# ─────────────────────────────────────────────
# ПОДВАЛ
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Paris Housing Analytics · LightGBM + ARIMA · "
    "Курсовая работа по анализу данных"
    "</div>",
    unsafe_allow_html=True
)