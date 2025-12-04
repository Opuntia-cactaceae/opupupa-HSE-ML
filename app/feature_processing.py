import re
import phik
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from phik.report import plot_correlation_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack


# Сюда перенесены функции для обработки датасета из ноутбука
# Попросил гпт написать комментарии чтобы было понятнее

def clean_max_power_train(series: pd.Series) -> pd.Series:
    s = series.copy()

    s_str = s.astype(str)
    mask_nan  = s.isna() | (s_str == 'nan')
    mask_zero = s_str == '0'

    numeric_str = s_str.str.extract(r'([\d.]+)', expand=False)
    numeric = pd.to_numeric(numeric_str, errors='coerce')

    result = numeric.copy()

    result[mask_nan] = np.nan
    result[mask_zero] = 0.0
    mask_weird = result.isna() & ~mask_nan & ~mask_zero
    result[mask_weird] = 0.0

    return result


def convert_mileage(series: pd.Series) -> pd.Series:
    """Переводим значения пробега в единую единицу (kmpl),
    обрабатываем 'kmkg' как kmpl/1.2 и выкидываем юниты."""
    numeric = (
        series.astype(str)
              .str.extract(r'([\d.]+)', expand=False)
              .astype(float)
    )
    units = series.astype(str).str.extract(r'([A-Za-z]+)', expand=False)

    result = numeric.copy()
    mask_kmkg = units == 'kmkg'
    result[mask_kmkg] = numeric[mask_kmkg] / 1.2
    return result


def drop_units(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
              .str.extract(r'([\d.]+)', expand=False)
              .astype(float)
    )


def parse_torque_value(x):
    """Разобрать одно значение крутящего момента -> (Nm, rpm)."""
    if pd.isna(x):
        return np.nan, np.nan

    s = str(x).strip()
    if not s or s == '/':
        return np.nan, np.nan

    s_low = s.lower()
    nums = re.findall(r'(\d+(?:\.\d+)?)', s_low)
    if not nums:
        return np.nan, np.nan

    torque_val = float(nums[0])

    if 'kgm' in s_low:
        torque_val = torque_val * 9.8

    rpm_val = np.nan
    if len(nums) >= 2:
        rpm_candidates = [float(n) for n in nums[1:]]
        rpm_val = max(rpm_candidates)

    return torque_val, rpm_val


def split_torque(series: pd.Series):
    """Разбить колонку torque на Nm и rpm."""
    torques = []
    rpms = []
    for x in series:
        t, r = parse_torque_value(x)
        torques.append(t)
        rpms.append(r)
    return pd.Series(torques, index=series.index), pd.Series(rpms, index=series.index)


def tweak_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Базовый препроцесс:

    - спец-очистка max_power на трейне;
    - удаление дубликатов по признакам (кроме selling_price);
    - парсинг engine / max_power / mileage / torque;
    - расчёт медиан для всех числовых колонок (кроме selling_price);
    - расчёт "моды"/'unknown' для категориальных колонок;
    - заполнение пропусков в train/test этими значениями.

    Возвращает:
        train_clean,
        test_clean,
        numeric_medians   (pd.Series: колонки -> медиана),
        cat_fill_values   (pd.Series: колонки -> значение-заглушка)
    """

    train = df_train.copy()
    test  = df_test.copy()

    if "max_power" in train.columns:
        train["max_power"] = clean_max_power_train(train["max_power"])

    if "selling_price" in train.columns:
        feature_cols = train.columns.drop("selling_price")
        train = (
            train
            .drop_duplicates(subset=feature_cols, keep="first")
            .reset_index(drop=True)
        )

    for df in (train, test):
        if "engine" in df.columns:
            df["engine"] = drop_units(df["engine"])
        if "max_power" in df.columns:
            df["max_power"] = drop_units(df["max_power"])
        if "mileage" in df.columns:
            df["mileage"] = convert_mileage(df["mileage"])

        if "torque" in df.columns:
            torque_nm, torque_rpm = split_torque(df["torque"])
            df["torque"] = torque_nm
            df["max_torque_rpm"] = torque_rpm

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if "selling_price" in numeric_cols:
        numeric_cols.remove("selling_price")

    numeric_medians = train[numeric_cols].median()

    train[numeric_cols] = train[numeric_cols].fillna(numeric_medians)
    test[numeric_cols]  = test[numeric_cols].fillna(numeric_medians)

    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    cat_fill_dict = {}
    for col in cat_cols:
        mode = train[col].dropna().mode()
        if len(mode) > 0:
            fill_value = mode.iloc[0]
        else:
            fill_value = "unknown"
        cat_fill_dict[col] = fill_value

    cat_fill_values = pd.Series(cat_fill_dict)

    if cat_cols:
        train[cat_cols] = train[cat_cols].fillna(cat_fill_values)
        test[cat_cols]  = test[cat_cols].fillna(cat_fill_values)

    return train, test, numeric_medians, cat_fill_values

def fill_missing_with_statistics(
    df: pd.DataFrame,
    numeric_medians: pd.Series,
    cat_fill_values: pd.Series,
) -> pd.DataFrame:
    """
    Заполняет пропуски в df с использованием статистик,
    посчитанных на train в tweak_dataset.
    Работает и для одной строки (df.shape = (1, n_cols)).
    """
    df_filled = df.copy()

    num_cols_to_fill = [col for col in numeric_medians.index if col in df_filled.columns]
    if num_cols_to_fill:
        df_filled[num_cols_to_fill] = df_filled[num_cols_to_fill].fillna(
            numeric_medians[num_cols_to_fill]
        )

    cat_cols_to_fill = [col for col in cat_fill_values.index if col in df_filled.columns]
    if cat_cols_to_fill:
        df_filled[cat_cols_to_fill] = df_filled[cat_cols_to_fill].fillna(
            cat_fill_values[cat_cols_to_fill]
        )

    return df_filled


def create_pairplots(df_train, df_test, num_cols, title_prefix="Pairplot"):
    g_train = sns.pairplot(df_train[num_cols])
    g_train.fig.suptitle(f"{title_prefix} — трейн", y=1.02, fontsize=16)

    g_test = sns.pairplot(df_test[num_cols])
    g_test.fig.suptitle(f"{title_prefix} — тест", y=1.02, fontsize=16)

    return g_train.fig, g_test.fig


def create_corr_heatmap(df, num_cols, title="Корреляционная матрица"):
    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(title)

    return fig


def create_phik_heatmap(df,
                        interval_cols=None,
                        vmin=0, vmax=1,
                        cmap="coolwarm",
                        title="phik корреляции"):
    """
    Строит phik-матрицу и возвращает figure для вывода в Streamlit.
    Работает и с числовыми, и с категориальными признаками.
    """

    if interval_cols is None:
        interval_cols = []

    corr_phik = df.phik_matrix(interval_cols=interval_cols)

    plot_correlation_matrix(
        corr_phik.values,
        x_labels=corr_phik.columns,
        y_labels=corr_phik.index,
        vmin=vmin,
        vmax=vmax,
        color_map=cmap,
        title=title,
        figsize=(12, 10),
        fontsize_factor=1.1,
    )

    fig = plt.gcf()
    return fig


def create_year_km_dependency_plots(df, lowess_frac=0.1):
    figs = []

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    (
        df.groupby("year")["selling_price"]
          .mean()
          .plot(ax=ax1)
    )
    ax1.set_title("Средняя цена по году")
    ax1.set_ylabel("Цена")
    figs.append(fig1)

    lowess = sm.nonparametric.lowess
    z = lowess(df["selling_price"], df["km_driven"], frac=lowess_frac)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(z[:, 0], z[:, 1])
    ax2.set_xlabel("Пробег")
    ax2.set_ylabel("Средняя цена")
    ax2.set_title("Сглаженная зависимость (LOWESS: Price vs km_driven)")
    ax2.grid(True)
    figs.append(fig2)

    km_driven_log = np.log1p(df["km_driven"])
    z2 = lowess(df["selling_price"], km_driven_log, frac=lowess_frac)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(z2[:, 0], z2[:, 1])
    ax3.set_xlabel("km_driven_log")
    ax3.set_ylabel("Цена (сглаженная)")
    ax3.set_title("LOWESS: Price vs log(km_driven)")
    ax3.grid(True)
    figs.append(fig3)

    return figs


def car_class_from_name(name: str) -> str:
    name = str(name).lower()

    # топ класс
    if any(b in name for b in ["bmw", "mercedes", "audi", "jaguar", "volvo", "lexus"]):
        return "luxury"

    # кроссовер / SUV
    suv_keys = [
        "scorpio","xuv","brezza","creta","ecosport","duster","fortuner",
        "venue","safari","s-cross","kodiaq","captiva","rexton","pajero",
        "compass","wr-v","city cross"
    ]
    if any(k in name for k in suv_keys):
        return "suv"

    # минивен / MPV
    mpv_keys = ["innova", "ertiga", "triber", "marazzo", "xylo", "enjoy", "mobilio", "stile"]
    if any(k in name for k in mpv_keys):
        return "mpv"

    # седан
    sedan_keys = [
        "city", "civic", "verna", "ciaz", "sx4", "swift dzire", "dzire",
        "accent", "rapid", "octavia", "vento", "corolla", "altis",
        "linea","fiesta","sunny","scala","elantra","sonata","sail sedan"
    ]
    if any(k in name for k in sedan_keys):
        return "sedan"

    # хэтчбек
    hatch_keys = [
        "alto","wagon r","i10","i20","santro","baleno","polo","tiago",
        "kwid","ignis","zen","estilo","figo","micra","ritz","eon","go","swift"
    ]
    if any(k in name for k in hatch_keys):
        return "hatchback"

    # пикап
    pickup_keys = ["bolero", "tuv","camper","supro"]
    if any(k in name for k in pickup_keys):
        return "pickup"

    return "other"


def prepare_brand_clusters(df_train: pd.DataFrame,
                           df_test: pd.DataFrame,
                           n_brand_clusters: int = 5):
    """
    Берёт df_train/df_test (с колонками name, selling_price),
    добавляет brand_cluster и заменяет name на номер кластера.
    """
    df_cat_train = df_train.copy()
    df_cat_test = df_test.copy()

    df_cat_train["brand"] = df_cat_train["name"].str.split().str[0]
    df_cat_test["brand"] = df_cat_test["name"].str.split().str[0]

    mean_price_per_brand = df_cat_train.groupby("brand")["selling_price"].mean()

    kmeans = KMeans(n_clusters=n_brand_clusters, random_state=42, n_init=10)
    brand_clusters_train = kmeans.fit_predict(mean_price_per_brand.to_frame())

    brand_to_cluster = dict(zip(mean_price_per_brand.index, brand_clusters_train))

    df_cat_train["brand_cluster"] = df_cat_train["brand"].map(brand_to_cluster)
    df_cat_test["brand_cluster"] = df_cat_test["brand"].map(brand_to_cluster)

    df_cat_test["brand_cluster"] = df_cat_test["brand_cluster"].fillna(-1).astype(int)

    df_cat_train["name"] = df_cat_train["brand_cluster"].astype(int)
    df_cat_test["name"] = df_cat_test["brand_cluster"].astype(int)

    df_cat_train = df_cat_train.drop(columns=["brand", "brand_cluster"])
    df_cat_test = df_cat_test.drop(columns=["brand", "brand_cluster"])

    return df_cat_train, df_cat_test, brand_to_cluster


def build_numeric_bonus_features(df_train: pd.DataFrame,
                                 df_test: pd.DataFrame,
                                 numeric_cols,
                                 current_year: int = 2024):
    """
    Строит числовые фичи из бонусной части на основе исходных df_train/df_test.
    Ожидает в них колонки numeric_cols + selling_price.
    """
    df_num_train = df_train[numeric_cols + ["selling_price"]].copy()
    df_num_test = df_test[numeric_cols + ["selling_price"]].copy()

    df_num_train["max_power_log"] = np.log1p(df_num_train["max_power"])
    df_num_test["max_power_log"] = np.log1p(df_num_test["max_power"])

    df_num_train["engine_log"] = np.log1p(df_num_train["engine"])
    df_num_test["engine_log"] = np.log1p(df_num_test["engine"])

    df_num_train["torque_log"] = np.log1p(df_num_train["torque"])
    df_num_test["torque_log"] = np.log1p(df_num_test["torque"])

    df_num_train["power_per_liter"] = df_num_train["max_power"] / df_num_train["engine"]
    df_num_test["power_per_liter"] = df_num_test["max_power"] / df_num_test["engine"]

    low, high = df_num_train["torque"].quantile([0.01, 0.99])
    df_num_train["torque"] = df_num_train["torque"].clip(low, high)
    df_num_test["torque"] = df_num_test["torque"].clip(low, high)

    mileage_median = df_num_train["mileage"].median()
    df_num_train.loc[df_num_train["mileage"] < 2, "mileage"] = mileage_median
    df_num_test.loc[df_num_test["mileage"] < 2, "mileage"] = mileage_median

    df_num_train["km_driven_log"] = np.log1p(df_num_train["km_driven"])
    df_num_test["km_driven_log"] = np.log1p(df_num_test["km_driven"])

    df_num_train["is_high_mileage"] = (df_num_train["km_driven"] > 150000).astype(int)
    df_num_test["is_high_mileage"] = (df_num_test["km_driven"] > 150000).astype(int)

    df_num_train["age"] = current_year - df_train["year"].values
    df_num_test["age"] = current_year - df_test["year"].values

    df_num_train["age_sq"] = df_num_train["age"] ** 2
    df_num_test["age_sq"] = df_num_test["age"] ** 2

    return df_num_train, df_num_test


def scale_numeric_features(df_num_train: pd.DataFrame,
                           df_num_test: pd.DataFrame):
    X_train_num = df_num_train.drop(columns=["selling_price"])
    y_train = df_num_train["selling_price"]

    X_test_num = df_num_test.drop(columns=["selling_price"])
    y_test = df_num_test["selling_price"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def encode_categorical_features(df_cat_train: pd.DataFrame,
                                df_cat_test: pd.DataFrame,
                                cat_cols):
    X_train_cat = df_cat_train.drop(columns=["selling_price"])
    X_test_cat = df_cat_test.drop(columns=["selling_price"])

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    X_train_cat_enc = ct.fit_transform(X_train_cat)
    X_test_cat_enc = ct.transform(X_test_cat)

    return X_train_cat_enc, X_test_cat_enc, ct


def assemble_features(X_train_num_scaled,
                      X_test_num_scaled,
                      X_train_cat_enc,
                      X_test_cat_enc):
    X_train = hstack([X_train_num_scaled, X_train_cat_enc])
    X_test = hstack([X_test_num_scaled, X_test_cat_enc])
    return X_train, X_test


def tweak_cat(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    numeric_cols,
    n_brand_clusters: int = 5,
    current_year: int = 2024
):
    """
    Высокоуровневая функция для построения финальных матриц признаков
    (числовые + категориальные) перед обучением/предсказанием.

    Ожидается, что df_train/df_test уже прошли через tweak_dataset.

    Возвращает:
        X_train_bonus, X_test_bonus,
        y_train_bonus, y_test_bonus,
        scaler_bonus, ct, brand_to_cluster
    """
    for col in ["year", "seats"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    df_cat_train, df_cat_test, brand_to_cluster = prepare_brand_clusters(
        df_train, df_test, n_brand_clusters=n_brand_clusters
    )

    df_cat_train["car_class"] = df_train["name"].apply(car_class_from_name).values
    df_cat_test["car_class"] = df_test["name"].apply(car_class_from_name).values

    df_num_train_bonus, df_num_test_bonus = build_numeric_bonus_features(
        df_train, df_test, numeric_cols=numeric_cols, current_year=current_year
    )

    (X_train_num_scaled,
     X_test_num_scaled,
     y_train_bonus,
     y_test_bonus,
     scaler_bonus) = scale_numeric_features(df_num_train_bonus, df_num_test_bonus)

    cat_cols = ["fuel", "seller_type", "transmission", "owner", "name", "seats", "car_class"]
    X_train_cat_enc, X_test_cat_enc, ct = encode_categorical_features(
        df_cat_train, df_cat_test, cat_cols=cat_cols
    )

    X_train_bonus, X_test_bonus = assemble_features(
        X_train_num_scaled, X_test_num_scaled, X_train_cat_enc, X_test_cat_enc
    )



    return (
        X_train_bonus,
        X_test_bonus,
        y_train_bonus,
        y_test_bonus,
        scaler_bonus,
        ct,
        brand_to_cluster,
    )

def create_boxplot_figure(df: pd.DataFrame,
                          column: str,
                          title: str | None = None):
    """
    Строит боксплот для одной числовой колонки и возвращает figure.
    Удобно использовать в Streamlit через st.pyplot(fig).
    """
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(title or column)
    return fig

