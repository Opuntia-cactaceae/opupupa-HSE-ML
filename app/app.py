import io
import traceback

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import feature_processing as fp
#здесь тоже попросил гпт написать комменты


REQUIRED_FEATURE_COLS = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

@st.cache_resource
def load_model():
    try:
        model_path = st.secrets.get("MODEL_PATH", "app/final_ridge_model.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Не удалось загрузить модель из final_ridge_model.pkl: {e}")
        st.stop()


@st.cache_resource
def load_train_df():
    CSV_PATH = st.secrets.get(
        "TRAIN_CSV",
        "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    )

    try:
        df_train = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Не удалось загрузить обучающий датасет по URL: {e}")
        st.stop()

    missing = [c for c in REQUIRED_FEATURE_COLS + ["selling_price"] if c not in df_train.columns]
    if missing:
        st.error(
            "В обучающем датасете (по URL) отсутствуют необходимые колонки: "
            + ", ".join(missing)
        )
        st.stop()

    return df_train

def make_features_for_model(
    df_raw: pd.DataFrame,
    df_train_raw: pd.DataFrame,
    model,
):
    """
    Главная функция для препроцессинга новых данных.

    df_raw        — загруженный CSV или одна строка (без selling_price),
    df_train_raw  — исходный train c selling_price,
    model         — обученный Ridge, нужен только для проверки размерности.

    Шаги:
    1) Прогоняем train + new через tweak_dataset (вся очистка + заполнение пропусков).
    2) Добавляем в new фиктивный selling_price (он нужен только функциям из ноутбука).
    3) Строим числовые/категориальные фичи через tweak_cat.
    4) Берём X_test_bonus и проверяем, что размерность совпадает с coef_ модели.
    """

    if df_raw.empty:
        raise ValueError("Загруженный датасет пустой.")

    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            "В датасете отсутствуют необходимые колонки: " + ", ".join(missing)
        )

    df_train_prep, df_new_prep, _, _ = fp.tweak_dataset(df_train_raw, df_raw)

    if "selling_price" not in df_new_prep.columns:
        df_new_prep = df_new_prep.copy()
        df_new_prep["selling_price"] = np.nan

    numeric_cols = df_train_prep.select_dtypes(include=[np.number]).columns.tolist()
    if "selling_price" in numeric_cols:
        numeric_cols.remove("selling_price")

    (
        X_train_bonus,
        X_new_bonus,
        y_train_bonus,
        y_new_bonus,
        scaler_bonus,
        ct,
        brand_to_cluster,
    ) = fp.tweak_cat(df_train_prep, df_new_prep, numeric_cols=numeric_cols)

    coef = np.array(model.coef_).ravel()
    if X_new_bonus.shape[1] != coef.shape[-1]:
        raise ValueError(
            f"Размерность признаков ({X_new_bonus.shape[1]}) не совпадает "
            f"с размерностью коэффициентов модели ({coef.shape[-1]}). "
        )

    return X_new_bonus


def show_eda(df: pd.DataFrame):
    st.subheader("Обзор")
    st.write("Размер датасета:", df.shape)
    st.dataframe(df.head())


    if "selling_price" in df.columns:
        df_prep, _, _, _ = fp.tweak_dataset(df, df.copy())
    else:
        # если нет selling_price
        df_prep = df.copy()
        if "engine" in df_prep.columns:
            df_prep["engine"] = fp.drop_units(df_prep["engine"])
        if "max_power" in df_prep.columns:
            df_prep["max_power"] = fp.drop_units(df_prep["max_power"])
        if "mileage" in df_prep.columns:
            df_prep["mileage"] = fp.convert_mileage(df_prep["mileage"])
        if "torque" in df_prep.columns:
            torque_nm, torque_rpm = fp.split_torque(df_prep["torque"])
            df_prep["torque"] = torque_nm
            df_prep["max_torque_rpm"] = torque_rpm

    numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_prep.select_dtypes(include=["object", "category"]).columns.tolist()


    if numeric_cols:
        st.subheader("Обзор числовых колонок")
        desc_num = df_prep[numeric_cols].describe().T
        st.dataframe(desc_num.astype("object"))
    else:
        st.info("Числовых колонок не найдено.")

    if cat_cols:
        st.subheader("Обзор категориальных колонок")
        desc_cat = df_prep[cat_cols].describe().T
        st.dataframe(desc_cat.astype("object"))

    if not numeric_cols:
        # если числовых колонок нет просто стопим
        return

    st.subheader("Боксплоты")
    cols_for_box = st.multiselect(
        "Выберите колонки для отображения",
        numeric_cols,
        default=[c for c in numeric_cols if c not in ("selling_price", "km_driven")][:5],
    )
    for col in cols_for_box:
        fig_box = fp.create_boxplot_figure(df_prep, col)
        st.pyplot(fig_box)

    if len(numeric_cols) >= 2:
        st.subheader("Корреляционная матрица")
        fig_corr = fp.create_corr_heatmap(
            df_prep,
            num_cols=numeric_cols,
            title="Корреляционная матрица"
        )
        st.pyplot(fig_corr)

    st.subheader("Pairplot")
    st.caption("Если нужно можно отобразить")
    cols_for_pair = st.multiselect(
        "Колонки для pairplot",
        numeric_cols,
        default=[c for c in numeric_cols if c in ("year", "selling_price", "km_driven", "mileage", "engine", "max_power")],
    )
    if cols_for_pair and st.button("Построить pairplot"):
        fig_pair_train, fig_pair_test = fp.create_pairplots(
            df_prep, df_prep, num_cols=cols_for_pair, title_prefix="Pairplot"
        )
        st.pyplot(fig_pair_train)

    if {"selling_price", "km_driven", "year"}.issubset(df_prep.columns):
        st.subheader("Зависимость цены от года и пробега (LOWESS)")
        figs_lowess = fp.create_year_km_dependency_plots(df_prep)
        for f in figs_lowess:
            st.pyplot(f)
    else:
        st.info("Для LOWESS-графиков нужны колонки year, km_driven и selling_price.")

    st.subheader("phik корреляции")
    st.caption("Может долго работать")

    if st.button("Построить phik-матрицу"):
        try:
            fig_phik = fp.create_phik_heatmap(
                df_prep,
                interval_cols=numeric_cols,
                vmin=0,
                vmax=1,
                title="phik корреляции"
            )

            st.pyplot(fig_phik)
        except Exception as e:
            st.error("Не удалось посчитать phik-корреляции.")


def page_predict_for_csv(model, df_train_raw):
    st.header("Предсказание для загруженного CSV")

    uploaded_file = st.file_uploader("Ваш CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Пока файл не загружен.")
        return

    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения CSV: {e}")
        st.exception(e)
        return

    st.subheader("Первые строки загруженного датасета:")
    st.write(df_raw.head())

    try:
        X_new = make_features_for_model(df_raw, df_train_raw, model)
    except Exception as e:
        st.error("Ошибка на этапе препроцессинга / построения признаков:")
        st.exception(e)
        return

    try:
        preds = model.predict(X_new)
    except Exception as e:
        st.error("Ошибка на этапе предсказания:")
        st.exception(e)
        return

    st.subheader("Предсказания (первые 20 строк)")
    st.write(preds[:20])

    try:
        df_out = df_raw.copy()
        df_out["prediction"] = preds
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        st.download_button(
            label="Скачать полные предсказания в CSV",
            data=csv_buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Не удалось подготовить файл для скачивания: {e}")
        st.exception(e)


def page_predict_single(model, df_train_raw):
    st.header("Предсказание для одного объекта")

    st.markdown("Заполните признаки автомобиля:")

    name = st.text_input("name", value="maruti swift dzire vdi")
    year = st.number_input("year", min_value=1990, max_value=2025, value=2015, step=1)
    km_driven = st.number_input("km_driven", min_value=0, value=50000, step=1000)

    fuel = st.selectbox("fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("transmission", ["Manual", "Automatic"])
    owner = st.selectbox(
        "owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
        ],
    )

    mileage = st.text_input("mileage (напр. '20.4 kmpl')", value="20.4 kmpl")
    engine = st.text_input("engine (напр. '1248 CC')", value="1248 CC")
    max_power = st.text_input("max_power (напр. '74 bhp')", value="74 bhp")
    torque = st.text_input("torque", value="190Nm@ 2000rpm")
    seats = st.number_input("seats", min_value=2, max_value=10, value=5, step=1)

    if st.button("Предсказать цену"):
        try:
            row = {
                "name": name,
                "year": int(year),
                "km_driven": int(km_driven),
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "torque": torque,
                "seats": int(seats),
            }
            df_raw = pd.DataFrame([row])

            X_new = make_features_for_model(df_raw, df_train_raw, model)
            pred = model.predict(X_new)[0]

            st.success(f"Оценочная цена: {pred:,.0f}")
        except Exception as e:
            st.error("Ошибка при предсказании для одного объекта:")
            st.exception(e)



def page_model_weights(model):
    st.header("Веса обученной модели")

    ridge = model

    try:
        coef = np.array(ridge.coef_).ravel()
    except Exception as e:
        st.error(f"Не удалось получить веса модели: {e}")
        st.exception(e)
        return

    feature_names = np.array([f"f_{i}" for i in range(len(coef))])

    abs_coef = np.abs(coef)
    top_n = st.slider("Количество отображаемых признаков", 5, 50, 20)

    idx = np.argsort(abs_coef)[-top_n:]

    df_weights = pd.DataFrame(
        {
            "feature": feature_names[idx],
            "coef": coef[idx],
            "abs_coef": abs_coef[idx],
        }
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    st.subheader("Топ важнейших признаков по весу")
    st.dataframe(df_weights)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax.barh(df_weights["feature"], df_weights["coef"])
    ax.set_xlabel("Коэффициент")
    ax.set_ylabel("Признак")
    ax.set_title("Веса модели")
    plt.tight_layout()
    st.pyplot(fig)


def page_eda(df_train_raw):
    st.header("Исследовательский анализ данных (EDA)")

    uploaded = st.file_uploader(
        "Ваш CSV для EDA (по умолчанию показан EDA по обучающему датасету)", type=["csv"]
    )

    if uploaded is not None:
        try:
            df_eda = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Ошибка при чтении CSV для EDA.")
            st.exception(e)
            return
        st.info("EDA по загруженному датасету.")
    else:
        st.info("Файл не загружен, показан EDA по обучающему датасету.")
        df_eda = df_train_raw

    show_eda(df_eda)


def main():
    st.title("Модель предсказания цены автомобиля")

    model = load_model()
    df_train_raw = load_train_df()

    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Раздел:",
        (
            "EDA",
            "Предсказание по CSV",
            "Предсказание для одной записи",
            "Веса модели",
        ),
    )

    try:
        if page == "EDA":
            page_eda(df_train_raw)
        elif page == "Предсказание по CSV":
            page_predict_for_csv(model, df_train_raw)
        elif page == "Предсказание для одной записи":
            page_predict_single(model, df_train_raw)
        elif page == "Веса модели":
            page_model_weights(model)
    except Exception as e:
        st.error("Неперехваченная ошибка на странице:")
        st.text("".join(traceback.format_exception(type(e), e, e.__traceback__)))


if __name__ == "__main__":
    main()
