import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


st.set_page_config(layout='wide', page_title='FinanScouts', page_icon='💸')

topcol1, topcol2 = st.columns([3, 1])

html_title = """
<div style='font-size:100px; font-family:"Roboto", sans-serif; font-weight:600;'>
    <span style='color:#00ff9c;'>Finan</span>Scouts 💸
</div>
"""
topcol1.markdown(html_title, unsafe_allow_html=True)
topcol1.markdown('*Ünlü Düşünür Donald J. Trump der ki;* Hisse senetlerine yatırım yapın, çünkü ben de onları sevdim ve ben her zaman doğruyum :)')
topcol2.image("Data/Trump.gif", use_column_width=True)

home_tab, graph_tab, modelling_tab, prediction_tab = st.tabs(["Ana Sayfa", "Veriye Genel Bakış", "Modelleme", "Tahminleme"])


col1, col2, col3 = home_tab.columns([1, 1, 1])

col1.subheader("Hoş Geldiniz!")
col1.markdown('Geleceğin Borsasını Keşfedin.')
col1.markdown('Güvenilir, doğru ve zamanında borsa tahminleri ile yatırımlarınızı güvenle yönetin. Sizler için en güncel analizleri ve tahminleri sunuyoruz.')
col1.markdown('Biz Kimiz?')
col1.markdown('Yatırımcılara en doğru bilgiyi sağlama misyonuyla çalışıp, gelişmiş algoritmalar ile piyasa hareketlerini önceden tahmin ederek size rehberlik ediyoruz.')
col3.subheader('Neler Sunuyoruz?')
col3.markdown('Günlük Borsa Tahminleri: Piyasanın nabzını tutarak, her gün için en doğru tahminleri sunuyoruz.')
col3.markdown('Analiz ve Raporlar: Geçmişe dönük verileri istatistiksel olarak analiz ederek 30 günlük tahmin sonuçlarını ve grafiğini sunuyoruz.')
col3.markdown('DİKKAT! Yatırım tavsiyesi değildir.')



col2.subheader("Nasıl çalışır?")
col2.markdown("İlk önce sizden yatırım tercihinizi belirlemenizi isteriz. Sonra, bu bilgiyi kullanarak, zaman serisi algoritmaları ve karmaşık özellik çıkarımlarına göre tahminler yaparız. Ayrıca, her gün güncellenen veri seti ile sizlere daha güncel tahminler sunarız. Böylece her ziyaretinizde yeni ve ilginizi çekebilecek yatırım fırsatlarını keşfedebilirsiniz.")
col2.markdown("Hisse senedine ait sembol kodunu girdikten sonra 'Veriye Genel Bakış', 'Modelleme' ve 'Tahminleme' bölümlerinden sonuçları görüntüleyebilirsiniz.")


def future(df):
    future_df = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=30, freq='B')
    future_df = pd.DataFrame(index=future_df)
    return future_df


def ts_decompose(y, model="additive"):
    #Datetime
    y.index = pd.to_datetime(y.index)
    #Hedef Değişkeni Yeninden Ölçeklendirme
    y = y['Adj Close'].resample('MS').mean()
    y.dropna(inplace=True)

    # Seasonal_decompose fonksiyonunu
    result = seasonal_decompose(y, model=model)

    # Grafik oluşturma
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)
    graph_tab.pyplot(fig)


def deviation (df,new_best_model):
    data = df["Adj Close"].tail(30)
    df = df.iloc[:-30]

    future_df1 = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=30, freq='B')
    future_df1 = pd.DataFrame(index=future_df1)

    for time in future_df1.index:
        df = df._append(future_df1.loc[time])
        df_copy = feature_engineering(df)
        adj_scaler = StandardScaler()

        if df_copy["Adj Close"].isna().any():
            mask= df_copy[df_copy["Adj Close"].isnull()]
            df_copy.dropna(inplace=True)
            df_copy._append(mask)

        df_copy["Adj Close"] = adj_scaler.fit_transform(df_copy[["Adj Close"]])

        train = df_copy.iloc[:int(len(df_copy)*0.80)]
        val = df_copy.iloc[int(len(df_copy)*0.80):]

        y_Train = train["Adj Close"]
        X_Train = train.drop("Adj Close", axis=1)
        X_val = val.drop("Adj Close", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train, test_size=0.30, random_state=17)

        model = new_best_model
        model.fit(X_train, y_train)

        tahmin = model.predict(X_val.iloc[[-1]])
        df["Adj Close"].iloc[[-1]] = tahmin
        tahmin_inverse = adj_scaler.inverse_transform([[tahmin[0]]])[0][0]
        df["Adj Close"].iloc[[-1]] = tahmin_inverse

    mae =  mean_absolute_error(data, df["Adj Close"].iloc[-30:])
    mse =  mean_squared_error(data, df["Adj Close"].iloc[-30:])
    print("MAE: " , mae)
    print("MSE: " , mse)
    return mae, mse


def feature_engineering(df):
    df_copy = df.copy().reset_index()
    df_copy.drop(["Open", "Volume", "High", "Low", "Close"], axis=1, inplace=True)


    df_copy['month'] = df_copy.Date.dt.month
    df_copy['day_of_month'] = df_copy.Date.dt.day
    df_copy['day_of_year'] = df_copy.Date.dt.dayofyear
    df_copy['week_of_year'] = df_copy.Date.dt.isocalendar().week
    df_copy['day_of_week'] = df_copy.Date.dt.dayofweek
    df_copy['year'] = df_copy.Date.dt.year
    df_copy['is_month_start'] = df_copy.Date.dt.is_month_start.astype(int)
    df_copy['is_month_end'] = df_copy.Date.dt.is_month_end.astype(int)
    df_copy["week_of_year"] = df_copy["week_of_year"].astype("int64")


    # Gürültü ekleme
    def random_noise(df):
        return np.random.normal(scale=1.6, size=(len(df)))

    #Günlük Değişim
    df_copy["daily_rate"] = df_copy['Adj Close'].pct_change()

    # Lag değerlerini ekleme
    lags = range(1,31)
    windows = range(3,17)
    alphas = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    for lag in lags:
        df_copy['Adj Close_Lag_' + str(lag)] = df_copy["Adj Close"].shift(lag) + random_noise(df)
    for window in windows:
        df_copy['Adj Close_Roll_Mean_' + str(window)] = df_copy["Adj Close"].shift(1).rolling(window=window, min_periods=3, win_type="triang").mean() + random_noise(df)

    for alpha in alphas:
        for lag_a in lags:
            df_copy['Adj Close_Ewm_Alpha_' + str(alpha).replace(".", "") + "lag" + str(lag_a)] = df_copy["Adj Close"].shift(lag_a).ewm(alpha=alpha).mean() + random_noise(df)


    # Değişken Tiplerini Yakalama

    def grab_col_names(df, cat_th=10, car_th=20):
        cat_cols = [col for col in df.columns if df[col].dtype == "O"]
        num_cols = [col for col in df.columns if df[col].dtype in ["int64", "int32", "float64"]]
        num_but_cat = [col for col in df.columns if df[col].dtype != "O" and df[col].nunique() < cat_th]
        cat_but_car = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() > car_th]

        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in [cat_but_car,"year"]]
        num_cols = [col for col in num_cols if col not in num_but_cat]
        num_cols = [col for col in num_cols if col != "Adj Close"]

        if "year" not in num_cols:
            num_cols.append("year")
        return cat_cols, num_cols, cat_but_car
    cat_cols, num_cols, cat_but_car = grab_col_names(df_copy)

    # Encoding İşlemi
    df_copy.set_index("Date", inplace=True)
    if df_copy[cat_cols].nunique().max() > 2:
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True)
    else:
        for col in cat_cols:
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col])

    df_copy = df_copy.apply(lambda x: x.astype("int64") if x.dtype == "bool" else x)

    #Standartlaştırma işlemi
    scaler = StandardScaler()
    df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])

    return df_copy


def cv(df_copy):
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "Cart": DecisionTreeRegressor(),
            "XGBRegressor": XGBRegressor(verbose=-1),
            "LightGBM": LGBMRegressor(verbose=-1),
            "CatBoost": CatBoostRegressor(verbose=False),
            "AdaBoost": AdaBoostRegressor()
        }

        df_copy.dropna(inplace=True)

        y = df_copy["Adj Close"]
        X = df_copy.drop("Adj Close", axis=1)

        result = {}
        best_model_name = None
        best_mae = float('inf')

        for name, model in models.items():
            scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
            cv_results = cross_validate(model, X, y, cv=2, scoring=scoring)
            mae = -cv_results["test_neg_mean_absolute_error"].mean()

            result[name] = {
                "mse": -cv_results["test_neg_mean_squared_error"].mean(),
                "mae": -cv_results["test_neg_mean_absolute_error"].mean(),
                "r2": cv_results["test_r2"].mean()
            }

            if mae < best_mae:
                best_mae = mae
                best_model_name = name


        best_model = models[best_model_name]
        model_name = best_model_name
        mae_value = result.get(model_name, {}).get("mae", None)


        print(result)
        print(best_model_name)
        return result, best_model, mae_value

def prediction(df, future_df,new_best_model):
    tahminler = []
    for time in future_df.index:
        df = df._append(future_df.loc[time])
        df_copy = feature_engineering(df)
        adj_scaler = StandardScaler()

        if df_copy["Adj Close"].isna().any():
            mask= df_copy[df_copy["Adj Close"].isnull()]
            df_copy.dropna(inplace=True)
            df_copy._append(mask)

        df_copy["Adj Close"] = adj_scaler.fit_transform(df_copy[["Adj Close"]])

        train = df_copy.iloc[:int(len(df_copy)*0.80)]
        val = df_copy.iloc[int(len(df_copy)*0.80):]

        y_Train = train["Adj Close"]
        X_Train = train.drop("Adj Close", axis=1)
        X_val = val.drop("Adj Close", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train, test_size=0.20, random_state=17)
        model = new_best_model
        model.fit(X_train, y_train)

        tahmin = model.predict(X_val.iloc[[-1]])
        df["Adj Close"].iloc[[-1]] = tahmin
        tahmin_inverse = adj_scaler.inverse_transform([[tahmin[0]]])[0][0]

        df["Adj Close"].iloc[[-1]] = tahmin_inverse
        tahminler.append(tahmin_inverse)
        print(f"tahmin {time}: {tahmin_inverse}")
    return df, tahminler


def graph(df, y_pred, future_df, ticker, mae, mse):
    y_pred = pd.Series(y_pred, index=future_df.index)

    # Plotly figürü oluşturma
    fig = go.Figure()

    # Mevcut fiyatları ekleme
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name="CURRENT PRICES",line=dict(color='blue')))

    # Gelecek tahminleri ekleme
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines', name="FUTURE PREDICTION", line=dict(color='orange')))

    # Başlık ve eksen etiketlerini ayarlama
    fig.update_layout(
        title=f"{ticker}'s Prices and Prediction - MAE : {round(mae, 2)}, MSE: {round(mse,2)}",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        hovermode="x unified"
    )

    # Streamlit'e figürü ekleme
    prediction_tab.plotly_chart(fig)


def main():
    st.sidebar.header("Kullanıcı Girişi")
    ticker = st.sidebar.text_input("Hisse Senedi Sembol Kodu")
    st.sidebar.markdown("Örneğin Apple hisse senedi için 'AAPL', Tüpraş hisse senedi için 'TUPRS.IS' giriniz. ")
    st.sidebar.title("Ekibimiz")
    st.sidebar.image("Data/cicek.png", width=70) 
with st.sidebar:
    st.write("---")  # Ayırıcı çizgi
    st.markdown(
        """
        <a href="https://www.linkedin.com/in/%C3%A7i%C3%A7ek-%C3%BCst%C3%BCn-5a598720b/" target="_blank" style="text-decoration: none; color: black;">
            Çiçek Üstün
        </a>
        """, unsafe_allow_html=True
    )
    st.sidebar.image("Data/oznur.png", width=70)
    st.sidebar.markdown("Öznur Yılmaz: https://www.linkedin.com/in/%C3%B6znur-y%C4%B1lmaz-048649203/")
    st.sidebar.image("Data/emre.png", width=70)
    st.sidebar.markdown("Emre Başer: https://www.linkedin.com/in/emrebaser/")
    st.sidebar.image("Data/cem.png", width=70)
    st.sidebar.markdown("İ. Cem Yılmaz: https://www.linkedin.com/in/i-cem-yilmaz-0a5b7b22b/")

    if ticker:
        try:
            df = yf.download(ticker, period="5y")
            if df.empty:
                modelling_tab.error(f"{ticker} için veri bulunamadı. Lütfen en az 5 yıllık arz geçmişi olan geçerli bir hisse senedi sembolü giriniz.")
            else:
                graph_tab.subheader("Veri Kümesi")
                graph_tab.write(df.head())
                graph_tab.write(df.tail())

                graph_tab.subheader("Zaman Serisi Ayrıştırması")
                ts_decompose(df)

                modelling_tab.subheader("Özellik Mühendisliği")
                df_features = feature_engineering(df)
                modelling_tab.write(df_features.head())
                modelling_tab.write(df_features.tail())

                modelling_tab.subheader("En İyi Model Seçimi")
                result, best_model, mae_value = cv(df_features)
                modelling_tab.write(f"En İyi Model:{best_model}")

                prediction_tab.subheader("Gelecek Tahminleri ve Sapma Hesaplaması")
                mae,mse = deviation(df, best_model)
                prediction_tab.write(f"Ortalama Mutlak Hata (MAE): {mae} - MSE : {mse}")

                future_df = future(df)

                prediction_tab.subheader("Gelecek Tahminleri")
                df_with_predictions, tahminler = prediction(df, future_df, best_model)
                prediction_tab.write(pd.DataFrame({'Tarih': future_df.index, 'Tahmin Edilen Kapanış Fiyatı': tahminler}))

                prediction_tab.subheader("Hisse Grafiği")
                graph(df, tahminler, future_df, ticker,mae,mse)
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()


