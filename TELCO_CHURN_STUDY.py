#######################################################################

### Telco Churn Feature Engineering ###

########################### İŞ PROBLEMİ ###############################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
# ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

####################### VERİ SETİ HİKAYESİ ############################

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
# İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin
# hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.


# 21 Değişken    7043 Gözlem    977.5 KB


# CustomerId       : Müşteri İd’si
# Gender           : Cinsiyet
# SeniorCitizen    : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner          : Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents       : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure           : Müşterinin şirkette kaldığı ay sayısı
# PhoneService     : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines    : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService  : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity   : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup     : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport      : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV      : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies  : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract         : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod    : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges   : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges     : Müşteriden tahsil edilen toplam tutar
# Churn            : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


###################### Görev 1 : Keşifçi Veri Analizi #####################

################### Adım 1: Genel resmi inceleyiniz.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("DATASETS/Telco-Customer-Churn.csv")
df.head()

#   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
# 0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check          29.850        29.85    No
# 1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check          56.950       1889.5    No
# 2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check          53.850       108.15   Yes
# 3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)          42.300      1840.75    No
# 4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check          70.700       151.65   Yes

# Veri Serinin Özet Tablosu
def summarize(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=["dtypes"])
    summary = summary.reset_index()
    summary["Name"] = summary["index"]
    summary = summary[["Name", "dtypes"]]
    summary["Missing"] = df.isnull().sum().values
    summary["Uniques"] = df.nunique().values
    summary["First Value"] = df.loc[0].values
    summary["Second Value"] = df.loc[1].values
    summary["Third Value"] = df.loc[2].values
    summary["Fourth Value"] = df.loc[3].values
    summary["Fifth Value"] = df.loc[4].values

    return summary

summarize(df)
# Dataset Shape: (7043, 21)
# Out[4]:
#                 Name   dtypes  Missing  Uniques       First Value  Second Value     Third Value               Fourth Value       Fifth Value
# 0         customerID   object        0     7043        7590-VHVEG    5575-GNVDE      3668-QPYBK                 7795-CFOCW        9237-HQITU
# 1             gender   object        0        2            Female          Male            Male                       Male            Female
# 2      SeniorCitizen    int64        0        2                 0             0               0                          0                 0
# 3            Partner   object        0        2               Yes            No              No                         No                No
# 4         Dependents   object        0        2                No            No              No                         No                No
# 5             tenure    int64        0       73                 1            34               2                         45                 2
# 6       PhoneService   object        0        2                No           Yes             Yes                         No               Yes
# 7      MultipleLines   object        0        3  No phone service            No              No           No phone service                No
# 8    InternetService   object        0        3               DSL           DSL             DSL                        DSL       Fiber optic
# 9     OnlineSecurity   object        0        3                No           Yes             Yes                        Yes                No
# 10      OnlineBackup   object        0        3               Yes            No             Yes                         No                No
# 11  DeviceProtection   object        0        3                No           Yes              No                        Yes                No
# 12       TechSupport   object        0        3                No            No              No                        Yes                No
# 13       StreamingTV   object        0        3                No            No              No                         No                No
# 14   StreamingMovies   object        0        3                No            No              No                         No                No
# 15          Contract   object        0        3    Month-to-month      One year  Month-to-month                   One year    Month-to-month
# 16  PaperlessBilling   object        0        2               Yes            No             Yes                         No               Yes
# 17     PaymentMethod   object        0        4  Electronic check  Mailed check    Mailed check  Bank transfer (automatic)  Electronic check
# 18    MonthlyCharges  float64        0     1585            29.850        56.950          53.850                     42.300            70.700
# 19      TotalCharges   object        0     6531             29.85        1889.5          108.15                    1840.75            151.65
# 20             Churn   object        0        2                No            No             Yes                         No               Yes


df.Churn.value_counts()
# No     5174
# Yes    1869

df["Churn"].value_counts().plot(kind="bar").set_title("Churn")
plt.show(block=True)

# TotalCharges değişkeninin tipinin sayısal değişkene çevrilmesi
# Çözüm 1:
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
# errors='coerce' değişken içerisindeki boşlukları yok saymayı sağlar

# Çözüm 2
df['TotalCharges'] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)

df["TotalCharges"].dtypes
# dtype('float64')

################### Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')

    # num_but_cat'e sadece gözlem için bakıyoruz cat_cols içinde zaten var
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# CAT_COLS: 17 Adet
# ['gender', 'Partner', 'Dependents', 'PhoneService',
#  'MultipleLines', 'InternetService', 'OnlineSecurity',
#  'OnlineBackup', 'DeviceProtection', 'TechSupport',
#  'StreamingTV', 'StreamingMovies', 'Contract',
#  'PaperlessBilling', 'PaymentMethod', 'Churn', 'SeniorCitizen']

# NUM_COLS: 3 Adet
# ['tenure', 'MonthlyCharges', 'TotalCharges']

# CAT_BUT_CAR: 1 Adet
# ['customerID']

################### Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

df[num_cols].nunique()
# tenure              73
# MonthlyCharges    1585
# TotalCharges      6530

df[cat_cols].nunique()
# gender              2
# Partner             2
# Dependents          2
# PhoneService        2
# MultipleLines       3
# InternetService     3
# OnlineSecurity      3
# OnlineBackup        3
# DeviceProtection    3
# TechSupport         3
# StreamingTV         3
# StreamingMovies     3
# Contract            3
# PaperlessBilling    2
# PaymentMethod       4
# Churn               2
# SeniorCitizen       2
# dtype: int64

df[cat_but_car].nunique()
# customerID    7043

# value_counts
for i in df[cat_cols]:
    print(df[i].value_counts())
    print("______________________________________")

# Male      3555
# Female    3488
# Name: gender, dtype: int64
# _______________________________________________
# No     3641
# Yes    3402
# Name: Partner, dtype: int64
# _______________________________________________
# No     4933
# Yes    2110
# Name: Dependents, dtype: int64
# _______________________________________________
# Yes    6361
# No      682
# Name: PhoneService, dtype: int64
# _______________________________________________
# No                  3390
# Yes                 2971
# No phone service     682
# Name: MultipleLines, dtype: int64
# _______________________________________________
# Fiber optic    3096
# DSL            2421
# No             1526
# Name: InternetService, dtype: int64
# _______________________________________________
# No                     3498
# Yes                    2019
# No internet service    1526
# Name: OnlineSecurity, dtype: int64
# _______________________________________________
# No                     3088
# Yes                    2429
# No internet service    1526
# Name: OnlineBackup, dtype: int64
# _______________________________________________
# No                     3095
# Yes                    2422
# No internet service    1526
# Name: DeviceProtection, dtype: int64
# _______________________________________________
# No                     3473
# Yes                    2044
# No internet service    1526
# Name: TechSupport, dtype: int64
# _______________________________________________
# No                     2810
# Yes                    2707
# No internet service    1526
# Name: StreamingTV, dtype: int64
# _______________________________________________
# No                     2785
# Yes                    2732
# No internet service    1526
# Name: StreamingMovies, dtype: int64
# _______________________________________________
# Month-to-month    3875
# Two year          1695
# One year          1473
# Name: Contract, dtype: int64
# _______________________________________________
# Yes    4171
# No     2872
# Name: PaperlessBilling, dtype: int64
# _______________________________________________
# Electronic check             2365
# Mailed check                 1612
# Bank transfer (automatic)    1544
# Credit card (automatic)      1522
# Name: PaymentMethod, dtype: int64
# _______________________________________________
# No     5174
# Yes    1869
# Name: Churn, dtype: int64
# _______________________________________________
# 0    5901
# 1    1142
# Name: SeniorCitizen, dtype: int64
# _______________________________________________

# unique
for column in df.columns:
    if df[column].dtypes == "object":
        print(f'{column}  : {df[column].unique()}')

# customerID  : ['7590-VHVEG' '5575-GNVDE' '3668-QPYBK' ... '4801-JZAZL' '8361-LTMKD'
#  '3186-AJIEK']
# gender  : ['Female' 'Male']
# Partner  : ['Yes' 'No']
# Dependents  : ['No' 'Yes']
# PhoneService  : ['No' 'Yes']
# MultipleLines  : ['No phone service' 'No' 'Yes']
# InternetService  : ['DSL' 'Fiber optic' 'No']
# OnlineSecurity  : ['No' 'Yes' 'No internet service']
# OnlineBackup  : ['Yes' 'No' 'No internet service']
# DeviceProtection  : ['No' 'Yes' 'No internet service']
# TechSupport  : ['No' 'Yes' 'No internet service']
# StreamingTV  : ['No' 'Yes' 'No internet service']
# StreamingMovies  : ['No' 'Yes' 'No internet service']
# Contract  : ['Month-to-month' 'One year' 'Two year']
# PaperlessBilling  : ['Yes' 'No']
# PaymentMethod  : ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
#  'Credit card (automatic)']
# Churn  : ['No' 'Yes']

################### Adım 4: Hedef değişken analizi yapınız.

# Hedef değişken = Churn

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 1 - Kategorik değişkenlere göre hedef değişkenin ortalaması
def target_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"CHURN_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("_____________________________________")

for col in cat_cols:
    target_cat(df, "Churn", col)

#         CHURN_MEAN
# gender
# 0            0.262
# 1            0.270
# _____________________________________
#          CHURN_MEAN
# Partner
# No            0.330
# Yes           0.197
# _____________________________________
#             CHURN_MEAN
# Dependents
# No               0.313
# Yes              0.155
# _____________________________________
#               CHURN_MEAN
# PhoneService
# No                 0.250
# Yes                0.267
# _____________________________________
#                   CHURN_MEAN
# MultipleLines
# No                     0.251
# No phone service       0.250
# Yes                    0.286
# _____________________________________
#                  CHURN_MEAN
# InternetService
# DSL                   0.190
# Fiber optic           0.419
# No                    0.074
# _____________________________________
#                      CHURN_MEAN
# OnlineSecurity
# No                        0.418
# No internet service       0.074
# Yes                       0.146
# _____________________________________
#                      CHURN_MEAN
# OnlineBackup
# No                        0.399
# No internet service       0.074
# Yes                       0.216
# _____________________________________
#                      CHURN_MEAN
# DeviceProtection
# No                        0.391
# No internet service       0.074
# Yes                       0.225
# _____________________________________
#                      CHURN_MEAN
# TechSupport
# No                        0.416
# No internet service       0.074
# Yes                       0.152
# _____________________________________
#                      CHURN_MEAN
# StreamingTV
# No                        0.335
# No internet service       0.074
# Yes                       0.301
# _____________________________________
#                      CHURN_MEAN
# StreamingMovies
# No                        0.337
# No internet service       0.074
# Yes                       0.300
# _____________________________________
#                 CHURN_MEAN
# Contract
# Month-to-month       0.427
# One year             0.113
# Two year             0.028
# _____________________________________
#                   CHURN_MEAN
# PaperlessBilling
# No                     0.164
# Yes                    0.336
# _____________________________________
#                            CHURN_MEAN
# PaymentMethod
# Bank transfer (automatic)       0.167
# Credit card (automatic)         0.153
# Electronic check                0.453
# Mailed check                    0.192
# _____________________________________
#        CHURN_MEAN
# Churn
# 0           0.000
# 1           1.000
# _____________________________________
#                CHURN_MEAN
# SeniorCitizen
# 0                   0.237
# 1                   0.417
# _____________________________________

# Grafik

sns.set()
for i in range(len(cat_cols)):
    counts = df.groupby([cat_cols[i], 'Churn']).size().unstack()
    ax = (counts.T * 100.0 / counts.T.sum()).T.plot(
        kind='bar', width=0.6, stacked=True, rot=0, figsize=(5, 3))
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2,
                y + height / 2,
                '{:.0f} %'.format(height),
                horizontalalignment='center',
                verticalalignment='center')
    plt.show(block=True)

# Yorum Olarak:
# Erkek ve kadın müşterilerin churn sayısında önemli bir fark olmamakla birlikte;
# - Elektronik çekle ödeme yapan müşterilerin,
# - Aylık sözleşmeleri olan müşterilerin,
# - Fiber optik internet kullanan müşterilerin,
# - Online Security olmayan müşterilerin
# - Cihaz koruması olmayan müşterilerin
# - Teknik desteği olmayan müşterilerin churn oranı yüksek görünmektedir.


# 2 - Hedef Değişkene göre num_cols ort.

# Çözüm 1
df.pivot_table(num_cols, "Churn")
#        MonthlyCharges  TotalCharges  tenure
# Churn
# No             61.265      2555.344  37.570
# Yes            74.441      1531.796  17.979

# Çözüm 2
for col in num_cols:
    print(df.pivot_table(col, "Churn"))
    print("_________________________")
#        tenure
# Churn
# No     37.570
# Yes    17.979
# __________________________
#        MonthlyCharges
# Churn
# No             61.265
# Yes            74.441
# __________________________
#        TotalCharges
# Churn
# No         2555.344
# Yes        1531.796
# __________________________

for col in num_cols:
    df.groupby("Churn").agg({col: "mean"}).plot(kind='bar', rot=0, figsize=(5, 3))
    plt.show(block=True)


################### Adım 5: Aykırı gözlem analizi yapınız.

# Bakılacak
# sns.boxplot(x=df["TotalCharges"], y=df["Churn"])
# plt.show(block=True)


# Gözlem 1
for col in num_cols:
    plt.subplots(figsize=(4, 3))
    sns.boxplot(x=df[col])
    plt.show(block=True)

# Gözlem 2
df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
#       SeniorCitizen   tenure  MonthlyCharges  TotalCharges    Churn
# count       7043.000 7043.000        7043.000      7032.000 7043.000
# mean           0.162   32.371          64.762      2283.300    0.265
# std            0.369   24.559          30.090      2266.771    0.442
# min            0.000    0.000          18.250        18.800    0.000
# 1%             0.000    1.000          19.200        19.900    0.000
# 5%             0.000    1.000          19.650        49.605    0.000
# 10%            0.000    2.000          20.050        84.600    0.000
# 25%            0.000    9.000          35.500       401.450    0.000
# 50%            0.000   29.000          70.350      1397.475    0.000
# 75%            0.000   55.000          89.850      3794.738    1.000
# 90%            1.000   69.000         102.600      5976.640    1.000
# 95%            1.000   72.000         107.400      6923.590    1.000
# 99%            1.000   72.000         114.729      8039.883    1.000
# max            1.000   72.000         118.750      8684.800    1.000

# Gözlem 3
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1) # 0.25
    quartile3 = dataframe[col_name].quantile(q3) # 0.75
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False
# TotalCharges False


# Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

#              n_miss  ratio
# TotalCharges      11  0.160


# Adım 7: Korelasyon analizi yapınız.

# Sadece sayısal değer gösterimi
corr_matrix = df[num_cols].corr()
corr_matrix

#                 tenure  MonthlyCharges  TotalCharges
# tenure           1.000           0.248         0.826
# MonthlyCharges   0.248           1.000         0.651
# TotalCharges     0.826           0.651         1.000


# Grafiksel olarak gösterimi (Churn dahil)
plt.figure(figsize=(5, 3))
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
# cmap    = colormap (renk kodu) (PuBuGn, YlOrRd, YlGnBu, terrain .....)
# annot   = veri değrlerini gösterir.
# figsize = Şeklin  kenarlarının inc cinsinden uzunluğudur.
plt.show(block=True)

df.corrwith(df["Churn"]).sort_values(ascending=False)

###################### Görev 2 : Feature Engineering #####################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# Eksik değer silme işlemi

na_cols = missing_values_table(df, True)
#               n_miss  ratio
# TotalCharges      11  0.160

# Eksik değer sayısı çok az ise direkt olarak silebiliriz.
df.dropna(inplace=True)

# Ortalama değer de atayabiliriz.
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Adım 2: Yeni değişkenler oluşturunuz.

df.head()

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("_________________________________________________")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

#         gender  Ratio
# Male      3549 50.469
# Female    3483 49.531
# _________________________________________________
#      Partner  Ratio
# No      3639 51.749
# Yes     3393 48.251
# _________________________________________________
#      Dependents  Ratio
# No         4933 70.151
# Yes        2099 29.849
# _________________________________________________
#      PhoneService  Ratio
# Yes          6352 90.330
# No            680  9.670
# _________________________________________________
#                   MultipleLines  Ratio
# No                         3385 48.137
# Yes                        2967 42.193
# No phone service            680  9.670
# _________________________________________________
#              InternetService  Ratio
# Fiber optic             3096 44.027
# DSL                     2416 34.357
# No                      1520 21.615
# _________________________________________________
#                      OnlineSecurity  Ratio
# No                             3497 49.730
# Yes                            2015 28.655
# No internet service            1520 21.615
# _________________________________________________
#                      OnlineBackup  Ratio
# No                           3087 43.899
# Yes                          2425 34.485
# No internet service          1520 21.615
# _________________________________________________
#                      DeviceProtection  Ratio
# No                               3094 43.999
# Yes                              2418 34.386
# No internet service              1520 21.615
# _________________________________________________
#                      TechSupport  Ratio
# No                          3472 49.374
# Yes                         2040 29.010
# No internet service         1520 21.615
# _________________________________________________
#                      StreamingTV  Ratio
# No                          2809 39.946
# Yes                         2703 38.439
# No internet service         1520 21.615
# _________________________________________________
#                      StreamingMovies  Ratio
# No                              2781 39.548
# Yes                             2731 38.837
# No internet service             1520 21.615
# _________________________________________________
#                 Contract  Ratio
# Month-to-month      3875 55.105
# Two year            1685 23.962
# One year            1472 20.933
# _________________________________________________
#      PaperlessBilling  Ratio
# Yes              4168 59.272
# No               2864 40.728
# _________________________________________________
#                            PaymentMethod  Ratio
# Electronic check                    2365 33.632
# Mailed check                        1604 22.810
# Bank transfer (automatic)           1542 21.928
# Credit card (automatic)             1521 21.630
# _________________________________________________
#    Churn  Ratio
# 0   5163 73.422
# 1   1869 26.578
# _________________________________________________
#    SeniorCitizen  Ratio
# 0           5890 83.760
# 1           1142 16.240
# _________________________________________________


df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

# Değişken 1:
# Cinsiyetleri, kendi içinde yaşsal olarak gruplandırma yaparak yeni değişken oluşturalım.
df.groupby("gender").agg({"Churn": ["mean", "count"]})
#        Churn
#         mean count
# gender
# 0      0.262  3555
# 1      0.269  3488

df.loc[((df['gender'] == 0) & (df["SeniorCitizen"] == 1)), 'SENIOR/YOUNG_GENDER'] = "senior_male"
df.loc[((df['gender'] == 0) & (df["SeniorCitizen"] == 0)), 'SENIOR/YOUNG_GENDER'] = "young_male"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"] == 1)), 'SENIOR/YOUNG_GENDER'] = "senior_female"
df.loc[((df['gender'] == 1) & (df["SeniorCitizen"] == 0)), 'SENIOR/YOUNG_GENDER'] = "young_female"
df.groupby("SENIOR/YOUNG_GENDER").agg({"Churn": ["mean", "count"]})
#                     Churn
#                      mean count
# SENIOR/YOUNG_GENDER
# senior_female       0.423   568
# senior_male         0.411   574
# young_female        0.239  2920
# young_male          0.233  2981

# Değişken 2:
# # Telefon Servislerini, kendi içinde cinsiyete göre gruplandırma yaparak yeni değişken oluşturalım.

df.groupby("PhoneService").agg({"Churn": ["mean", "count"]})
#              Churn
#               mean count
# PhoneService
# No           0.249   682
# Yes          0.267  6361

df.loc[((df['gender'] == 0) & (df["PhoneService"] == "Yes")), 'PHONE_SER_GENDER'] = "phone_ser_male"
df.loc[((df['gender'] == 0) & (df["PhoneService"] == "No")), 'PHONE_SER_GENDER'] = "no_phone_ser_male"
df.loc[((df['gender'] == 1) & (df["PhoneService"] == "Yes")), 'PHONE_SER_GENDER'] = "phone_service_female"
df.loc[((df['gender'] == 1) & (df["PhoneService"] == "No")), 'PHONE_SER_GENDER'] = "no_phone_ser_female"
df.groupby("PHONE_SER_GENDER").agg({"Churn": ["mean", "count"]})
#                      Churn
#                       mean count
# PHONE_SER_GENDER
# no_phone_ser_female  0.242   331
# no_phone_ser_male    0.256   351
# phone_ser_male       0.262  3204
# phone_service_female 0.272  3157

# Değişken 3:
# Kontrantları, kendi içinde cinsiyete göre gruplandırma yaparak yeni değişken oluşturalım.

df.groupby("Contract").agg({"Churn": ["mean", "count"]})
#                Churn
#                 mean count
# Contract
# Month-to-month 0.427  3875
# One year       0.113  1473
# Two year       0.028  1695

df.loc[((df['gender'] == 0) & (df["Contract"] == "Month-to-month")), 'GENDER_CONTRACT'] = "male_monthly_contract"
df.loc[((df['gender'] == 1) & (df["Contract"] == "Month-to-month")), 'GENDER_CONTRACT'] = "female_monthly_contract"
df.groupby("GENDER_CONTRACT").agg({"Churn": ["mean", "count"]})
#                         Churn
#                          mean count
# GENDER_CONTRACT
# female_monthly_contract 0.437  1925
# male_monthly_contract   0.417  1950

# Değişken 4:
# Ödeme yöntemlerini, kendi içinde cinsiyete göre gruplandırma yaparak yeni değişken oluşturalım.

df.groupby("PaymentMethod").agg({"Churn": ["mean", "count"]})
#                           Churn
#                            mean count
# PaymentMethod
# Bank transfer (automatic) 0.167  1544
# Credit card (automatic)   0.152  1522
# Electronic check          0.453  2365
# Mailed check              0.191  1612

df.loc[((df['gender'] == 0) & (df["PaymentMethod"] == "Electronic check")), 'GENDER_PAYMENT'] = "male_electronic_check_pay"
df.loc[((df['gender'] == 1) & (df["PaymentMethod"] == "Electronic check")), 'GENDER_PAYMENT'] = "female_electronic_check_pay"
df.groupby("GENDER_PAYMENT").agg({"Churn": ["mean", "count"]})
#                             Churn
#                              mean count
# GENDER_PAYMENT
# female_electronic_check_pay 0.446  1170
# male_electronic_check_pay   0.459  1195

############## MODELLEME #######################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoder

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

df.head()

# One-Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

# Fantazi Amaçlı (Evde denemeyin :) )
def heatMap(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", linewidths=.2, annot_kws={"size": 6})
    plt.xticks(range(len(corr.columns)), corr.columns, size=7);
    plt.yticks(range(len(corr.columns)), corr.columns, size=7)
    plt.show(block=True)

heatMap(df)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

df[num_cols].nunique()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20, figsize=(5, 3))
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()
#   tenure  MonthlyCharges  TotalCharges
# 0  -1.280          -1.162        -0.994
# 1   0.064          -0.261        -0.174
# 2  -1.240          -0.364        -0.960
# 3   0.512          -0.748        -0.195
# 4  -1.240           0.196        -0.940

df.head()
df.shape


# Adım 5: Model oluşturunuz.

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.790521327014218