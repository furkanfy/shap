'''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Veri setini oku
df = pd.read_csv("C:/Users/ASUS/Documents/Medical_Appointment/KaggleV2-May-2016.csv")

# Temizlemeler (örnek):
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df = df.drop(['PatientId', 'AppointmentID'], axis=1)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['DayDiff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df.drop(['ScheduledDay', 'AppointmentDay'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('No-show', axis=1)
y = df['No-show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)




# TreeExplainer = RandomForest gibi modeller için
explainer = shap.TreeExplainer(model)

# X_train'deki ilk 100 örnek için SHAP değerleri
shap_values = explainer.shap_values(X_train[:100])

# Genel etki grafiği
shap.summary_plot(shap_values[1], X_train[:100])
plt.show()
'''


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Matplotlib backend'ini ayarla (VS Code için)
plt.switch_backend('TkAgg')  # Alternatif: 'Qt5Agg'

# Veri setini oku
df = pd.read_csv("C:/Users/ASUS/Documents/Medical_Appointment/KaggleV2-May-2016.csv")

# Temizlemeler
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df = df.drop(['PatientId', 'AppointmentID'], axis=1)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['DayDiff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df.drop(['ScheduledDay', 'AppointmentDay'], axis=1)
df = pd.get_dummies(df, drop_first=True)

# Negatif DayDiff değerlerini sıfırla (opsiyonel)
df['DayDiff'] = df['DayDiff'].apply(lambda x: max(0, x))

# Kayıp değerleri kontrol et ve doldur
print("Kayıp değerler:", df.isnull().sum())
df = df.fillna(0)

X = df.drop('No-show', axis=1)
y = df['No-show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# SHAP ile açıklayıcı nesne oluştur
explainer = shap.TreeExplainer(model)

# SHAP değerlerini hesapla (ilk 100 örnek için)
shap_values = explainer.shap_values(X_train[:100])

# SHAP grafiğini çiz ve göster
shap.summary_plot(shap_values[1], X_train[:100])
plt.show()  # BU ŞART!

# Hata kontrolü
print("SHAP değerleri şekli:", shap_values[1].shape)
print("X_train şekli:", X_train[:100].shape)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train[:100])

# Grafik oluştur ama dosyaya kaydet
plt.figure()
shap.summary_plot(shap_values[1], X_train[:100], show=False)
plt.savefig("shap_summary_plot.png", bbox_inches='tight')
print("Grafik shap_summary_plot.png olarak kaydedildi.")


