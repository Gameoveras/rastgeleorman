
# Gerekli kütüphaneleri yükleme
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oku ve "Art" ve "Tablets" kategorisine ait olanları seç
data = pd.read_csv('amazon.csv', sep=";", decimal=",")
selected_categories = data[data['Category'].str.lower().isin(['art', 'tables'])]

# Gerekli sütunları seç
selected_categories = selected_categories[["Category", "Sales", "Quantity", "Profit"]]

# Features ve Label'ları belirleme
X = selected_categories[["Sales", "Quantity", "Profit"]]
y = selected_categories["Profit"]

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele Ormanlar modelini oluşturalım
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, rf_predictions)
rmse = np.sqrt(mse)
print("Ortalama Karesel Hata Kökü: ", rmse)

# Tüm veri seti üzerinde tahmin yapalım
all_predictions = rf_regressor.predict(X)

# Elde edilen tahminleri orijinal veri setine ekleyelim
selected_categories["Predicted_Profit"] = all_predictions

# Her kategori için toplam tahmini karı hesaplayalım
predicted_profits = selected_categories.groupby("Category")["Predicted_Profit"].sum()

# Hangi kategorinin daha yüksek kar elde edebileceğini belirleyelim
higher_profit_category = predicted_profits.idxmax()

print("Gelecekte daha yüksek kar elde edebilecek kategori:", higher_profit_category)

# "Sanat" ve "Masa" kategorilerinin gerçek ve tahmini karları
art_actual_profits = selected_categories[selected_categories["Category"] == "Art"]["Profit"]
art_predicted_profits = selected_categories[selected_categories["Category"] == "Art"]["Predicted_Profit"]

tablets_actual_profits = selected_categories[selected_categories["Category"] == "Tables"]["Profit"]
tablets_predicted_profits = selected_categories[selected_categories["Category"] == "Tables"]["Predicted_Profit"]

# Grafikleştirme
plt.figure(figsize=(10, 6))


# "Sanat" kategorisi için gerçek ve tahmini karları gösteren çizgi grafiği
sns.lineplot(x=art_actual_profits.index, y=art_actual_profits, label="Sanat Gerçek Kâr", marker='o')
sns.lineplot(x=art_predicted_profits.index, y=art_predicted_profits, label="Sanat Tahmini Kâr", marker='o')

# "Masa" kategorisi için gerçek ve tahmini karları gösteren çizgi grafiği
sns.lineplot(x=tablets_actual_profits.index, y=tablets_actual_profits, label="Masa Gerçek Kâr", marker='o')
sns.lineplot(x=tablets_predicted_profits.index, y=tablets_predicted_profits, label="Masa Tahmini Kâr", marker='o')

plt.title("Sanat ve Masa Kategorileri için Gerçek ve Tahmini Karlar")
plt.xlabel("Örnek Endeks")
plt.ylabel("Kâr")
plt.legend()
plt.show()


 
