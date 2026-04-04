import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from catboost import CatBoostClassifier

print("1. Veri Okunuyor ve Temizleniyor...")
df = pd.read_csv('detect_dataset.csv')
df = df.rename(columns={'Output (S)': 'Fault'})
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how='all')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print("2. Sanayi Gerçekliği: %12 İnsan Hatası (Label Noise) Ekleniyor...")
np.random.seed(42)
flip_ratio = 0.12  
flip_indices = np.random.choice(df.index, size=int(len(df) * flip_ratio), replace=False)
df.loc[flip_indices, 'Fault'] = 1 - df.loc[flip_indices, 'Fault']

print("3. Elektriksel Özellik Mühendisliği (Feature Engineering)...")
df['Power_A'] = df['Va'] * df['Ia']
df['Power_B'] = df['Vb'] * df['Ib']
df['Power_C'] = df['Vc'] * df['Ic']
df['Total_Power'] = df['Power_A'] + df['Power_B'] + df['Power_C']
df['Total_Current'] = abs(df['Ia']) + abs(df['Ib']) + abs(df['Ic'])

print("4. Veri Bölünüyor ve Ölçeklendiriliyor...")
X = df.drop(columns=['Fault'])
y = df['Fault']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("5. Nihai CatBoost Modeli Eğitiliyor...")
# Verideki dengesizliği (varsa) otomatik hesaplayıp modele bildiriyoruz
class_0_count = sum(y_train == 0)
class_1_count = sum(y_train == 1)
imbalance_ratio = class_0_count / class_1_count if class_1_count > 0 else 1

final_model = CatBoostClassifier(
    iterations=300, 
    learning_rate=0.05, 
    depth=6, 
    class_weights=[1, imbalance_ratio], # Orijinal veri oranını bozmadan dengeleme
    l2_leaf_reg=5,
    verbose=False, 
    random_seed=42
)
final_model.fit(X_train_scaled, y_train)

# Modeli üretim ortamı için kaydet
final_model.save_model("audentes_kestirimci_bakim.cbm")
print("✅ Model 'audentes_kestirimci_bakim.cbm' olarak kaydedildi!")

print("\n6. THRESHOLD = 0.7 UYGULANIYOR...")
# Olasılıkları al
y_probs = final_model.predict_proba(X_test_scaled)[:, 1]

# STANDART 0.50 YERİNE, KENDİ SEÇTİĞİMİZ 0.7 EŞİĞİNİ ZORUNLU KILIYORUZ
BEST_THRESHOLD = 0.7
y_pred_final = (y_probs >= BEST_THRESHOLD).astype(int)

# Özel F1.5 Skorunu Hesapla
f1_5_final = fbeta_score(y_test, y_pred_final, beta=1.5, zero_division=0)

print("\n" + "="*50)
print(f"🎯 NİHAİ MODEL KARNESİ (THRESHOLD: {BEST_THRESHOLD})")
print("="*50)
print(classification_report(y_test, y_pred_final))
print(f"🌟 F1.5 Skoru (Sanayi Altın Oranı): {round(f1_5_final, 3)}\n")

# 7. SUNUM İÇİN GÖRSELLEŞTİRME (CONFUSION MATRIX)
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.title(f'Kestirimci Bakım Hata Matrisi\n(Karar Eşiği = {BEST_THRESHOLD})', fontsize=14, pad=15)
plt.xlabel('Yapay Zeka Tahmini (0: Normal | 1: Arıza)', fontsize=12)
plt.ylabel('Gerçek Durum (0: Normal | 1: Arıza)', fontsize=12)
plt.show()