

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

def smart_memory_pdm(train_path):
    df = pd.read_csv(train_path)
    
    # 1. Dinamik İsimlendirme
    num_cols = df.shape[1]
    col_names = ['Unit_ID', 'Cycle', 'Set1', 'Set2', 'Set3'] + [f'S{i}' for i in range(1, num_cols-4)]
    df.columns = col_names[:df.shape[1]]

    # 2. Hedef (45 Günlük Menzil - Zor Mod)
    max_c = df.groupby('Unit_ID')['Cycle'].transform('max')
    df['Target_Binary'] = ((max_c - df['Cycle']) <= 45).astype(int)

    # 3. Hafıza Ekleme (Rolling Features)
    sensor_cols = [c for c in df.columns if c.startswith('S')]
    
    # Pencereleme (window=5)
    df_rolling = df.groupby('Unit_ID')[sensor_cols].rolling(window=5).mean().reset_index(level=0, drop=True)
    df_rolling.columns = [f"{c}_rolling" for c in df_rolling.columns]
    
    # --- KRİTİK DÜZELTME BURADA ---
    # fillna(method='bfill') yerine doğrudan .bfill() kullanıyoruz
    df_final = pd.concat([df, df_rolling], axis=1).bfill().ffill()

    # 4. Sızıntı Engelleme
    leaky_features = ['Set1', 'Set2', 'Set3']
    features = [f for f in df_final.columns if (f.startswith('S')) and f not in leaky_features]
    
    X = df_final[features].copy()
    
    # 5. Agresif Gürültü (%20)
    for col in X.columns:
        X[col] = X[col] + np.random.normal(0, X[col].std() * 0.20, size=len(X))

    y = df['Target_Binary']
    groups = df['Unit_ID']

    # 6. Model: Kısıtlı ama Hafızalı
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4, 
        learning_rate=0.03,
        reg_lambda=100, 
        scale_pos_weight=4, 
        random_state=42
    )

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))

    print(f"🧠 {len(features)} özellik (Hafıza dahil) ile dürüst eğitim başlıyor...")

    for t_idx, v_idx in gkf.split(X, y, groups=groups):
        clf.fit(X.iloc[t_idx], y.iloc[t_idx])
        oof_preds[v_idx] = clf.predict(X.iloc[v_idx])

    # SONUÇLAR
    print("\n" + "="*45)
    print("🏆 HAFIZALI VE DÜRÜST MODEL SONUÇLARI")
    print("="*45)
    print(classification_report(y, oof_preds))
    
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y, oof_preds), annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Final Model: Hafıza + Dürüstlük')
    plt.show()

# Final hamlesini tekrar deneyelim!
if __name__ == "__main__":
    smart_memory_pdm('cleaned_train_data.csv')