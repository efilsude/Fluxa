import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, precision_recall_curve, 
                             roc_curve, auc, roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("🚀 Kestirimci Bakım: İleri Seviye (Cost-Based & ROC/PR) Eğitim Başlıyor...\n")

try:
    # 1. Veri Hazırlığı
    df = pd.read_csv("cogaltilmis_model_verisi.csv")
    ayirma_noktasi = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:ayirma_noktasi], df.iloc[ayirma_noktasi:]

    X_train = train_df.drop(['Zaman_Adimi', 'Makine_Durumu'], axis=1)
    X_test = test_df.drop(['Zaman_Adimi', 'Makine_Durumu'], axis=1)
    y_test_gercek = test_df['Makine_Durumu']

    # 2. Ölçeklendirme ve Eğitim
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(random_state=42)
    model.fit(X_train_scaled)

    # Anomali skorlarını al (Büyük skor = Arıza ihtimali yüksek)
    skorlar = -model.decision_function(X_test_scaled)

    # ==========================================
    # 🌟 YENİLİK 1 & 2: ROC-AUC ve PR Curve
    # ==========================================
    roc_auc = roc_auc_score(y_test_gercek, skorlar)
    fpr, tpr, roc_thresholds = roc_curve(y_test_gercek, skorlar)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test_gercek, skorlar)
    pr_auc = auc(recalls, precisions)

    # ==========================================
    # 🌟 YENİLİK 3: COST-BASED OPTİMİZASYON
    # ==========================================
    # Gerçek dünya senaryosu: 
    # Fabrika durursa (False Negative - Arızayı kaçırmak) maliyet = 50.000 TL
    # Boşuna teknisyen giderse (False Positive - Yanlış alarm) maliyet = 1.000 TL
    MALIYET_FN = 50000 
    MALIYET_FP = 1000  

    maliyet_listesi = []
    
    for esik in pr_thresholds:
        tahminler = (skorlar >= esik).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_gercek, tahminler).ravel()
        toplam_maliyet = (fp * MALIYET_FP) + (fn * MALIYET_FN)
        maliyet_listesi.append(toplam_maliyet)

    # Şirkete en az para kaybettirecek Eşik Değerini (Threshold) bul
    en_iyi_index = np.argmin(maliyet_listesi)
    finansal_esik = pr_thresholds[en_iyi_index]
    minimum_maliyet = maliyet_listesi[en_iyi_index]

    # Klasik 0.50 eşiğinde (veya hiç model olmazsa) maliyet ne olurdu? (Kıyaslama için)
    hic_model_olmasa_maliyet = sum(y_test_gercek) * MALIYET_FN # Tüm arızalar kaçar
    kazanc = hic_model_olmasa_maliyet - minimum_maliyet

    print(f"💰 Cost-Based Optimizasyon Tamamlandı!")
    print(f"   - Modeli Kullanmasaydık Çıkacak Fatura: {hic_model_olmasa_maliyet:,} TL")
    print(f"   - Yapay Zeka Sayesinde Düşen Fatura  : {minimum_maliyet:,} TL")
    print(f"   - ŞİRKETİN NET KAZANCI               : {kazanc:,} TL\n")

    # Finansal eşiğe göre son tahminler
    y_pred = (skorlar >= finansal_esik).astype(int)

    # FİNAL RAPORU
    print("="*50)
    print("🏆 İDEATHON FİNAL BAŞARI RAPORU (Finansal Optimize)")
    print("="*50)
    print(f"ROC-AUC Skoru : %{roc_auc*100:.2f} (Modelin gücü)")
    print(f"PR-AUC Skoru  : %{pr_auc*100:.2f} (Dengesiz veri performansı)\n")
    print(classification_report(y_test_gercek, y_pred, target_names=['Normal (0)', 'Arıza (1)']))

    # ==========================================
    # 🎨 GÖRSELLEŞTİRMELER (Sunum İçin 3 Ayrı Grafik)
    # ==========================================
    
    # 1. GRAFİK: ROC ve PR Eğrileri (Yan yana)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_title('ROC Eğrisi')
    ax1.set_xlabel('False Positive Rate (Boş Alarm)')
    ax1.set_ylabel('True Positive Rate (Arıza Yakalama)')
    ax1.legend(loc="lower right")

    ax2.plot(recalls, precisions, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax2.set_title('Precision-Recall Eğrisi (Dengesiz Veri İçin)')
    ax2.set_xlabel('Recall (Duyarlılık)')
    ax2.set_ylabel('Precision (Kesinlik)')
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("1_roc_pr_egrileri.png", dpi=300)

    # 2. GRAFİK: Cost (Maliyet) Optimizasyon Eğrisi
    plt.figure(figsize=(8, 5))
    plt.plot(pr_thresholds, maliyet_listesi, color='red', lw=2)
    plt.axvline(x=finansal_esik, color='green', linestyle='--', label=f'Optimal Eşik: {finansal_esik:.3f}')
    plt.title("Finansal Maliyet Optimizasyonu (Cost-Based)")
    plt.xlabel("Anomaly Alarm Eşiği (Threshold)")
    plt.ylabel("Tahmini Maliyet (TL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("2_maliyet_optimizasyonu.png", dpi=300)

    # 3. GRAFİK: Zaman Serisi Alarm Grafiği (Klasikleşen şov grafiğimiz)
    plt.figure(figsize=(12, 5))
    plt.plot(test_df['Zaman_Adimi'], test_df['Rulman_1_RMS'], label="Titreşim (RMS)", color='blue', alpha=0.5)
    arizalar = test_df.iloc[[i for i, x in enumerate(y_pred) if x == 1]]
    plt.scatter(arizalar['Zaman_Adimi'], arizalar['Rulman_1_RMS'], color='darkred', label="AI Finansal Alarmları", s=15, zorder=5)
    plt.title("Zaman Serisinde Arıza Tespiti (Cost-Optimized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("3_zaman_serisi_alarmlari.png", dpi=300)

    # MODELİ KAYDET (Joblib ile)
    joblib.dump(model, 'isolation_forest_modeli.joblib')
    print("\n✅ Tüm grafikler PNG olarak ve model '.joblib' olarak kaydedildi!")

except Exception as e:
    print(f"HATA: {e}")