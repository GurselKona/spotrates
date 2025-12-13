import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time

def calculate_single_header_structure():
    input_file = 'tbp_bulten.xlsx'
    output_file = f'sonuc_zero_faizler_{int(time.time())}.xlsx'

    target_results = []      
    curve_data = []          
    bond_audit_data = []     

    try:
        print("--- HESAPLAMA SÜRECİ (TEK SATIR BAŞLIK YAPISI) ---")
        
        # 1. VERİYİ OKUMA (Header=0 diyerek 1. satırı başlık kabul ediyoruz)
        # Ancak A1 hücresinde 'Tip' yazmıyor, TARİH yazıyor.
        try:
            df = pd.read_excel(input_file, header=0)
        except Exception as e:
            print("HATA: Dosya okunamadı. Dosya açık olabilir veya format bozuk.")
            return

        # 2. TARİHİ ALMA (A1 Hücresi artık DataFrame'in ilk kolonunun ismidir)
        # df.columns[0] bize A1 hücresindeki değeri verir.
        try:
            raw_date_val = df.columns[0]
            ref_date = pd.to_datetime(raw_date_val, dayfirst=True)
            print(f"Değerleme Tarihi (A1'den okundu): {ref_date.date()}")
        except Exception as e:
            print(f"HATA: A1 hücresindeki '{raw_date_val}' değeri tarih formatına çevrilemedi.")
            print("Lütfen A1 hücresinde sadece tarih olduğundan emin olun (Örn: 12.12.2025).")
            return

        # 3. KOLON İSİMLERİNİ DÜZELTME
        # Kodun geri kalanının çalışması için kolon isimlerini standart hale getiriyoruz.
        # İlk kolon (Date idi) -> 'tip' yapıyoruz.
        # Diğer kolonları sırasıyla atıyoruz.
        
        # Mevcut kolon sayısı kadar yeni isim listesi oluştur
        new_columns = ['tip', 'vade', 'fiyat_oran'] + [f'col_{i}' for i in range(3, 11)]
        
        # Eğer dosyada daha fazla kolon varsa (hedef vade gibi), onları da koru
        if len(df.columns) > 11:
            # 11. indeksten sonrasını (hedef vadeleri) olduğu gibi bırak veya isimlendir
            current_extra_cols = list(df.columns[11:])
            new_columns.extend(current_extra_cols)
        
        # Sadece bizim kullanacağımız ilk 11 kolonu yeniden adlandırıyoruz
        # df.columns immutable olduğu için tamamını atamamız lazım
        
        # Tüm kolon isimlerini bir listeye çevir
        all_cols = list(df.columns)
        
        # İlk 11 tanesini değiştir
        for i in range(min(len(all_cols), 11)):
            all_cols[i] = new_columns[i]
            
        df.columns = all_cols

        # 4. VERİ HAZIRLIĞI
        instruments = df.iloc[:, 0:11].copy()
        
        # Tip Temizliği
        instruments['tip'] = instruments['tip'].astype(str).str.strip().str.lower()
        
        # Tarih İşlemleri
        instruments['vade'] = pd.to_datetime(instruments['vade'], dayfirst=True)
        instruments['kalan_gun'] = (instruments['vade'] - ref_date).dt.days
        
        # Vadesi geçmişleri filtrele
        instruments = instruments[instruments['kalan_gun'] > 0].sort_values(by='kalan_gun')
        
        print(f"İşlenecek Satır Sayısı: {len(instruments)}")

        # 5. BOOTSTRAP DÖNGÜSÜ
        curve_points = [(0, 1.0)] 

        for idx, row in instruments.iterrows():
            target_day = row['kalan_gun']
            val = row['fiyat_oran']
            itype = row['tip']
            vade_tarihi = row['vade']
            
            # --- REPO ---
            if 'repo' in itype:
                print(f"-> REPO TESPİT EDİLDİ: {vade_tarihi.date()} (Oran: %{val})")
                
                df_calc = 1 / (1 + val * target_day / 36500)
                curve_points.append((target_day, df_calc))
                curve_points = sorted(curve_points, key=lambda x: x[0])
                
                z_rate = (1/df_calc - 1)*(36500/target_day)
                curve_data.append({
                    'Tip': 'REPO', 'Vade': vade_tarihi, 'Gün': target_day, 
                    'Fiyat/Oran': val, 'DF': df_calc, 'Zero Faiz (%)': z_rate
                })
                continue

            # --- BONO/TAHVİL NAKİT AKIŞLARI ---
            cash_flows = []
            market_price = val
            
            if 'bono' in itype:
                cash_flows.append({'day': target_day, 'amt': 100000, 'date': vade_tarihi})
            
            elif 'tahvil' in itype:
                for i in range(3, 9, 2):
                    c_d = row[f'col_{i}']
                    c_a = row[f'col_{i+1}']
                    if pd.notna(c_d) and pd.notna(c_a):
                        c_date_obj = pd.to_datetime(c_d, dayfirst=True)
                        cdays = (c_date_obj - ref_date).days
                        if 0 < cdays <= target_day:
                            cash_flows.append({'day': cdays, 'amt': c_a, 'date': c_date_obj})
                
                cash_flows = sorted(cash_flows, key=lambda x: x['day'])
                if cash_flows:
                    if cash_flows[-1]['day'] == target_day:
                        cash_flows[-1]['amt'] += 100000
                    else:
                        cash_flows.append({'day': target_day, 'amt': 100000, 'date': vade_tarihi})
                else:
                     cash_flows.append({'day': target_day, 'amt': 100000, 'date': vade_tarihi})

            # --- SOLVER ---
            last_day = curve_points[-1][0]
            last_df = curve_points[-1][1]
            
            intermediate = [f for f in cash_flows if last_day < f['day'] < target_day]
            if len(intermediate) > 1:
                print(f"UYARI: {vade_tarihi.date()} atlandı (Veri boşluğu).")
                continue

            days_arr = np.array([p[0] for p in curve_points])
            df_arr = np.array([p[1] for p in curve_points])
            interp = interp1d(days_arr, df_arr, kind='linear', fill_value="extrapolate")
            
            sum_known = 0; sum_const = 0; coeff_x = 0
            
            for f in cash_flows:
                d = f['day']; amt = f['amt']
                if d <= last_day:
                    sum_known += amt * float(interp(d))
                else:
                    w = 1.0 if target_day == last_day else (d - last_day) / (target_day - last_day)
                    sum_const += amt * last_df * (1 - w)
                    coeff_x += amt * w
            
            if coeff_x == 0: continue
            
            new_df = (market_price - sum_known - sum_const) / coeff_x
            curve_points.append((target_day, new_df))
            curve_points = sorted(curve_points, key=lambda x: x[0])
            
            z_rate = (1/new_df - 1)*(36500/target_day) if new_df > 0 else 0
            
            curve_data.append({
                'Tip': itype.upper(), 'Vade': vade_tarihi, 'Gün': target_day, 
                'Fiyat/Oran': val, 'DF': new_df, 'Zero Faiz (%)': z_rate
            })

            # --- AUDIT ---
            if 'tahvil' in itype or 'bono' in itype:
                check_sum = 0
                for f in cash_flows:
                    d = f['day']; amt = f['amt']
                    final_days = np.array([p[0] for p in curve_points])
                    final_dfs = np.array([p[1] for p in curve_points])
                    final_int = interp1d(final_days, final_dfs, kind='linear', fill_value="extrapolate")
                    used_df = float(final_int(d))
                    pv = amt * used_df
                    check_sum += pv
                    flow_z = (1/used_df - 1)*(36500/d) if used_df > 0 else 0
                    
                    bond_audit_data.append({
                        'Ana Enstrüman': f"{vade_tarihi.date()} {itype.upper()}",
                        'Akış Tarihi': f['date'], 'Gün': d, 'Tutar': amt,
                        'DF': used_df, 'PV': pv, 'Zero (%)': flow_z
                    })
                bond_audit_data.append({'Ana Enstrüman': "--- KONTROL ---", 'PV': check_sum, 'Zero (%)': f"Fark: {check_sum - val:.4f}"})
                bond_audit_data.append({})

        # 6. HEDEFLER
        final_days = np.array([p[0] for p in curve_points])
        final_dfs = np.array([p[1] for p in curve_points])
        f_interp = interp1d(final_days, final_dfs, kind='linear', fill_value="extrapolate")
        
        # Hedef vadenin yeri değişmediyse (12. kolon / index 11)
        if len(df.columns) > 11:
            t_dates = pd.to_datetime(df.iloc[:, 11].dropna(), dayfirst=True)
            for td in t_dates:
                days = (td - ref_date).days
                if days <= 0:
                    df_t, zr = 1.0, 0.0
                else:
                    df_t = float(f_interp(days))
                    zr = (1/df_t - 1)*(36500/days) if df_t > 0 else 0
                target_results.append({'Hedef Vade': td, 'Gün': days, 'DF': df_t, 'Zero (%)': zr})

        # KAYIT
        if curve_data:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                pd.DataFrame(target_results).to_excel(writer, sheet_name='Hedef_Vadeler', index=False)
                pd.DataFrame(curve_data).to_excel(writer, sheet_name='Egri_Verisi', index=False)
                pd.DataFrame(bond_audit_data).to_excel(writer, sheet_name='Tahvil_Detaylari', index=False)
            print(f"\nDosya kaydedildi: {output_file}")
            print("Kontrol: A1 hücresindeki tarih referans alındı, 'Repo' satırı (varsa) işlendi.")
        else:
            print("Hesaplanacak veri bulunamadı.")

    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_single_header_structure()