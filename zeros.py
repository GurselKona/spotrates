import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time

def calculate_and_force_save():
    input_file = 'tbp_bulten.xlsx'
    # Dosya ismini değiştirdim ki "Dosya açık" hatasına takılmasın
    output_file = f'sonuc_zero_faizler_V2_{int(time.time())}.xlsx'

    # Verileri saklayacağımız listeler (Hata olsa bile bunları yazdıracağız)
    target_results = []
    verification_data = []

    try:
        print("--- İŞLEM BAŞLIYOR ---")
        
        # 1. VERİ OKUMA
        df_date = pd.read_excel(input_file, header=None, nrows=1)
        ref_date = pd.to_datetime(df_date.iloc[0, 0], dayfirst=True)
        print(f"Değerleme Tarihi: {ref_date.date()}")

        df = pd.read_excel(input_file, header=2)
        instruments = df.iloc[:, 0:11].copy()
        instruments.columns = ['tip', 'vade', 'fiyat_oran'] + [f'col_{i}' for i in range(3, 11)]
        
        instruments['vade'] = pd.to_datetime(instruments['vade'], dayfirst=True)
        instruments['kalan_gun'] = (instruments['vade'] - ref_date).dt.days
        
        instruments = instruments[instruments['kalan_gun'] > 0].sort_values(by='kalan_gun')
        print(f"İşlenecek Enstrüman Sayısı: {len(instruments)}")

        # 2. BOOTSTRAP
        curve_points = [(0, 1.0)]

        for idx, row in instruments.iterrows():
            target_day = row['kalan_gun']
            val = row['fiyat_oran']
            itype = row['tip'].lower()
            
            # Repo/Bono/Tahvil mantığı (Özet)
            cash_flows = []
            market_price = val
            
            if itype == 'repo':
                df_calc = 1 / (1 + val * target_day / 36500)
                curve_points.append((target_day, df_calc))
                verification_data.append({
                    'Tip': 'Repo', 'Vade': row['vade'], 'Fiyat': val, 'DF': df_calc
                })
                curve_points = sorted(curve_points, key=lambda x: x[0])
                continue

            elif itype == 'bono':
                cash_flows.append({'day': target_day, 'amt': 100000})
            
            elif itype == 'tahvil':
                for i in range(3, 9, 2):
                    c_d = row[f'col_{i}']
                    c_a = row[f'col_{i+1}']
                    if pd.notna(c_d) and pd.notna(c_a):
                        c_date_obj = pd.to_datetime(c_d, dayfirst=True)
                        cdays = (c_date_obj - ref_date).days
                        if 0 < cdays <= target_day:
                            cash_flows.append({'day': cdays, 'amt': c_a})
                if cash_flows: 
                    # Eğer son akış vade günündeyse üzerine ekle, değilse yeni ekle
                    cash_flows = sorted(cash_flows, key=lambda x: x['day'])
                    if cash_flows[-1]['day'] == target_day:
                        cash_flows[-1]['amt'] += 100000
                    else:
                        cash_flows.append({'day': target_day, 'amt': 100000})
                else:
                     cash_flows.append({'day': target_day, 'amt': 100000})

            # Solver
            last_day = curve_points[-1][0]
            last_df = curve_points[-1][1]
            
            # --- ARA BOŞLUK KONTROLÜ (Gap Check) ---
            # Eğer yeni eklediğin veri çok uzak bir tarihteyse ve arayı dolduramıyorsa
            # Kod burada "continue" diyip o satırı atlıyor olabilir.
            # Bunu görmek için uyarı ekliyorum:
            intermediate = [f for f in cash_flows if last_day < f['day'] < target_day]
            if len(intermediate) > 1:
                print(f"!!! ATLANDI: {row['vade'].date()} (Çok büyük boşluk var, hesaplanmadı)")
                verification_data.append({
                    'Tip': 'ATLANDI', 'Vade': row['vade'], 'DF': 0, 'Not': 'Veri Boşluğu'
                })
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
                    w = 1.0 if target_day == last_day else (d-last_day)/(target_day-last_day)
                    sum_const += amt * last_df * (1-w)
                    coeff_x += amt * w
            
            if coeff_x != 0:
                new_df = (market_price - sum_known - sum_const) / coeff_x
                curve_points.append((target_day, new_df))
                curve_points = sorted(curve_points, key=lambda x: x[0])
                
                z_rate = (1/new_df - 1)*(36500/target_day) if new_df > 0 else 0
                verification_data.append({
                    'Tip': itype.upper(), 'Vade': row['vade'], 'Fiyat': val, 'DF': new_df, 'Zero': z_rate
                })

        # 3. HEDEFLER
        print("Hedef vadeler hesaplanıyor...")
        final_days = np.array([p[0] for p in curve_points])
        final_dfs = np.array([p[1] for p in curve_points])
        f_interp = interp1d(final_days, final_dfs, kind='linear', fill_value="extrapolate")
        
        t_dates = pd.to_datetime(df.iloc[:, 11].dropna(), dayfirst=True)
        for td in t_dates:
            days = (td - ref_date).days
            if days > 0:
                df_t = float(f_interp(days))
                zr = (1/df_t - 1)*(36500/days) if df_t > 0 else 0
                target_results.append({'Vade': td, 'DF': df_t, 'Zero': zr})

    except Exception as e:
        print(f"\n!!! KRİTİK HATA OLUŞTU: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # HATA OLSA BİLE KAYDET
        print(f"\n--- KAYIT AŞAMASI ---")
        if not target_results and not verification_data:
            print("Kaydedilecek veri oluşmadı! Veri okuma aşamasını kontrol et.")
        else:
            try:
                df1 = pd.DataFrame(target_results)
                df2 = pd.DataFrame(verification_data)
                
                print(f"Yazılacak Hedef Sayısı: {len(df1)}")
                print(f"Yazılacak Eğri Noktası Sayısı: {len(df2)}")
                
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    df1.to_excel(writer, sheet_name='Hedefler', index=False)
                    df2.to_excel(writer, sheet_name='Egri_Verisi', index=False)
                
                print(f"BAŞARILI: '{output_file}' oluşturuldu.")
            except PermissionError:
                print("HATA: Dosya açık olduğu için yazılamadı. Lütfen Excel'i kapatıp tekrar dene.")
            except Exception as save_err:
                print(f"Kayıt sırasında hata: {save_err}")

if __name__ == "__main__":
    calculate_and_force_save()