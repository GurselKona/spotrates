import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time

def calculate_corrected_rates():
    input_file = 'tbp_bulten.xlsx'
    output_file = f'sonuc_zero_faizler_DUZELTILMIS_{int(time.time())}.xlsx'

    target_results = []      
    curve_data = []          
    bond_audit_data = []     

    try:
        print("--- HESAPLAMA SÜRECİ (DÜZELTİLMİŞ) ---")
        
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

        # 2. BOOTSTRAP
        curve_points = [(0, 1.0)] 

        for idx, row in instruments.iterrows():
            target_day = row['kalan_gun']
            val = row['fiyat_oran']
            itype = row['tip'].lower()
            vade_tarihi = row['vade']
            
            # --- A. REPO ---
            if itype == 'repo':
                # Repo oranı zaten yüzde gelir (Örn: 45). Formül: 1 / (1 + 45 * gün / 36500)
                df_calc = 1 / (1 + val * target_day / 36500)
                curve_points.append((target_day, df_calc))
                curve_points = sorted(curve_points, key=lambda x: x[0])
                
                # Zero Rate hesabı (36500 ile çarptığımız için sonuç 45.0 çıkar)
                z_rate = (1/df_calc - 1)*(36500/target_day)
                
                curve_data.append({
                    'Tip': 'REPO', 'Vade': vade_tarihi, 'Gün': target_day, 
                    'Fiyat/Oran': val, 'DF': df_calc, 
                    'Zero Faiz (%)': z_rate # DÜZELTME: *100 Kaldırıldı
                })
                continue

            # --- B. HAZIRLIK ---
            cash_flows = []
            market_price = val
            
            if itype == 'bono':
                cash_flows.append({'day': target_day, 'amt': 100000, 'date': vade_tarihi})
            
            elif itype == 'tahvil':
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

            # --- C. SOLVER ---
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
                'Fiyat/Oran': val, 'DF': new_df, 
                'Zero Faiz (%)': z_rate # DÜZELTME: *100 Kaldırıldı
            })

            # --- D. DETAYLI KUPON DENETİMİ ---
            if itype in ['tahvil', 'bono']:
                check_sum = 0
                for f in cash_flows:
                    d = f['day']
                    amt = f['amt']
                    c_date = f['date']
                    
                    final_days_arr = np.array([p[0] for p in curve_points])
                    final_dfs_arr = np.array([p[1] for p in curve_points])
                    final_interp_func = interp1d(final_days_arr, final_dfs_arr, kind='linear', fill_value="extrapolate")
                    
                    used_df = float(final_interp_func(d))
                    pv_flow = amt * used_df
                    check_sum += pv_flow
                    
                    flow_z_rate = (1/used_df - 1)*(36500/d) if used_df > 0 else 0

                    bond_audit_data.append({
                        'Ana Enstrüman': f"{vade_tarihi.date()} {itype.upper()}",
                        'Enst. Fiyatı': val,
                        'Akış Tipi': 'Anapara+Kupon' if d == target_day else 'Ara Kupon',
                        'Akış Tarihi': c_date,
                        'Akış Günü': d,
                        'Tutar': amt,
                        'Kullanılan DF': used_df,
                        'PV': pv_flow,
                        'Implied Zero (%)': flow_z_rate # DÜZELTME: *100 Kaldırıldı
                    })
                
                bond_audit_data.append({
                    'Ana Enstrüman': "--- KONTROL ---", 'Enst. Fiyatı': val,
                    'PV': check_sum, 'Implied Zero (%)': f"Fark: {check_sum - val:.4f}"
                })
                bond_audit_data.append({})

        # 3. HEDEFLER
        final_days = np.array([p[0] for p in curve_points])
        final_dfs = np.array([p[1] for p in curve_points])
        f_interp = interp1d(final_days, final_dfs, kind='linear', fill_value="extrapolate")
        
        t_dates = pd.to_datetime(df.iloc[:, 11].dropna(), dayfirst=True)
        for td in t_dates:
            days = (td - ref_date).days
            if days <= 0:
                df_t, zr = 1.0, 0.0
            else:
                df_t = float(f_interp(days))
                zr = (1/df_t - 1)*(36500/days) if df_t > 0 else 0
            
            target_results.append({
                'Hedef Vade': td, 'Kalan Gün': days, 
                'Hesaplanan DF': df_t, 
                'Zero Faiz (%)': zr # DÜZELTME: *100 Kaldırıldı
            })

    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if curve_data:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                pd.DataFrame(target_results).to_excel(writer, sheet_name='Hedef_Vadeler', index=False)
                pd.DataFrame(curve_data).to_excel(writer, sheet_name='Egri_Verisi', index=False)
                pd.DataFrame(bond_audit_data).to_excel(writer, sheet_name='Tahvil_Detaylari', index=False)
            
            print(f"\nDosya kaydedildi: {output_file}")
            print("Zero Faizler artık doğru ölçekte (Örn: %45.50 -> 45.50)")

if __name__ == "__main__":
    calculate_corrected_rates()