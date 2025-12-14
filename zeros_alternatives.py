import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import plotly.graph_objects as go
import time

# --- MODEL FONKSİYONLARI (NSS) ---
def nelson_siegel_svensson(params, t):
    """
    NSS Modeli: 4 Beta ve 2 Tau parametresi ile eğriyi çizer.
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    t = np.maximum(t, 0.001) # 0'a bölünme hatasını önle
    
    term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    term2 = term1 - np.exp(-t / tau1)
    term3 = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
    
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

def fit_nss_model(days, rates):
    """
    Verilen gün ve faiz çiftlerine en uygun NSS parametrelerini bulur.
    """
    def objective(params, t, y):
        model_rates = nelson_siegel_svensson(params, t)
        return np.sum((model_rates - y) ** 2)

    mean_rate = np.mean(rates)
    # Başlangıç tahminleri
    initial_guess = [mean_rate, -mean_rate, 0, 0, 100, 300]
    # Sınırlar (Bounds)
    bounds = ((0, 100), (-100, 100), (-100, 100), (-100, 100), (0.1, 5000), (0.1, 5000))
    
    result = minimize(objective, initial_guess, args=(days, rates), bounds=bounds, method='L-BFGS-B')
    return result.x

# --- ANA PROGRAM ---
def calculate_yield_models():
    input_file = 'tbp_bulten.xlsx'
    output_file = 'sonuc_modelleme.xlsx'

    curve_data = [] 

    print("--- YIELD CURVE MODELLEME (2. Derece Polinom & NSS) ---")

    try:
        # 1. VERİ OKUMA
        try:
            df = pd.read_excel(input_file, header=0)
            raw_date_val = df.columns[0]
            ref_date = pd.to_datetime(raw_date_val, dayfirst=True)
            print(f"Değerleme Tarihi: {ref_date.date()}")
        except:
            print("HATA: Dosya okunamadı. A1 hücresinde tarih olduğundan emin olun.")
            return

        # Kolon Düzeltme
        new_columns = ['tip', 'vade', 'fiyat_oran'] + [f'col_{i}' for i in range(3, 11)]
        if len(df.columns) > 11: new_columns.extend(list(df.columns[11:]))
        
        all_cols = list(df.columns)
        for i in range(min(len(all_cols), 11)): all_cols[i] = new_columns[i]
        df.columns = all_cols

        instruments = df.iloc[:, 0:11].copy()
        instruments['tip'] = instruments['tip'].astype(str).str.strip().str.lower()
        instruments['vade'] = pd.to_datetime(instruments['vade'], dayfirst=True)
        instruments['kalan_gun'] = (instruments['vade'] - ref_date).dt.days
        instruments = instruments[instruments['kalan_gun'] > 0].sort_values(by='kalan_gun')

        print(f"Bootstrap için işlenen kıymet sayısı: {len(instruments)}")

        # 2. BOOTSTRAP (Ham Faizleri Bulma)
        curve_points = [(0, 1.0)] 

        for idx, row in instruments.iterrows():
            target_day = row['kalan_gun']
            val = row['fiyat_oran']
            itype = row['tip']
            
            # Repo
            if 'repo' in itype:
                df_calc = 1 / (1 + val * target_day / 36500)
                curve_points.append((target_day, df_calc))
                curve_points = sorted(curve_points, key=lambda x: x[0])
                z_rate = (1/df_calc - 1)*(36500/target_day)
                curve_data.append({'Tip': 'REPO', 'Gün': target_day, 'Vade': row['vade'], 'Zero': z_rate})
                continue

            # Bono/Tahvil
            cash_flows = []
            if 'bono' in itype:
                cash_flows.append({'day': target_day, 'amt': 100000})
            elif 'tahvil' in itype:
                for i in range(3, 9, 2):
                    c_d = row[f'col_{i}']; c_a = row[f'col_{i+1}']
                    if pd.notna(c_d) and pd.notna(c_a):
                        cdays = (pd.to_datetime(c_d, dayfirst=True) - ref_date).days
                        if 0 < cdays <= target_day: cash_flows.append({'day': cdays, 'amt': c_a})
                cash_flows = sorted(cash_flows, key=lambda x: x['day'])
                if cash_flows:
                    if cash_flows[-1]['day'] == target_day: cash_flows[-1]['amt'] += 100000
                    else: cash_flows.append({'day': target_day, 'amt': 100000})
                else: cash_flows.append({'day': target_day, 'amt': 100000})

            # Solver
            last_day = curve_points[-1][0]; last_df = curve_points[-1][1]
            intermediate = [f for f in cash_flows if last_day < f['day'] < target_day]
            if len(intermediate) > 1: continue 

            days_arr = np.array([p[0] for p in curve_points])
            df_arr = np.array([p[1] for p in curve_points])
            interp = interp1d(days_arr, df_arr, kind='linear', fill_value="extrapolate")
            
            sum_known = 0; sum_const = 0; coeff_x = 0
            for f in cash_flows:
                d = f['day']; amt = f['amt']
                if d <= last_day: sum_known += amt * float(interp(d))
                else:
                    w = 1.0 if target_day == last_day else (d - last_day) / (target_day - last_day)
                    sum_const += amt * last_df * (1 - w)
                    coeff_x += amt * w
            
            if coeff_x == 0: continue
            new_df = (val - sum_known - sum_const) / coeff_x
            curve_points.append((target_day, new_df))
            curve_points = sorted(curve_points, key=lambda x: x[0])
            z_rate = (1/new_df - 1)*(36500/target_day) if new_df > 0 else 0
            
            curve_data.append({'Tip': itype.upper(), 'Gün': target_day, 'Vade': row['vade'], 'Zero': z_rate})

        if not curve_data:
            print("Veri bulunamadı."); return

        # 3. MODELLEME (Regresyonlar & NSS)
        print("Modeller hesaplanıyor...")
        
        df_curve = pd.DataFrame(curve_data)
        X_train = df_curve['Gün'].values
        y_train = df_curve['Zero'].values 

        # A. Lineer Interpolasyon (Piyasa Çizgisi)
        f_linear = interp1d(X_train, y_train, kind='linear', fill_value="extrapolate")

        # B. Lineer Regresyon (Trend)
        lin_coeffs = np.polyfit(X_train, y_train, 1)
        f_lin_reg = np.poly1d(lin_coeffs)

        # C. Polinomial Regresyon (GÜNCELLENDİ: 2. Derece)
        poly_coeffs = np.polyfit(X_train, y_train, 2) # <-- BURASI DEĞİŞTİ
        f_poly_reg = np.poly1d(poly_coeffs)
        print(f"2. Derece Polinom Katsayıları: {poly_coeffs}")

        # D. Nelson-Siegel-Svensson
        nss_params = fit_nss_model(X_train, y_train)
        print(f"NSS Parametreleri: {np.round(nss_params, 2)}")

        # 4. HEDEF VADELER İÇİN TAHMİN
        target_results = []
        if len(df.columns) > 11:
            t_dates = pd.to_datetime(df.iloc[:, 11].dropna(), dayfirst=True)
            for td in t_dates:
                days = (td - ref_date).days
                if days <= 0: continue
                
                target_results.append({
                    'Hedef Vade': td,
                    'Kalan Gün': days,
                    'Interpolasyon (%)': float(f_linear(days)),
                    'Lineer Reg. (%)': float(f_lin_reg(days)),
                    'Polinom (2.Der) (%)': float(f_poly_reg(days)),
                    'NSS (Swenson) (%)': float(nelson_siegel_svensson(nss_params, days))
                })

        # 5. KAYIT
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                pd.DataFrame(target_results).to_excel(writer, sheet_name='Model_Sonuclari', index=False)
                pd.DataFrame(curve_data).to_excel(writer, sheet_name='Ham_Veri', index=False)
            print(f"Dosya kaydedildi: {output_file}")
        except PermissionError:
            print("HATA: Dosya açık olduğu için kayıt yapılamadı.")

        # 6. GRAFİK (PLOTLY)
        print("Grafik oluşturuluyor...")
        fig = go.Figure()

        # X ekseni (Pürüzsüz çizim için)
        max_day = max(df_curve['Gün'].max(), pd.DataFrame(target_results)['Kalan Gün'].max() if target_results else 0) + 30
        x_smooth = np.linspace(1, max_day, 500)

        # Ham Veri
        fig.add_trace(go.Scatter(x=df_curve['Gün'], y=df_curve['Zero'], mode='markers', name='Piyasa Verisi', marker=dict(size=8, color='black')))
        
        # Modeller
        fig.add_trace(go.Scatter(x=x_smooth, y=f_linear(x_smooth), mode='lines', name='Interpolasyon', line=dict(color='gray', dash='dot', width=1)))
        fig.add_trace(go.Scatter(x=x_smooth, y=f_lin_reg(x_smooth), mode='lines', name='Lineer Trend', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=x_smooth, y=f_poly_reg(x_smooth), mode='lines', name='Polinom (2. Derece)', line=dict(color='orange', width=2)))
        
        y_nss = [nelson_siegel_svensson(nss_params, t) for t in x_smooth]
        fig.add_trace(go.Scatter(x=x_smooth, y=y_nss, mode='lines', name='Nelson-Siegel-Svensson', line=dict(color='red', width=3)))

        fig.update_layout(
            title='Getiri Eğrisi Modellemesi (2. Derece Polinom & NSS)',
            xaxis_title='Vade (Gün)',
            yaxis_title='Zero Faiz (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        fig.show()

    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_yield_models()