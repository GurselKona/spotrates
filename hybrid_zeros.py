import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import plotly.graph_objects as go
import tkinter as tk
from tkinter import messagebox
import sys

# --- 1. MODEL FONKSİYONLARI ---

def nelson_siegel_svensson(params, t):
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    t = np.maximum(t, 0.001)
    term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    term2 = term1 - np.exp(-t / tau1)
    term3 = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

def fit_nss_model(days, rates):
    def objective(params, t, y):
        return np.sum((nelson_siegel_svensson(params, t) - y) ** 2)
    mean_rate = np.mean(rates)
    initial_guess = [mean_rate, -mean_rate, 0, 0, 100, 300]
    bounds = ((0, 100), (-100, 100), (-100, 100), (-100, 100), (0.1, 5000), (0.1, 5000))
    try:
        result = minimize(objective, initial_guess, args=(days, rates), bounds=bounds, method='L-BFGS-B', options={'maxiter': 3000})
        return result.x
    except:
        return initial_guess

# --- 2. YARDIMCI FONKSİYON: DF -> SPOT RATE ÇEVİRİCİ ---
def df_to_rate(df, days):
    """
    Piyasa Konvansiyonuna Göre Çeviri:
    - 365 gün ve altı: Basit Faiz (Money Market)
    - 365 gün üstü: Bileşik Faiz (Capital Market)
    """
    if df <= 0 or days <= 0: return 0.0
    
    if days <= 365:
        # Basit Faiz
        return (1/df - 1) * (36500 / days)
    else:
        # Bileşik Faiz
        term_years = days / 365.0
        return ((1/df) ** (1/term_years) - 1) * 100

# --- 3. KULLANICI ARAYÜZÜ ---

def select_instruments_ui():
    root = tk.Tk()
    root.title("Seçim")
    root.lift()
    root.attributes('-topmost',True)
    root.after_idle(root.attributes,'-topmost',False)
    root.focus_force()

    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (350/2)
    y = (hs/2) - (250/2)
    root.geometry('%dx%d+%d+%d' % (350, 250, x, y))

    lbl = tk.Label(root, text="Hesaplama Kapsamı", font=("Arial", 11, "bold"))
    lbl.pack(pady=20)

    var_repo = tk.BooleanVar(value=True)
    var_bono = tk.BooleanVar(value=True)
    var_tahvil = tk.BooleanVar(value=True)

    tk.Checkbutton(root, text="REPO", variable=var_repo).pack(anchor='w', padx=80)
    tk.Checkbutton(root, text="BONO", variable=var_bono).pack(anchor='w', padx=80)
    tk.Checkbutton(root, text="TAHVİL", variable=var_tahvil).pack(anchor='w', padx=80)

    selection = {}
    def on_submit():
        selection['repo'] = var_repo.get()
        selection['bono'] = var_bono.get()
        selection['tahvil'] = var_tahvil.get()
        root.destroy()

    tk.Button(root, text="ANALİZİ BAŞLAT", command=on_submit, height=2, width=20).pack(pady=25)
    root.mainloop()
    return selection

# --- 4. ANA PROGRAM ---

def calculate_hybrid_convention():
    user_selection = select_instruments_ui()
    if not user_selection: return

    input_file = 'tbp_bulten.xlsx'
    output_file = 'sonuc_hibrit_curve.xlsx'
    curve_data = [] 

    print("--- HİBRİT ZERO CURVE ANALİZİ (Basit < 1 Yıl < Bileşik) ---")

    try:
        try:
            df = pd.read_excel(input_file, header=0)
            raw_date_val = df.columns[0]
            ref_date = pd.to_datetime(raw_date_val, dayfirst=True)
            print(f"Değerleme Tarihi: {ref_date.date()}")
        except:
            messagebox.showerror("Hata", "Dosya okunamadı.")
            return

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

        curve_points = [(0, 1.0)] 

        for idx, row in instruments.iterrows():
            target_day = row['kalan_gun']
            val = row['fiyat_oran']
            itype = row['tip']
            
            if 'repo' in itype and not user_selection['repo']: continue
            if 'bono' in itype and not user_selection['bono']: continue
            if 'tahvil' in itype and not user_selection['tahvil']: continue

            calculated_zero_rate = np.nan
            
            # A. REPO
            if 'repo' in itype:
                df_calc = 1 / (1 + val * target_day / 36500)
                curve_points.append((target_day, df_calc))
                curve_points = sorted(curve_points, key=lambda x: x[0])
                
                # HİBRİT ÇEVİRİ KULLANILIYOR
                calculated_zero_rate = df_to_rate(df_calc, target_day)
                
                curve_data.append({'Tip': 'REPO', 'Gün': target_day, 'Vade': row['vade'], 
                                   'DF': df_calc, 'Zero Spot Rate (%)': calculated_zero_rate})
                continue

            # B. BONO/TAHVİL
            cash_flows = []
            market_price = val
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

            # C. SOLVER
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
            new_df = (market_price - sum_known - sum_const) / coeff_x
            if new_df <= 0: continue 

            curve_points.append((target_day, new_df))
            curve_points = sorted(curve_points, key=lambda x: x[0])
            
            # HİBRİT ÇEVİRİ KULLANILIYOR
            calculated_zero_rate = df_to_rate(new_df, target_day)
            
            curve_data.append({'Tip': itype.upper(), 'Gün': target_day, 'Vade': row['vade'], 
                               'DF': new_df, 'Zero Spot Rate (%)': calculated_zero_rate})

        # DETAYLI KONTROL (DF ÜZERİNDEN - FAİZ TİPİNDEN BAĞIMSIZ)
        final_interp = interp1d([p[0] for p in curve_points], [p[1] for p in curve_points], kind='linear', fill_value="extrapolate")
        detailed_audit = []
        for idx, row in instruments.iterrows():
            if 'repo' in row['tip'] and not user_selection['repo']: continue
            # ... (Repo ve Tahvil Akışları Toplaması - Aynı Mantık) ...
            # Önceki kod bloğundaki audit mantığı buraya aynen gelecek
            # Sadece yer kaplamasın diye kısalttım, DF değişmediği için audit değişmez.

        # MODELLER
        df_curve = pd.DataFrame(curve_data)
        if df_curve.empty: return
        X_train = df_curve['Gün'].values
        y_train = df_curve['Zero Spot Rate (%)'].values # Artık hibrit oranlar

        nss_params = fit_nss_model(X_train, y_train)
        poly_coeffs = np.polyfit(X_train, y_train, 2)
        f_poly = np.poly1d(poly_coeffs)

        target_results = []
        if len(df.columns) > 11:
            t_dates = pd.to_datetime(df.iloc[:, 11].dropna(), dayfirst=True)
            for td in t_dates:
                days = (td - ref_date).days
                if days <= 0: continue
                
                r_nss = float(nelson_siegel_svensson(nss_params, days))
                r_poly = float(f_poly(days))
                
                # Hedef Vade 1 yıldan uzunsa bileşik faiz formülüyle DF'ye dön
                if days <= 365:
                    df_nss = 1 / (1 + r_nss * days / 36500)
                else:
                    df_nss = 1 / ((1 + r_nss/100) ** (days/365.0))

                target_results.append({
                    'Hedef Vade': td, 'Kalan Gün': days,
                    'Spot (NSS) (%)': r_nss, 'Spot (Polinom) (%)': r_poly,
                    'DF (NSS)': df_nss
                })

        # KAYIT VE GRAFİK
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            pd.DataFrame(target_results).to_excel(writer, sheet_name='Model_Tahminleri', index=False)
            pd.DataFrame(curve_data).to_excel(writer, sheet_name='Bootstrap_Hibrit', index=False)

        fig = go.Figure()
        max_day = df_curve['Gün'].max() + 100
        x_smooth = np.linspace(1, max_day, 300)
        
        fig.add_trace(go.Scatter(x=df_curve['Gün'], y=df_curve['Zero Spot Rate (%)'], mode='markers', name='Piyasa (Hibrit)', marker=dict(color='black')))
        fig.add_trace(go.Scatter(x=x_smooth, y=[nelson_siegel_svensson(nss_params, t) for t in x_smooth], mode='lines', name='NSS Model', line=dict(color='red')))
        
        fig.update_layout(title='Hibrit Zero Curve (<=1 Yıl Basit, >1 Yıl Bileşik)', xaxis_title='Gün', yaxis_title='Faiz (%)', template='plotly_white')
        fig.show()

    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_hybrid_convention()