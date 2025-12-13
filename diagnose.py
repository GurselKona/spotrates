import pandas as pd

def x_ray_excel():
    input_file = 'tbp_bulten.xlsx'
    print("--- EXCEL RÖNTGENİ (RAW SCAN) ---")
    
    try:
        # Header=None diyerek tüm sayfayı olduğu gibi okuyoruz
        df = pd.read_excel(input_file, header=None)
        
        print(f"Toplam Satır Sayısı: {len(df)}")
        print(f"Toplam Sütun Sayısı: {len(df.columns)}")
        print("-" * 50)
        
        repo_found = False
        
        # Tüm hücreleri tek tek gezip "repo" arayalım
        for row_idx in df.index:
            for col_idx in df.columns:
                cell_value = str(df.at[row_idx, col_idx]).strip().lower()
                
                if 'repo' in cell_value:
                    print(f"\n!!! BULUNDU !!!")
                    print(f"Kelime: '{df.at[row_idx, col_idx]}'")
                    print(f"Konum: Satır {row_idx + 1} | Sütun {col_idx + 1}")
                    
                    # Yorumlama
                    if row_idx < 2:
                        print("ANALİZ: Bu satır Header=2 ayarından ÖNCE geliyor. Bu yüzden veri olarak okunmuyor.")
                        print("ÇÖZÜM: Repo satırını Excel'de 4. satıra veya daha aşağıya taşı.")
                    elif row_idx == 2:
                        print("ANALİZ: Bu satır tam BAŞLIK satırında. Pandas bunu kolon ismi sanıyor.")
                        print("ÇÖZÜM: Repo satırını bir alt satıra taşı.")
                    else:
                        print("ANALİZ: Satır konumu doğru görünüyor. Belki sütun yanlıştır?")
                        if col_idx > 0:
                            print(f"UYARI: Repo kelimesi 1. sütunda değil, {col_idx+1}. sütunda yazıyor.")
                            print("Kod sadece 1. sütundaki (Tip) veriye bakıyor.")

                    repo_found = True

        if not repo_found:
            print("\nSONUÇ: Dosyanın hiçbir yerinde 'repo' kelimesi bulunamadı.")
            print("Sayfa ismini kontrol et. Belki veri 'Sheet1'de değil başka sayfadadır.")
            # Sayfa isimlerini listele
            xl = pd.ExcelFile(input_file)
            print(f"Mevcut Sayfalar: {xl.sheet_names}")

    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    x_ray_excel()