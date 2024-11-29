
try:#CSV import
    import csv
except:print("CSV könyvtár importálása sikertelen!"), exit()
try:#Tabulate import
    from tabulate import tabulate # type: ignore # Tabulate install: "pip install tabulate"
except ImportError: print("Hiányzó könyvtár: Tabulate | A könyvtár telepítése cmd: pip install tabulate") , exit()
try:#Numpy import
    import numpy as np
except ImportError: print("Hiányzó könyvtár: Numpy | A könyvtár telepítése cmd: pip install numpy") , exit()
try:#matplotlib import
    import matplotlib.pyplot as plt
except ImportError: print("Hiányzó könyvtár: matplotlib | A könyvtár telepítése cmd: pip install matplotlib") , exit()
try:#sklearnimport
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError: print("Hiányzó könyvtár: sklearn | A könyvtár telepítése cmd: pip install sklearn") , exit()

    
data = []        # Adattábla
data_place = []  # Első oszlop 2. sortol
data_header = [] # Első sor

try:                                                                                            # Fájl beolvasás
    with open('ingatlan.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file,delimiter=";")                                                 # CSV olvasó (soronként olvas), delimiter: elválasztó karakter
        reader2 = list(reader)                                                                  
        data = [row[1:] for idx, row in enumerate(reader2) if idx > 0]                          # row[1:] (első oszlop skip), enumerate(reader) if idx > 0 (első sor skip)
        data = [[int(cell) for cell in row] for row in data]                                    # integerré alakítja a számokat a későbbi műveletekhez
        data_place = [row[0:1] for idx, row in enumerate(reader2) if idx > 0]                   # Első oszlop 2. sortol
        data_place = [[place[0].replace('vármegye', '').strip()] for place in data_place]       # Le szedem a "vármegyét" a nevekből, diagramokknál problémát okoz      
        data_header = [row for idx, row in enumerate(reader2) if idx == 0]                      # Első sor
except Exception as e:
    if type(e) == ValueError:
        print("Nem minden adat alakítható integerre!") , exit()
    else:
        print("Nem sikerült beolvasni az adatokat! Hiba típusa: {e}"), exit()


    
#=================================-- Test --=========================================  

print(tabulate(data, tablefmt="grid"))

for row in data:
    print(row)
for row2 in data_place:
    print(row2)
print(data_header)


#Statisztika

prices = [row[6] for row in data]  # Tegyük fel, hogy az első oszlop tartalmazza az árakat
average_price = np.mean(prices)
median_price = np.median(prices)
min_price = np.min(prices)
max_price = np.max(prices)

# Statisztikai összegzés kiírása
print("\nStatisztikai Összegzés:")
print(f"Átlagár: {average_price:.2f} ezer/m² HUF")
print(f"Mediánár: {median_price:.2f} ezer/m² HUF")
print(f"Minimum ár: {min_price:.2f} ezer/m² HUF")
print(f"Maximum ár: {max_price:.2f} ezer/m² HUF")

#=================================-- Plotting --=========================================


# Adat átváltása X/Y tengelyekre

megye = [place[0] for place in data_place]  # Megyék kiszedése
y_values = [row[0] for row in data]  # Első oszlop családi nm2 árak

# Vonal Plot


plt.figure(figsize=(18, 8))  # Increased figure size
plt.plot(megye, y_values, marker='o', markersize=8, linestyle='-', color='green')
plt.title("Családi ház m² árak megyék szerint", fontsize=16)
plt.xlabel("Megye", fontsize=14)
plt.ylabel("Családi ház átlagár, ezer Ft/m² ", fontsize=14)

# X elforditása 
plt.xticks(rotation=45) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Labelek kicsi eltolással rakjuk fel
for i, value in enumerate(y_values):
    plt.text(i, value + 10, f"{value}", ha='center', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig("line_plot_nm2_csalad_megye.png", dpi=300)  # Lementjük PNG-be
plt.close() 


# Adat átváltása X/Y tengelyekre

y_values = [row[6] for row in data]  # Első oszlop családi nm2 árak

# Plot Diagramm


plt.figure(figsize=(18, 8))  # Increased figure size
plt.plot(megye, y_values, marker='o', markersize=8, linestyle='-', color='green')
plt.title("Átlak lakás m² árak megyék szerint", fontsize=16)
plt.xlabel("Megye", fontsize=14)
plt.ylabel("Lakás ház átlagár, ezer Ft/m² ", fontsize=14)

# X elforditása 
plt.xticks(rotation=45) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Labelek kicsi eltolással rakjuk fel
for i, value in enumerate(y_values):
    plt.text(i, value + 10, f"{value}", ha='center', fontsize=10, color='blue')

plt.tight_layout()
plt.savefig("line_plot_nm2_osszes_megye.png", dpi=300)  # Lementjük PNG-be
plt.close() 


y_values_0 = [row[0] for row in data]  # Az első oszlopból
y_values_2 = [row[2] for row in data]  # A harmadik oszlopból
y_values_4 = [row[4] for row in data]  # Az ötödik oszlopból
y_values_6 = [row[6] for row in data]  # A hetedik oszlopból



# Bar plot készítés (grouped bar chart)
x = np.arange(len(megye))  # X tengely pozíciók
bar_width = 0.2  # Az oszlopok szélessége

plt.figure(figsize=(14, 8))

# Minden sorhoz külön oszlopot hozunk létre
plt.bar(x - 1.5 * bar_width, y_values_0, width=bar_width, label='Családi ', color='skyblue')
plt.bar(x - 0.5 * bar_width, y_values_2, width=bar_width, label='Többlakásos', color='orange')
plt.bar(x + 0.5 * bar_width, y_values_4, width=bar_width, label='Panel', color='green')
plt.bar(x + 1.5 * bar_width, y_values_6, width=bar_width, label='Átlag', color='red')

# Diagram cím, tengelyek
plt.title("Plot diagram - Ingatlanok m² ára alapján", fontsize=16)
plt.xlabel("Megyék", fontsize=14)
plt.ylabel("ezer FT / m² ", fontsize=14)
plt.xticks(x, megye, rotation=45)  # Helyek az X-tengelyen

# Legends
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mentés
plt.tight_layout()
plt.savefig("csoportos_plot_diagram.png", dpi=300)
plt.show()


# ============================= Lineáris Regresszió ============================= #

# Példa adatok: Az X legyen a helyek sorszáma, az Y pedig a 6. oszlop (árak)
X = np.arange(len(data)).reshape(-1, 1)  # Helyek sorszáma (pl. 0, 1, 2, ...)
Y = np.array([row[6] for row in data])  # A 6. oszlop (pl. árak)

# Lineáris regresszió modell
model = LinearRegression()
model.fit(X, Y)

# Regressziós egyenes kiszámítása
Y_pred = model.predict(X)

# ========================= Regresszió Eredményei ========================= #

# Modell metrikák
mse = mean_squared_error(Y, Y_pred)  # Négyzetes hiba
r2 = r2_score(Y, Y_pred)  # R^2 pontosság

print("Lineáris Regresszió Eredményei:")
print(f"Y = {model.coef_[0]:.2f} * X + {model.intercept_:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ============================== Vizualizáció ============================== #

plt.figure(figsize=(14, 8))
plt.scatter(X, Y, color='blue', label='Eredeti Adatok')  # Eredeti adatok
plt.plot(X, Y_pred, color='red', label='Regressziós Egyenes')  # Regressziós egyenes
plt.title("Lineáris Regresszió - Helyek és Árak", fontsize=16)
plt.xlabel("Helyek Sorszáma", fontsize=14)
plt.ylabel("Árak (HUF)", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("linear_regression.png", dpi=300)  # Eredmény mentése
plt.show()