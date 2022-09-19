# Machine-Learning-Terapan
# Laporan Proyek Machine Learning - Mukhammad Fahlevi Ali Rafsanjani

## Domain Proyek
Project Machine Learning Terapan : membuat model Predictive Analysis, menggunakan dataset yang berdomain ekonomi mengenai prediksi harga mobil VolkSwagen.
### Latar Belakang
Latar Belakang pemilihan topik ini adalah dikarenakan ingin melihat tingkat penjualan mobil bekas, dimana dalam kasus ini VolkSwagen, dengan fitur - fitur tertentu yang dapat berpenaguh pada nilai di pasar.
Pentingnya bagi para pemilik mobil jika ingin menjual mobilnya perlu untuk melihat harga yang terdapat di pasaran, namun bagi para penjual cukup sulit untuk menentukan harga mobilnya agar mendapatkan harga yang sesuai keinginannya dan juga dapat terjual dengan mudah, oleh karena itu pembuatan prediksi harga yang cocok penting.
## Business Understanding
 ingin melihat tingkat penjualan mobil bekas, dimana dalam kasus ini VolkSwagen, dengan fitur - fitur tertentu yang dapat berpenaguh pada nilai di pasar.

### Problem Statements
- Bagaimana cara mengetahui fitur yang berpengaruh terhadap harga mobil?
- Bagaimana cara menentukan harga mobil dengan data yang ada?
### Goals
- Mengetahui fitur(karakteristik) yang berpengaruh terhadap harga mobil.
- Mengetahui cara untuk menentukan harga mobil dengan menggunakan model machine learning.
### Solution statements
Solusi model yang kami berikan menggunakan Linear dan Polynomial, karena dengan metode tersebut cocok untuk melakukan prediksi terkait harga. 
Untuk model yang digunakan :
- **Linear Model**. Model linier adalah cara untuk menggambarkan variabel respon dalam hal kombinasi linier variabel prediktor. Respon harus berupa variabel kontinu dan paling tidak terdistribusi secara normal.
  - **Kelebihan** Sangat baik untuk model prediksi model regresi terutama dalam mengidentifikasi sekuat apa pengurah yang diberikan oleh variabel independen.
  - **Kekurangan** karena hasil ramalan dari analisis regresi merupakan nilai estimasi, sehingga kemungkinan untuk tidak sesuai dengan data aktual tetaplah ada.
- **Polynomial Model**. Model polinomial adalah alat yang hebat untuk menentukan faktor input mana yang mendorong respons dan ke arah mana. Ini juga merupakan model yang paling umum digunakan untuk analisis eksperimen yang dirancang. Model polinomial kuadratik (orde kedua) untuk dua variabel penjelas.
  - **Kelebihan** dapat fit dengan data karena menyesuaikan garis lemgkungan berdasarkan derajat yang ditentukan.
  - **Kekurangan** mudah juga untuk mendapatkan overfitting.
- **LinearRegression**. Linear Regression cocok dengan model linier dengan koefisien w = (w1, …, wp) untuk meminimalkan jumlah sisa kuadrat antara target yang diamati dalam                             kumpulan data, dan target yang diprediksi oleh pendekatan linier.
  - **Kelebihan** sangat baik untuk model yang memiliki dimensi yg banyak seperti pada dataset titanic dan lainnya dan juga sangat baik untuk kasus regresi.
  - **Kekurangan** data-data yang diukur harus linear untuk memperoleh hasil yang baik
- **DecisionTreeRegressor**. Decision Tree Regressor. Decision Tree membangun model regresi atau klasifikasi dalam bentuk struktur pohon. Ini memecah dataset menjadi subset yang lebih kecil dan lebih kecil sementara pada saat yang sama pohon keputusan terkait dikembangkan secara bertahap. Hasil akhirnya adalah pohon dengan simpul keputusan dan simpul daun.
  - **Kelebihan** Bagus untuk klasifikasi dengan jumlah dimensi yang sedikit tetapi karena regressor sehingga dapat menklasifikasi untuk permasalahan regresi.
  - **Kekurangan** Kurang baik untuk klasifikasi dalam jumlah dimensi yang banyak.  
- **MLPRegressor**. Multi Layer Perceiptron Regressor. Model ini mengoptimalkan kesalahan kuadrat menggunakan LBFGS atau penurunan gradien stokastik.
  - **Kelebihan** Sangat bagus untuk klasifikasi dengan jumlah dimensi yang banyak tetapi karena regressor sehingga dapat menklasifikasi untuk permasalahan regresi.
  - **Kekurangan** Memiliki banyak kelemahan untuk permasalahan kasus regresi karena MLP lebih baik pada kasus klasifikasi.
## Data Understanding
Untuk mengunduh Dataset dapat mengunjungi link berikut [Kaggle Dataset](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes).
Disini menggunakan [*100,000 UK Used Car Data set*](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=cclass.csv) dari situs Kaggle yang berisi data tentang mobil bekas yang terjual di UK dengan variable, mileage, model, engineSize, year, transmission, fuelType, dan price yang menjadi label pada data ini. Dataset ini berisi  3899 dengan 7 kolom dengan 2 kategorikal dan 5 numerikal.
 
Variabel - variabel yang terdapat di Dataset vw (Volkswagen Dataset) :
- model = Volswagen Model(T-Rock, Golf, Polo, T-Cross, Tiguan, Caddy, Etc..). Model mobil Voklswagen
- year = Registration Year (Tahun Registrasi)
- price = Price in Pound Britania (£) (Harga mobil dalam Pound Britania (£))
- transmission = type of gearbox (Automatic, Manual and Semi-Auto) ( Tipe Gearbox mobil)
- mileage = distance used (penggunaan jarak)
- fuelType = Engine Fuel (Diesel, Petrol, Hybird and Other) (Tipe Bensin)
- tax = Road Tax (Pajak jalan)
- mpg = Miles per Galon (mil per galon)
- engineSize= size in litres (ukuran dalam liter)

Data Loading sebagai berikut

|model           |year|price|transmission|mileage|fuelType|tax|mpg  |engineSize|
|----------------|----|-----|------------|-------|--------|---|-----|----------|
| T-Roc          |2019|25000|Automatic   |13904  |Diesel  |145|49.6 |2.0       |
| T-Roc          |2019|26883|Automatic   |4562   |Diesel  |145|49.6 |2.0       |
| T-Roc          |2019|20000|Manual      |7414   |Diesel  |145|50.4 |2.0       |
| T-Roc          |2019|33492|Automatic   |4825   |Petrol  |145|32.5 |2.0       |
| T-Roc          |2019|22900|Semi-Auto   |6500   |Petrol  |150|39.8 |1.5       |
| T-Roc          |2020|31895|Manual      |10     |Petrol  |145|42.2 |1.5       |
| T-Roc          |2020|27895|Manual      |10     |Petrol  |145|42.2 |1.5       |
| T-Roc          |2020|39495|Semi-Auto   |10     |Petrol  |145|32.5 |2.0       |
| T-Roc          |2019|21995|Manual      |10     |Petrol  |145|44.1 |1.0       |
| T-Roc          |2019|23285|Manual      |10     |Petrol  |145|42.2 |1.5       |
|---|---|---|---|---|---|---|---|---|
| Eos            |2015|12495|Manual      |41850  |Diesel  |125|58.9 |2.0       |
| Eos            |2014|8950 |Manual      |58000  |Diesel  |125|58.9 |2.0       |
| Eos            |2006|2995 |Manual      |92640  |Diesel  |200|48.0 |2.0       |
| Eos            |2012|5990 |Manual      |74000  |Diesel  |125|58.9 |2.0       |
| Fox            |2008|1799 |Manual      |88102  |Petrol  |145|46.3 |1.2       |
| Fox            |2009|1590 |Manual      |70000  |Petrol  |200|42.0 |1.4       |
| Fox            |2006|1250 |Manual      |82704  |Petrol  |150|46.3 |1.2       |
| Fox            |2007|2295 |Manual      |74000  |Petrol  |145|46.3 |1.2       |

Dataset tersebut juga dapat dilihat deskripsi statistiknya seperti berikut:

|               year|         price|        mileage|           tax  |        mpg| 
|------|-------------|------------------------------|---------------|------------|
|count | 15157.000000|  15157.000000|   15157.000000|  15157.000000 | 15157.000000|   
|mean  |  2017.255789|  16838.952365|   22092.785644|    112.744277 |    53.753355|   
|std   |     2.053059|   7755.015206|   21148.941635|     63.482617 |    13.642182|   
|min   |  2000.000000|    899.000000|       1.000000|      0.000000 |     0.300000|   
|25%   |  2016.000000|  10990.000000|    5962.000000|     30.000000 |    46.300000|   
|50%   |  2017.000000|  15497.000000|   16393.000000|    145.000000 |    53.300000|   
|75%   |  2019.000000|  20998.000000|   31824.000000|    145.000000 |    60.100000|   
|max   |  2020.000000|  69994.000000|  212000.000000|    580.000000 |   188.300000|   

#### Visualization Data

Apabila jenis data dikategorikan seperti diatas dapat dilihat bentuk tabel dan grafik masing masing data sebagai berikut:

Informasi General Dataset ...
|count|unique|top|freq|
|:---:|:---:|:---:|:---:|
|15157|3|Manual|9417|

![Transmission](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/Transmission.jpg?raw=true)

|count|unique|top|freq|
|:---:|:---:|:---:|:---:|
|15157|4|Petrol|8553|

![fuelType](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/Barplot%20fuelType.jpg?raw=true)

 **Total Pembelian Mobil VolkSwagen Terbanyak**
  Top 3 Mobil Golf, Tiguan dan juga Polo merupakan mobil yang sering digunakan pada kumpulan data dari semua mobil di VW
    ![3 mobil terbanyak](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/terbanyak.png?raw=true)

  **Jumlah Pembelian mobil tiap Tahun**
      - jika dilihat pada tahun 2019 dan 2020 merupakan tahun yang dimana jumlah pembeli mobil VW terbanyak
      ![peningkatan pembelian pertahun](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/pertahun.png?raw=true)

  **Pair Plot**
    disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi
      
   ![Pair Plot](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/pairplot.jpg?raw=true)
   - disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi
   - ada grafik yg memiliki kesamaan yaitu transmission dan fueltype, lalu tax dan engineSize memiliki kesamaan.
      
## Data Preparation
- Sebelum datasetnya di latih atau training, dari model sebelumnya perlu melakukan encoding lalu pemisahan data antara data latih dan test setelah itu melakukan scaling untuk data categorical agar data dapat dilatih.
#### Encoding
Proses encoding atau mengubah data categorikal menjadi 0 dan 1 seperti data pada kolom model, transmission dan fuelType yang datanya didalamnya dipisah kembali menjadi 0 dan 1.

#### Train-Test Split
Proses splitting data atau pembagian dataset menjadi data latih *(train)* dan data uji *(test)* merupakan hal yang harus dilakukan sebelum melakukan pemodelan supervised. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan *train test split* karena untuk efisiensi dan tidak melakukan *data leakage* ketika melakukan scaling. 

#### Standardisasi
Data numerik yang terdapat di dataset perlu dilakukannya proses **Standardisasi** sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritma machine learning dan membuatnya konvergen lebih cepat selain itu menghindari overfitting dan juga data imbalance.

## Modeling
- Pada Proyek yang dibuat, digunakan model algoritma *Machine Learning* yaitu **Linear Regression**,**Decision Tree Regressor**, dan **Multi Layer Perceptron Regressor**. Model tersebut dipilih dikarenakan permasalahan dari model *Machine Learning* yang dibuat adalah permasalahan regresi. hasil dari model yang dipilih akan dibandingkan berdasarkan label yang telah terpilih sebelmunya yaitu *price*. Berikut adalah potongan kode dari model tersebut.

- Lalu melihat hasil model regresi

|No|Features|	Model|	Score|
|:---:|:---:|:---:|:---:|
|0|	Linear	|LinearRegression(copy_X=True, fit_intercept=Tr... |	0.930364|
|1|	Linear	|(DecisionTreeRegressor(ccp_alpha=0.0, criterio... |	0.956395|
|2|	Linear	|MLPRegressor(activation='relu', alpha=0.0001, ... |	-29.835217|


lalu hasil model menggunakan polynomial

|No|Features|	Model|	Score|
|:---:|:---:|:---:|:---:|
|0|	Linear	|LinearRegression(copy_X=True, fit_intercept=Tr... |	0.930364|
|1|	Linear	|(DecisionTreeRegressor(ccp_alpha=0.0, criterio... |	0.956395|
|2|	Linear	|MLPRegressor(activation='relu', alpha=0.0001, ... |	-29.835217|
|3|	Polynomial|	LinearRegression(copy_X=True, fit_intercept=Tr... |	0.930364|
|4|	Polynomial|	(DecisionTreeRegressor(ccp_alpha=0.0, criterio...	| 0.956410|
|5|	Polynomial|	MLPRegressor(activation='relu', alpha=0.0001, ...	| -2.941278|

Dari Tabel dapat dilihat bahwa nilai *RF* lebih mendekati dengan nilai aslinya, sehingga model yang paling cocok adalah *Decision Tree Regressior* menggunakan Polynomial.
## Evaluation
- R-Squared (coefficient of determination).
 -Disini saya menggunakan Metric Evaluation yaitu R^2_score atau R-squared. R-Squared itu sendiri adalah skor terbaik yang mungkin adalah 1,0 dan bisa negatif (karena modelnya bisa sewenang-wenang lebih buruk). Sebuah model konstan yang selalu memprediksi nilai yang diharapkan dari y, mengabaikan fitur input, akan mendapatkan skor 0,0.
  - untuk persamaannya seperti ini
 
    - ![R2-SQUARED MACHINE LEARNING](https://user-images.githubusercontent.com/64582353/135482517-1f589eb6-d59f-4872-8d9d-eddd673c1124.png)
- **Kelebihannya**
  - dapat memprediksi hasil di masa depan atau pengujian hipotesis , berdasarkan informasi terkait lainnya.
  - memberikan ukuran seberapa baik hasil yang diamati direplikasi oleh model, berdasarkan proporsi variasi total hasil yang dijelaskan oleh model.
  - sangat cocok untuk metrics akurasi pada model Regresi.
- **Kekurangan**
  - tidak menunjukan apakah regresi yang benar digunakan
  - tidak dapat memberitahu apakah model tersebut overfit/underfit dan lainnya.

- Dengan menggunakan R2_score dapat memberikan hasil yang baik sebsar 0.953241
