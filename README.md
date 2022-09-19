# Machine-Learning-Terapan
# Laporan Proyek Machine Learning - Rizqi Alpiansyah

## Domain Proyek
Project Machine Learning Terapan : membuat model Predictive Analysis, menggunakan dataset yang berdomain kesehatan mengenai diabetes.
### Latar Belakang
Diabetes terjadi ketika glukosa darah Anda, umumnya dikenal sebagai gula darah, terlalu tinggi, Anda mengembangkan diabetes. Sumber energi utama Anda, glukosa darah, diperoleh dari makanan yang Anda makan. Glukosa dari makanan diangkut ke dalam sel Anda oleh hormon insulin, yang diproduksi oleh pankreas. Tubuh Anda kadang-kadang menghasilkan insulin yang tidak mencukupi atau tidak ada sama sekali, atau menggunakan insulin dengan buruk. Setelah itu, glukosa tetap berada dalam sirkulasi Anda dan tidak masuk ke dalam sel Anda.
Seiring waktu, memiliki terlalu banyak glukosa dalam darah Anda dapat menyebabkan masalah kesehatan. Meskipun tidak ada obat untuk diabetes, ada beberapa hal yang dapat Anda lakukan untuk mengelolanya dan tetap sehat.

Diabetes kadang-kadang disebut sebagai "diabetes ambang" atau "sentuhan gula." Ungkapan ini menyiratkan bahwa seseorang tidak benar-benar menderita diabetes atau memiliki kasus yang lebih ringan, namun diabetes selalu memiliki konsekuensi yang menghancurkan.

Referensi: [Epidemiology of Diabetes and Diabetes-Related Complications](https://academic.oup.com/ptj/article/88/11/1254/2858146)

## Business Understanding
### Problem Statements
- Variable apa saja yang berpengaruh pada diagnosa diabetes seseorang ?
- Apakah variable yang ada pada dataset dapat tepat memprediksi diabetes secara tepat ?
 
### Goals
- Mengetahui variable apa saja yang dapat mempengaruhi seseorang terkena diabetes
- Membuat model mechine learning sebaik mungkin dengan score yang tinggi berdasarkan variable variable yang ada 

### Solution Statements
Untuk mendapatkan model mechine learning pada prediksi diabetes yang saya buat. saya membandingkan 4 algortima LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GaussianNB
- LogisticRegression merupakan analisis regresi yang tepat untuk dilakukan ketika variabel dependen bersifat dikotomis (biner). Seperti semua analisis regresi, regresi logistik adalah analisis prediktif. Regresi logistik digunakan untuk menggambarkan data dan untuk menjelaskan hubungan antara satu variabel biner dependen dan satu atau lebih variabel independen nominal, ordinal, interval, atau tingkat rasio.
- DecisionTreeClassifier adalah algoritma yang melakukan pendekatan berbasis aturan untuk masalah klasifikasi dan regresi. Mereka menggunakan nilai di setiap fitur untuk membagi kumpulan data ke titik di mana semua titik data yang memiliki kelas yang sama dikelompokkan bersama.
- RandomForestClassifier adalah sebuah algoritma meta estimator yang cocok dengan sejumlah pengklasifikasi pohon keputusan pada berbagai sub-sampel dari dataset dan menggunakan rata-rata untuk meningkatkan akurasi prediksi dan kontrol over-fitting. Ukuran sub-sampel dikontrol dengan parameter max_samples jika bootstrap=True (default), jika tidak, seluruh dataset digunakan untuk membangun setiap pohon.
- GaussianNB adalah algoritma yang mendukung fitur dan model bernilai kontinu masing-masing sesuai dengan distribusi Gaussian (normal). Pendekatan untuk membuat model sederhana adalah dengan mengasumsikan bahwa data dideskripsikan oleh distribusi Gaussian tanpa kovarians (dimensi independen) antar dimensi.

## Data Understanding
Dataset yang saya gunakan merupakan dataset survey 253.680 orang responden yang sudah bersih. Dataset ini terdiri dari 22 kolom (variabel) yang semuanya bertipe data float64.
Sumber: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv).

Variabel - variabel yang terdapat di Dataset :
- Diabetes_binary = 0 = tidak diabetes 1 = prediabetes 2 = diabetes
- HighBP: 0 = no high BP 1 = high BP
- HighChol: 0 = no high cholesterol 1 = high cholesterol
- CholCheck: 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years
- BMI: Body Mass Index
- Smoker: Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes
- Stroke: (Ever told) you had a stroke. 0 = no 1 = yes
- HeartDiseaseorAttack: coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes
- PhysActivity: physical activity in past 30 days - not including job 0 = no 1 = yes
- Fruits: Consume Fruit 1 or more times per day 0 = no 1 = yes
- Veggies: Consume Vegetables 1 or more times per day 0 = no 1 = yes
- HvyAlcoholConsump: (adult men >=14 drinks per week and adult women>=7 drinks per week) 0 = no 1 = yes
- AnyHealthcare: Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes
- NoDocbcCost: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes
- GenHlth: Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor
- MentHlth: days of poor mental health scale 1-30 days
- PhysHlth: physical illness or injury days in past 30 days scale 1-30
- DiffWalk: Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes
- Sex: 0 = female 1 = male
- Age: 13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older
- Education: Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = elementary etc.
- Income: Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more
 
Data Visualisasi
hal pertama yang kita lihat yaitu berapa banyak jumlah yang terkena diabetes 
|count|No diabetes|diabetes|
|:---:|:---------:|:------:|
|100%  |86.07% |13.93%|


![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/piechart.png?raw=True)

melihat distribusi variable yang ada dalam data set

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/distribusi.png?raw=True)


Pair Plot disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/pairplot.png?raw=True)
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/pairplot1.png?raw=True)
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/pairplot2.png?raw=True)
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/pairplot3.png?raw=True)

setelah itu kita melihat korelasi pada setiap variable yang ada didalam data set 
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/korelasi.png?raw=True)

selain itu juga kita mengecek untuk apakah ada data outlier 
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/before%20boxplot.png?raw=True)

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/outlier.jpeg?raw=True)

bisa dilihat dari gambar di atas data yang dipakai harus ada diantara -1.5(Q3-Q1) dan 1.5(Q3-Q1) atau bisa ditulis 
-1.5(Q3-Q1)<data<1.5(Q3-Q1)

setelah data outlier dibersihkan 
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/after%20boxplot.png?raw=True)

## Data Preparation
Sebelum datasetnya di latih atau training, dari model sebelumnya perlu melakukan encoding lalu pemisahan data antara data latih dan test setelah itu melakukan scaling untuk data categorical agar data dapat dilatih.


#### Standardisasi 
Data numerik yang terdapat di dataset perlu dilakukannya proses Standardisasi sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritma machine learning dan membuatnya konvergen lebih cepat selain itu menghindari overfitting dan juga data imbalance. pada proyek kali ini saya menggunakan fungsi MaxAbsScaler(). fungsi ini berguna untuk melakukan standarisasi pada data. 

#### Train-Test Split
Proses splitting data atau pembagian dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum melakukan pemodelan supervised. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan train test split karena untuk efisiensi dan tidak melakukan data leakage ketika melakukan scaling. pada proyek kali ini kita membagi data menjadi 80:20 dengan random state = 93 


## Modeling

pada proyek yang dibuat kali ini, digunakan model algoritma mechine learning yaitu Machine Learning yaitu LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GaussianNB. model tersebut dipilih karena tujuanya ingin memprediksi binary classification. hasil dari model yang kita buat akan dibandingkan berdasarkan variable yang telah terpilih yaitu Diabetes_binary

- pada LogisticRegression kita hanya menggunakan fungsi fit tanpa tambahan parameterlain 
`logisticRegressionModel = LogisticRegression().fit(X_train, y_train)`
  - kelebihan pada algoritma ini yaitu Output memiliki interpretasi probabilistik yang bagus, dan algoritme dapat diatur untuk menghindari overfitting. Model logistik dapat diperbarui dengan mudah dengan data baru menggunakan penurunan gradien stokastik.
dari algoritma diatas pada saat proses modeling dan evaluasi semua algoritma yang digunakan bekerja dengan cukup baik dalam hal memprediksi diabetes. hal ini dapat ditunjukan nilai akurasi, MSE, dan RMSE pada saat training dan testing. namun pada akhirnya score yang paling tinggi yaitu ketika menggunakan algoritma RandomForestClassifier 
  - kekurangan Regresi logistik cenderung berperforma buruk bila ada beberapa atau tidak linier batas keputusan. Mereka tidak cukup fleksibel untuk menangkap lebih kompleks secara alami

- pada DecisionTreeClassifier pada proyek ini menggunakan parameter tambahan yaitu min_samples_split.  min sample split sendiri yaitu jumlah minimum sampel yang diperlukan untuk membagi simpul internal dan code yang ada didalam proyek yaitu `treeModel = DecisionTreeClassifier(min_samples_split = 60).fit(X_train, y_train)`
  - kelebihan dari algoritma ini yaitu menghasilkan aturan yang dapat dimengerti, dapat melakukan klasifikasi tanpa memerlukan banyak perhitungan, mampu menangani variabel kontinu dan kategorikal, dapat memberikan indikasi yang jelas tentang bidang mana yang paling penting untuk prediksi atau klasifikasi.
  - kekurangan kurang tepat untuk tugas estimasi di mana tujuannya adalah untuk memprediksi nilai atribut kontinu, rentan terhadap kesalahan dalam masalah klasifikasi dengan banyak kelas dan jumlah contoh pelatihan yang relatif kecil, membutuhkan pekerjaan komputasi yang ekstensif. Setiap bidang pemisahan potensial pada setiap node perlu diurutkan sebelum pemisahan idealnya dapat ditentukan. Beberapa algoritma menggunakan kombinasi bidang, oleh karena itu perlu untuk mencari bobot kombinasi terbaik. Karena banyak kandidat sub-pohon harus dibuat dan dievaluasi, algoritma pemangkasan juga bisa mahal.

- pada algoritma RandomForestClassifier sama seperti DecisionTreeClassifier menggunakan parameter tambahan yaitu min_samples_split. tujuanya untuk memberikan nilai umlah minimum sampel yang diperlukan untuk membagi simpul internal. pengaplikasianya dalam proyek kali ini yaitu `forestModel = RandomForestClassifier(min_samples_split = 60).fit(X_train, y_train)` 
  - kelebihanya random forest berdasarkan algoritma bagging dan menggunakan teknik Ensemble Learning. Ini menciptakan sebanyak mungkin pohon pada subset data dan menggabungkan output dari semua pohon. Dengan cara ini mengurangi masalah overfitting di pohon keputusan dan juga mengurangi varians dan karena itu meningkatkan akurasi, dapat digunakan untuk menyelesaikan masalah klasifikasi maupun regresi, dengan baik dengan variabel kategorikal dan kontinu, dapat secara otomatis menangani nilai yang hilang, Tidak diperlukan penskalaan fitur (standarisasi dan normalisasi) dalam kasus Hutan Acak karena menggunakan pendekatan berbasis aturan alih-alih perhitungan jarak.
  - kekuranganya menciptakan banyak pohon (tidak seperti hanya satu pohon dalam kasus pohon keputusan) dan menggabungkan hasilnya. Secara default, ini membuat 100 pohon di pustaka sklearn Python. Untuk melakukannya, algoritma ini membutuhkan lebih banyak daya dan sumber daya komputasi. Di sisi lain pohon keputusan sederhana dan tidak memerlukan begitu banyak sumber daya komputasi, membutuhkan lebih banyak waktu untuk berlatih dibandingkan dengan pohon keputusan karena menghasilkan banyak pohon (bukan satu pohon dalam kasus pohon keputusan) dan membuat keputusan berdasarkan suara terbanyak.

- pada algoritma GaussianNB dalam proyek kali ini tidak menambahkan paramater lain dan pengaplikasianya pada proyek kali ini yaitu  `bayesModel = GaussianNB().fit(X_train, y_train)` 
  - kelebihan Algoritma ini bekerja dengan cepat dan dapat menghemat banyak waktu, Naive Bayes cocok untuk memecahkan masalah prediksi multi-kelas, Jika asumsi independensi fiturnya benar, ia dapat berkinerja lebih baik daripada model lain dan membutuhkan lebih sedikit data pelatihan, Naive Bayes lebih cocok untuk variabel input kategoris daripada variabel numerik.
  - kekuranganya Naive Bayes mengasumsikan bahwa semua prediktor (atau fitur) adalah independen, jarang terjadi dalam kehidupan nyata. Ini membatasi penerapan algoritme ini dalam kasus penggunaan di dunia nyata, Algoritme ini menghadapi 'masalah frekuensi nol' di mana ia memberikan probabilitas nol untuk variabel kategoris yang kategorinya dalam kumpulan data uji tidak tersedia dalam kumpulan data pelatihan. Akan lebih baik jika Anda menggunakan teknik smoothing untuk mengatasi masalah ini, Estimasinya bisa salah dalam beberapa kasus, jadi Anda tidak boleh menganggap hasil probabilitasnya terlalu serius.

dari model yang akan digunakan menurut saya RandomForestClassifier akan menghasilkan score yang terbaik karena kelebihan-kelebihan yang ada pada RandomForestClassifier sangat bagus digunakan pada data ini algoritma ini juga dapat mengurangi masalah overfitting yang dapat meningkatkan akurasi serta pendekatan yang ada pada model ini yaitu pendekatanya berbasis aturan.



## Evaluasi 

dari model yang sudah dijalankan didapatkan hasil seperti berikut 

|no|Model	|Score|mse|rmse|
|:---:|:---------------------------------:|:---------------:|:-----------------:|:-----------------:|
|0 |LogisticRegression |0.8629178492589089|0.13708215074109115|0.37024606782664304|
|1 |DecisionTreeClassifier |0.8581480605487228|0.1418519394512772|0.37663236644143744|
|2 |RandomForestClassifier |0.8646128981393882|0.13538710186061179|0.3679498632430943|
|3 |GaussianNB |0.818137023021129|0.18186297697887102|0.42645395645822193|

lalu dalam menganalisa evaluasi saya menggunakan ConfusionMatrixDisplay untuk melihat hasil dari prediksi dari model yang sudah dibuat 

ConfusionMatrixDisplay sendiri yaitu representasi matriks NxN dari label yang diprediksi dan aktual, di mana N adalah jumlah kelas yang harus diklasifikasi oleh model klasifikasi. Dalam matriks ini, kita dapat melihat jumlah nilai prediksi dan nilai aktual terhadap satu sama lain secara grafis. Ini memberi kita cara yang sangat jelas untuk menganalisis seberapa baik model kita mengklasifikasikan label individu dan kita juga dapat melihat label apa yang tidak diklasifikasikan oleh model kita dengan benar. Perhatikan gambar di bawah ini dari matriks konfusi dari klasifikasi biner

pada matrik ini dibagi menjadi 4 
- truelabel(1),predictlabel(1) = Jika nilai aktualnya positif dan nilai prediksinya juga positif, maka disebut True Positive
- truelabel(1),predictlabel(0) = Jika nilai sebenarnya positif dan nilai prediksi negatif, maka disebut Negatif Palsu
- truelabel(0),predictlabel(1) = Jika nilai aktual negatif dan nilai prediksi positif, maka disebut Positif Palsu
- truelabel(0),predictlabel(0) = Jika nilai aktual negatif dan nilai prediksi juga negatif, maka disebut True Negative


hasil ConfusionMatrixDisplay pada setiap model 

- LogisticRegression

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/log_cm.png?raw=True)

- DecisionTreeClassifier

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/dect_cm.png?raw=True)

- RandomForestClassifier

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/rf_cm.png?raw=True)

- GaussianNB

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/gd_cm.png?raw=True)


konklusi yang didapat dari percobaan proyek kali ini yaitu kita mendapatkan bahwa RandomForestClassifier memiliki score terbaik pada dataset yang kita miliki yaitu score yang didapat 0.8646128981393882 
