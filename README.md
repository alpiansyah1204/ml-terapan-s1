# Machine-Learning-Terapan
# Laporan Proyek Machine Learning - Rizqi Alpiansyah

## Domain Proyek
Project Machine Learning Terapan : membuat model Predictive Analysis, menggunakan dataset yang berdomain kesehatan mengenai diabetes.
### Latar Belakang
Diabetes terjadi ketika glukosa darah Anda, umumnya dikenal sebagai gula darah, terlalu tinggi, Anda mengembangkan diabetes. Sumber energi utama Anda, glukosa darah, diperoleh dari makanan yang Anda makan. Glukosa dari makanan diangkut ke dalam sel Anda oleh hormon insulin, yang diproduksi oleh pankreas. Tubuh Anda kadang-kadang menghasilkan insulin yang tidak mencukupi atau tidak ada sama sekali, atau menggunakan insulin dengan buruk. Setelah itu, glukosa tetap berada dalam sirkulasi Anda dan tidak masuk ke dalam sel Anda.
Seiring waktu, memiliki terlalu banyak glukosa dalam darah Anda dapat menyebabkan masalah kesehatan. Meskipun tidak ada obat untuk diabetes, ada beberapa hal yang dapat Anda lakukan untuk mengelolanya dan tetap sehat.

Diabetes kadang-kadang disebut sebagai "diabetes ambang" atau "sentuhan gula." Ungkapan ini menyiratkan bahwa seseorang tidak benar-benar menderita diabetes atau memiliki kasus yang lebih ringan, namun diabetes selalu memiliki konsekuensi yang menghancurkan.
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

setelah data outlier dibersihkan 
![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/after%20boxplot.png?raw=True)

## Data Preparation
Sebelum datasetnya di latih atau training, dari model sebelumnya perlu melakukan encoding lalu pemisahan data antara data latih dan test setelah itu melakukan scaling untuk data categorical agar data dapat dilatih.

#### Train-Test Split
Proses splitting data atau pembagian dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum melakukan pemodelan supervised. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan train test split karena untuk efisiensi dan tidak melakukan data leakage ketika melakukan scaling. 

#### Standardisasi 
Data numerik yang terdapat di dataset perlu dilakukannya proses Standardisasi sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritma machine learning dan membuatnya konvergen lebih cepat selain itu menghindari overfitting dan juga data imbalance.

## Modeling
