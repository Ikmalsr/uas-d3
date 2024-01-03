# Laporan Proyek Machine Learning
### Nama : Ikmal Saepul Rohman 
### Nim : 211351063
### Kelas : Malam B

## Domain Proyek

Proyek ini klasifikasi kaca berdasarkan 10 atribut yang ditentukan

## Business Understanding

Memudahkan untuk menentukan jenis kaca apa yang tepat dalam sebuah pembangunan atau pemakaian.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Ketidaktahuan seseorang terhadap jenis kaca.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memudahkan para pembangun proyek pembangunan dalam pemilihan kaca yang cocok.

    ### Solution statements
    - Dibuatkannya aplikasi klasifikasi kaca berdasarkan parameter yang telah ditentukan. dan dihitung menggunakan algoritma decision tree.
      
## Data Understanding
Data yang digunakan di dasarkan pada dataset yang di sediakan oleh kaggle dimana di dalamnya terdapat 10 atribut.

[Glass Classification](https://www.kaggle.com/datasets/uciml/glass/data).


### Variabel-variabel pada Dataset tersebut adalah sebagai berikut:
Variable dan tipedata yang di gunakan meliputi :

- Id number: 1 to 214 (removed from CSV file)
- RI: refractive index
- Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
- Mg: Magnesium
- Al: Aluminum
- Si: Silicon
- K: Potassium
- Ca: Calcium
- Ba: Barium
- Fe: Iron
- 
## Data Preparation
Pertama tama kita persiapkan dataset yang akan di pergunakan untuk menjadi model Machine Learning, selanjutnya kita lakukan data preparation dengan memanggil library yang dibutuhkan

```bash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
```bash
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
```
```bash
import pickle
```
Selanjutnya kita koneksikan google collab kita dengan kaggle menggunakan token kaggle dengan perintah
```bash
from google.colab import files
files.upload()
```
maka kita akan mengupload file token kaggle kita. dan bisa kita lanjutkan dengan membuat direktori untuk menyimpan file token tersebut
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
selanjutkan kita download data setnya
```bash
!kaggle datasets download -d uciml/glass
```
Jika sudah, kita bisa membuat folder baru untuk menyimpan dan mengekstrak dataset yang sudah kita download
```bash
!mkdir glass
!unzip glass.zip -d glass
!ls glass
```
Kemudian kita mount data nya dengan perintah
```bash
df = pd.read_csv('/content/glass/glass.csv')
```
Jika data sudah di mount, maka kita bisa mencoba memastikan apakah data akan terpanggil atau tidak dengan perintah
```bash
df.head()
```
jika sudah benar maka data 5 teratas akan muncul<br>
Untuk mengetahui jumlah baris dan kolomnya kita bisa ketikan
```bash
df.shape
```
untuk mengecek apa data ada yang duplikat atau tidak.
```bash
df.duplicated().head(40)
```
## EDA
untuk menampilkan data berdasarkan tipe.
```bash
sns.countplot(x = df['Type'], color = 'green')
```
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/38061552-d169-447a-9ea7-fd83e83438d3)

## Modeling
Selanjutnya jika data preparation sudah selesai maka kita bisa lakukan proses modeling.
Pertama tama yang harus di siapkan adalah nilai X dan Y, diman X menjadi atrribut dan Y menjadi label
```bash
X = df.drop (columns='Outcome', axis=1)
y = df['Outcome']
```
Kita lihat untuk nilai X nya adalah semua kolom kecuali Outcome<br>
Selanjutnya bisa kita lakukan standarisasi data
```bash
scaler = StandardScaler()
```
```bash
scaler.fit(X)
```
```bash
standarized_data = scaler.transform(X)
```
```bash
X = standarized_data
y = df['Outcome']
```
```bash
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=2)
```
```bash
classifier = svm.SVC(kernel='linear')
```
```bash
classifier.fit(X_train, y_train)
```

Jikalau sudah kita bisa coba inputkan data
```bash
input_data_test = (6,148,72,35,0,33.6,0.627,50)

array = np.array(input_data_test)

reshape = array.reshape(1,-1)

std_data = scaler.transform(reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('Tidak Beresiko Diabetes')
else :
    print('Beresiko Diabetes')
```
```bash
[[ 0.63994726  0.84832379  0.14964075  0.90726993 -0.69289057  0.20401277
   0.46849198  1.4259954 ]]
[1]
Beresiko Diabetes
```
Jika outputnya sudah keluar sesuai keinginan kita bisa import file nya
```bash
import pickle
filename = 'diabetes.sav'
pickle.dump(classifier,open(filename,'wb'))
```
## Evaluation
Proses evaluasi dilakukan dengan pengecekan akurasi. Dan Proses ini sudah cukup untuk melakukan pengecekan pada algoritma klasifikasi dengan perintah dan output berupa :
```bash
x_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Tingkat akurasi data training = ', training_data_accuracy)
Tingkat akurasi data training =  0.7821229050279329
```

- Proses pengecekan akurasi bisa di ambil ketika data train dan data test nya sudah memiliki akurasi yang cukup tinggi
- Jika ingin melakukan evaluasi dengan algoritma lain, maka harus di tambahkan algoritma permodelanya


## Deployment
Aplikasi saya
[Klik Disini](https://uas-d3-ikmal.streamlit.app/)
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/5cf0c969-388a-429c-81f9-6b27e160ec81)


