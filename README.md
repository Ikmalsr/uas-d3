# Laporan Proyek Machine Learning
### Nama : Ikmal Saepul Rohman 
### Nim : 211351063
### Kelas : Malam B

## Domain Proyek

Proyek ini klasifikasi kaca berdasarkan 10 atribut yang ditentukan untuk menentukan peruntukan kaca dalam penggunaan nya entah itu dalam proyek pembangunan atau produk.

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
- semua atribut diatas bertipe data float
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

Selanjutnya mengecek banyaknya sample
```bash
numerical = [
    'RI',
    'Na',
    'Mg',
    'Al',
    'Si',
    'K',
    'Ca',
    'Ba',
    'Fe'
]

categorical = [
    'Type'
]
```
```bash
for i in df[numerical].columns:
    plt.hist(df[numerical][i])
    plt.xticks()
    plt.xlabel(i)
    plt.ylabel('number of samples')
    plt.show()
```
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/e90889ee-9dd3-4ec9-81fc-4f41d9ca504a)
Kita juga bisa memvisualkan RI terhadap kadungan silikon dengan scater plot
```bash
sns.lmplot(x='RI', y='Si', data=df)
plt.show()
```
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/faf3f1c4-bb6a-487c-93d0-dfb346aee912)

Kita juga bisa melihat boxplot setiap atributnya
```bash
plt.figure(figsize=(10,10))
sns.boxplot(data=df, orient="h");
```
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/311e23bb-b8cd-4d86-8cfb-44af3f7feeb3)


## Modeling
Selanjutnya jika data preparation sudah selesai maka kita bisa lakukan proses modeling.
Pertama tama yang harus di siapkan adalah nilai X dan Y, diman X menjadi atrribut dan Y menjadi label
```bash
X = df.drop('Type', axis = 1)
y = df['Type']
```
Kita lihat untuk nilai X nya adalah semua kolom kecuali Outcome<br>
Kita masukan kondisi untuk data train dan dasta test nya
```bash
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
```
lakukan standarisasi data
```bash
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
masukan algoritma decison tree
```bash
dtc = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0,
    random_state=42, splitter='best'
)

model = dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Data Train Accuracy = {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Data Test Accuracy = {dtc_acc} \n")
```
```bash
Data Train Accuracy = 0.8187919463087249
Data Test Accuracy = 0.703125
```
Berdasarkan data diatas, akurasi untuk data train adalah 81% dan data test 70%
Jika sudah kita lakukan pengetesan dengan cara
```bash
input_data = (1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.0,0.0)

input_data_as_numpy_array = np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 1):
    print('Kaca Bangunan (Bukan Float Proses)')
elif (prediction[0] == 2):
        print('Kaca Bangunan (Float Proses)')
elif (prediction[0] == 3):
        print('Kaca Mobil (Non Float Proses)')
elif (prediction[0] == 4):
        print('Kaca Mobil (Float Proses)')
elif (prediction[0] == 5):
        print('Wadah Kaca')
elif (prediction[0] == 6):
        print('Peralatan Dapur')
else :
    print('Kaca Lampu')
```
```bash
[[ 0.82337811  0.32241914  1.26680335 -0.71500479 -1.10889536 -0.59942436
  -0.1832657  -0.37897356 -0.5993232 ]]
[1]
Kaca Bangunan (Bukan Float Proses)
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
```

## Evaluation
Proses evaluasi dilakukan dengan pengecekan akurasi. Dan Proses ini sudah cukup untuk melakukan pengecekan pada algoritma klasifikasi confusion matrix dengan perintah dan output berupa :
```bash
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=dtc.classes_, yticklabels=dtc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.show()

print("Classification Report:\n", class_report)
```
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/c49c231d-c85f-4ea8-b314-196a5777fde5)
```bash
Classification Report:
               precision    recall  f1-score   support

           1       0.62      0.75      0.68        20
           2       0.74      0.80      0.77        25
           3       1.00      0.12      0.22         8
           5       0.40      1.00      0.57         2
           6       1.00      1.00      1.00         2
           7       1.00      0.71      0.83         7

    accuracy                           0.70        64
   macro avg       0.79      0.73      0.68        64
weighted avg       0.76      0.70      0.68        64
```

- Proses pengecekan akurasi bisa di ambil ketika data train dan data test nya sudah memiliki akurasi yang cukup tinggi
- Jika ingin melakukan evaluasi dengan algoritma lain, maka harus di tambahkan algoritma permodelanya


## Deployment
Aplikasi saya
[Klik Disini](https://uas-d3-ikmal.streamlit.app/)
![image](https://github.com/Ikmalsr/uas-d3/assets/93483784/5cf0c969-388a-429c-81f9-6b27e160ec81)


