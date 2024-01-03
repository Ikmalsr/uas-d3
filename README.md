# Laporan Proyek Machine Learning
### Nama : Ikmal Saepul Rohman 
### Nim : 211351063
### Kelas : Malam B

## Domain Proyek

Proyek ini mendiagnosa perihal resiko diabetes pada wanita yang sedang mengandung 

## Business Understanding

Wanita yang sedang mengandung sejatinya beresiko terkena diabetes, didasarkan pada penelitian yang di ajukan di india pada wanita berusia diatas 21 tahun dengan 787 sample.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Mendiagnosa resiko kanker terhadap wanita yang sedang mengandung

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memudahkan para pasien wanita yang sedang hamil untuk mendiagnosa apakah memiliki resiko terkena diabetes

    ### Solution statements
    - Sebagai metode untuk mengetahui klasifikasi melalui algoritma SVC
    - Didasarkan data penelitian yang menguji 787 sample sehingga cukup akurat untuk proses klasifikasi

## Data Understanding
Data yang digunakan di dasarkan pada dataset yang di sediakan oleh kaggle dimana di dalamnya terdapat 787 studi kasus.

[Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/).


### Variabel-variabel pada Dataset tersebut adalah sebagai berikut:
Variable dan tipedata yang di gunakan meliputi :

- Pregnancies = masa kehamilan pasien (int64)  
- Glucose = tingkat kandungan gulkosa (int64)  
- BloodPressure = tekanan darah pasien (int64)  
- SkinThickness = ketebalan kulit terhadap lemak (int64)  
- Insulin = tingkat insulin (int64)  
- BMI = tingkat BMI (float64)
- DiabetesPedigreeFunction = nilai DPF (float64)
- Age = usia pasien (int64) 
- Outcome = diagnosa apakah mengalami diabetes atau tidak (int64)
## Data Preparation
Pertama tama kita persiapkan dataset yang akan di pergunakan untuk menjadi model Machine Learning, selanjutnya kita lakukan data preparation dengan memanggil library yang dibutuhkan

```bash
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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
!kaggle datasets download -d uciml/pima-indians-diabetes-database/
```
Jika sudah, kita bisa membuat folder baru untuk menyimpan dan mengekstrak dataset yang sudah kita download
```bash
!mkdir diabetes
!unzip pima-indians-diabetes-database.zip -d diabetes
!ls diabetes
```
Kemudian kita mount data nya dengan perintah
```bash
df =  pd.read_csv('diabetes/diabetes.csv')
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
selanjutnya kita bisa lakukan Visualisai data
```bash
def plot_distributions_by_outcome(data, columns):
    fig, axes = plt.subplots(2, len(columns)//2, figsize=(15, 10))
    for i, column in enumerate(columns):
        row = i // (len(columns)//2)
        col = i % (len(columns)//2)
        sns.kdeplot(data=data, x=column, hue='Outcome', fill=True, ax=axes[row, col])
        axes[row, col].set_title(f"Distribution of {column}, by Outcome")
    plt.tight_layout()
    plt.show()
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
plot_distributions_by_outcome(df, columns)
```
![output](https://github.com/Ikmalsr/uts-cancer/assets/93483784/21f6ce83-5f9b-4ac6-a961-9d004532e613)
```bash
sns.countplot(x=df['Outcome'], data=df)
plt.show()
```
![bar](https://github.com/Ikmalsr/uts-cancer/assets/93483784/65ce2837-537c-419b-968e-25ec14321495)
```bash
plt.figure(figsize=(16,8))
sns.countplot(x=df['Age'],hue_order=df.Age.value_counts().index[:10])
plt.show()
```
![chart](https://github.com/Ikmalsr/uts-cancer/assets/93483784/509cc6e7-aaa6-4a5a-a06e-1fb77164c5ea)

Selanjutnya kita lakukan modeling
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
[Klik Disini](https://ikmalmachine.streamlit.app/)
![image](https://github.com/Ikmalsr/uts-cancer/assets/93483784/88df7a76-7893-4648-a031-db68bed71cf1)


