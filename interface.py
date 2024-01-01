import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model = pickle.load(open('glass_classification_d3.sav', 'rb'))

st.title('Klasifikasi Jenis Penggunaan Kaca dengan Algoritma Decision Tree')

RI = st.text_input('Nomor Refractive Index (RI)')
Na = st.text_input('Kandungan Sodium (Na)')
Mg = st.text_input('Kandungan Magnesium (Mg)')
Al = st.text_input('Kandungan Aluminium (Al)')
Si = st.text_input('Kandungan Silicone (Si)')
K = st.text_input('Kandungan Potasium (K)')
Ca = st.text_input('Kandungan Calcium (Ca)')
Ba = st.text_input('Kandungan Barium (Ba)')
Fe = st.text_input('Kandungan Besi Murni (Fe)')

classification = ''

if st.button('Klasifikasi'):
    klasifikasi = model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
    
    if (klasifikasi[0] == 1):
        classification = 'Kaca Bangunan (Bukan Float Proses)'
    elif (klasifikasi[0] == 2):
        classification = 'Kaca Bangunan (Float Proses)'
    elif (klasifikasi[0] == 3):
        classification = 'Kaca Mobil (Non Float Proses)'
    elif (klasifikasi[0] == 4):
        classification = 'Kaca Mobil (Float Proses)'
    elif (klasifikasi[0] == 5):
        classification = 'Wadah Kaca'
    elif (klasifikasi[0] == 6):
        classification = 'Peralatan Dapur'
    else :
        classification = 'Kaca Lampu'

    st.success(classification)

if st.button('Visualisasi'):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'], ax=ax, class_names=['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps'])
    st.pyplot(fig)