# %% Skrip eksekusi batch untuk pretext dan klasifikasi CARLA pada dataset SMD
import numpy as np
import pandas as pd
import os
import subprocess

# %%
# Contoh membaca info file anomali (tidak digunakan saat ini)
# with open('/home/zahraz/hz18_scratch/zahraz/datasets/MSL_SMAP/labeled_anomalies.csv', 'r') as file:
#     csv_reader = pd.read_csv(file, delimiter=',')
# data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']

# Ambil daftar file training dari dataset SMD (ubah path sesuai lingkungan Anda)
all_files = os.listdir(os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'SMD/train'))
# Filter hanya file yang diawali dengan 'machine-'
file_list = [file for file in all_files if file.startswith('machine-')]
# Urutkan nama file agar pemrosesan konsisten
file_list = sorted(file_list)
print(file_list)

# Alternatif: gunakan dataset lain (contoh UCR)
# file_list = os.listdir(os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'UCR'))                        
# file_list = sorted(file_list)

# for filename in files: #['swat']: #files: #data_info['chan_id']:
#     if filename != 'GECCO':
#         print(filename)
        
#         # Jalankan skrip pretext (pretraining) untuk menghasilkan representasi
#         subprocess.run([
#             'python', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/carla_pretext.py',
#             '--config_env', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/env.yml',
#             '--config_exp', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/pretext/carla_pretext_smd.yml',
#             '--fname', filename
#         ])
        
#         # Jalankan skrip klasifikasi untuk evaluasi/latihan downstream
#         subprocess.run([
#             'python', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/carla_classification.py',
#             '--config_env', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/env.yml',
#             '--config_exp', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/classification/carla_classification_smd.yml',
#             '--fname', filename
#         ])
# Cari indeks file tertentu jika ingin mulai dari posisi tersebut (opsional)
index = file_list.index("machine-3-11.txt")

# Loop melalui setiap file yang lolos filter untuk diproses
for filename in file_list: #[index:]:  #['GECCO']: #['machine-2-4.txt']:
    # if 'real_' in filename:
    # Lewati file bernama 'GECCO' (jika ada) sebagai pengecualian
    if filename != 'GECCO':
        print(filename)
        # Nama model/generator anomali yang diharapkan (saat ini tidak digunakan di bawah)
        genmodel = 'gen_anom_' + filename+'.pth'

        # Jalankan tahap pretext (pretraining) dengan konfigurasi lingkungan dan eksperimen
        subprocess.run([
            'python', 'carla_pretext.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/pretext/carla_pretext_smd.yml',
            '--fname', filename
        ], check=True)  # check=True akan melempar error jika proses return code != 0
        
        # Jalankan tahap klasifikasi untuk evaluasi/latihan downstream pada file yang sama
        subprocess.run([
            'python', 'carla_classification.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/classification/carla_classification_smd.yml',
            '--fname', filename
        ], check=True)  # hentikan eksekusi jika terjadi error


