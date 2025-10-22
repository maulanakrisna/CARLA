
# Import library standar Python
import argparse  # untuk parsing argumen baris perintah
import os  # untuk operasi sistem file
import torch  # framework deep learning PyTorch
import pandas  # untuk manipulasi data tabular
import numpy as np  # untuk operasi numerik

# Import modul utilitas custom CARLA
from utils.mypath import MyPath  # utility untuk path dataset/model
from termcolor import colored  # untuk pewarnaan output log di terminal
from utils.config import create_config  # untuk memuat konfigurasi environment/eksperimen
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_val_transformations1, \
                                get_train_dataset, get_train_dataloader, get_aug_train_dataset,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate  # utility evaluasi klasifikasi
from utils.train_utils import self_sup_classification_train  # loop training tahap klasifikasi self-supervised
from statsmodels.tsa.stattools import adfuller  # untuk uji stasioneritas Augmented Dickey-Fuller
import random  # untuk operasi random

def set_seed(seed):
    """
    Fungsi untuk mengatur seed random untuk memastikan hasil yang dapat direproduksi
    Args:
        seed (int): nilai seed untuk random number generator
    """
    # Set seed untuk semua library yang menggunakan random
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU random
    torch.cuda.manual_seed(seed)  # PyTorch GPU random
    torch.backends.cudnn.deterministic = True  # Pastikan operasi CUDNN deterministik
    torch.backends.cudnn.benchmark = False  # Nonaktifkan benchmark untuk konsistensi

# Set seed dengan nilai tetap untuk reprodusibilitas
set_seed(4)

# Deteksi dan konfigurasi device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gunakan GPU jika tersedia

# Parser argumen baris perintah untuk tahap klasifikasi
FLAGS = argparse.ArgumentParser(description='classification Loss')
FLAGS.add_argument('--config_env', help='Lokasi file konfigurasi environment')
FLAGS.add_argument('--config_exp', help='Lokasi file konfigurasi eksperimen')
FLAGS.add_argument('--fname', help='Nama file dataset yang akan diproses')

def main():
    """
    Fungsi utama untuk menjalankan tahap klasifikasi CARLA
    Tahap ini menggunakan model yang sudah dilatih pada tahap pretext untuk melakukan
    klasifikasi self-supervised dengan loss berbasis tetangga terdekat/terjauh
    """
    global best_f1  # variabel global untuk menyimpan F1 score terbaik
    
    # Parse argumen dari baris perintah
    args = FLAGS.parse_args()
    
    # Buat konfigurasi gabungan dari environment + eksperimen + nama file
    p = create_config(args.config_env, args.config_exp, args.fname)
    print(colored('CARLA Self-supervised Classification stage --> ', 'yellow'))

    # Konfigurasi CUDNN untuk optimasi GPU (saat ini dinonaktifkan)
    # torch.backends.cudnn.benchmark = True

    # === PREPARASI DATA ===
    print(colored('\n- Get dataset and dataloaders for ' + p['train_db_name'] + ' dataset - timeseries ' + p['fname'], 'green'))
    
    # Setup transformasi/augmentasi dan injeksi sub-anomali
    train_transformations = get_train_transformations(p)  # transformasi untuk training
    sanomaly = inject_sub_anomaly(p)  # injeksi sub-anomali sesuai konfigurasi
    val_transformations = get_val_transformations1(p)  # transformasi untuk validasi
    
    # Dataset training dengan augmentasi untuk membentuk pasangan tetangga
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)

    # === PENANGANAN DATASET BERDASARKAN JENIS ===
    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        if p['fname'] == 'All':
            # Mode gabungan: proses semua channel dari satelit MSL/SMAP
            # Baca informasi channel dari file CSV yang berisi label anomali
            with open(os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv'), 'r') as file:
                csv_reader = pandas.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == p['train_db_name']]
            ii = 0
            for file_name in data_info['chan_id']:
                p['fname'] = file_name
                if ii == 0 :
                    # Inisialisasi dataset dasar dan validasi untuk channel pertama
                    base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                     to_neighbors_dataset=True)
                    val_dataset = get_val_dataset(p, val_transformations, sanomaly, True, base_dataset.mean,
                                                  base_dataset.std)
                else:
                    # Tambahkan channel berikutnya ke dataset gabungan
                    new_base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                     to_neighbors_dataset=True)
                    new_val_dataset = get_val_dataset(p, val_transformations, sanomaly, True, new_base_dataset.mean,
                                                  new_base_dataset.std)
                    val_dataset.concat_ds(new_val_dataset)  # gabungkan dataset validasi
                    base_dataset.concat_ds(new_base_dataset)  # gabungkan dataset training
                ii+=1
        else:
            # Mode single channel: proses satu channel saja
            # Ambil statistik dari training set lalu bangun validation set
            info_ds = get_train_dataset(p, train_transformations, sanomaly, to_neighbors_dataset=False)
            val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, info_ds.mean, info_ds.std)

    elif p['train_db_name'] == 'yahoo':
        # === PENANGANAN DATASET YAHOO ===
        # Dataset Yahoo berbentuk file CSV per time series dengan kolom 'value' dan 'is_anomaly'
        filename = os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'Yahoo/', p['fname'])
        dataset = []
        
        # Baca data dari file CSV
        df = pandas.read_csv(filename)
        dataset.append({
            'value': df['value'].tolist(),  # nilai time series
            'label': df['is_anomaly'].tolist()  # label anomali (0/1)
        })

        # Ekstrak data dan label
        ts = dataset[0]
        data = np.array(ts['value'])
        labels = np.array(ts['label'])
        l = len(data) // 2  # bagi data menjadi train/test (50/50)

        n = 0
        # === PROSES DIFERENSIASI UNTUK STASIONERITAS ===
        # Lakukan diferensiasi berulang hingga segmen awal menjadi stasioner (uji ADF)
        while adfuller(data[:l], 1)[1] > 0.05 or adfuller(data[:l])[1] > 0.05:
            data = np.diff(data)  # diferensiasi pertama
            labels = labels[1:]  # sesuaikan label
            n += 1
        l -= n  # sesuaikan panjang setelah diferensiasi

        # Pisahkan data train dan test
        all_train_data = data[:l]
        all_test_data = data[l:]
        all_train_labels = labels[:l]
        all_test_labels= labels[l:]

        # Normalisasi test data berdasarkan statistik train data
        mean, std = all_train_data.mean(), all_train_data.std()
        all_test_data = (all_test_data - mean) / std

        # Assign ke variabel yang digunakan untuk dataset
        TRAIN_TS = all_train_data
        train_label = all_train_labels
        TEST_TS = all_test_data
        test_label = all_test_labels

        # Buat dataset dengan data yang sudah diproses
        base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                          to_augmented_dataset=True, data=TRAIN_TS, label=train_label)
        val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, base_dataset.mean, base_dataset.std,
                                        TEST_TS, test_label)

    elif p['train_db_name'] == 'smd' or p['train_db_name'] == 'kpi' or p['train_db_name'] == 'swat' \
        or p['train_db_name'] == 'swan' or p['train_db_name'] == 'gecco' or p['train_db_name'] == 'wadi' or p['train_db_name'] == 'ucr':
        # === DATASET STANDAR ===
        # Dataset yang sudah didukung langsung oleh loader utility
        base_dataset = get_train_dataset(p, train_transformations, sanomaly, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, base_dataset.mean,
                                      base_dataset.std)

    # Buat DataLoader untuk validasi
    val_dataloader = get_val_dataloader(p, val_dataset)

    print(colored('-- Train samples size: %d - Test samples size: %d' %(len(train_dataset), len(val_dataset)), 'green'))

    # === INISIALISASI MODEL ===
    # Load model klasifikasi dengan bobot dari tahap pretext
    model = get_model(p, p['pretext_model'])  # inisialisasi model klasifikasi dari bobot pretext
    model = torch.nn.DataParallel(model)  # paralelisasi untuk multi-GPU
    model = model.to(device)  # pindahkan ke device (GPU/CPU)

    # === SETUP OPTIMIZER ===
    # Konfigurasi optimizer dengan opsi update hanya cluster head
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])  # opsi update hanya head cluster

    # Warning jika hanya cluster head yang diupdate
    if p['update_cluster_head_only']:
        print(colored('WARNING: classification will only update the cluster head', 'red'))

    # === SETUP LOSS FUNCTION ===
    # Inisialisasi loss function untuk klasifikasi self-supervised
    criterion = get_criterion(p)  # loss untuk klasifikasi self-supervised
    criterion.to(device)  # pindahkan ke device

    print(colored('\n- Model initialisation', 'green'))
    
    # === CHECKPOINT MANAGEMENT ===
    # Cek apakah ada checkpoint dari training sebelumnya
    if os.path.exists(p['classification_checkpoint']):
        # Lanjutkan training dari checkpoint sebelumnya
        print(colored('-- Model initialised from last checkpoint: {}'.format(p['classification_checkpoint']), 'green'))
        checkpoint = torch.load(p['classification_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])  # load bobot model
        optimizer.load_state_dict(checkpoint['optimizer'])  # load state optimizer
        start_epoch = checkpoint['epoch']  # epoch terakhir yang diselesaikan
        best_loss = checkpoint['best_loss']  # loss terbaik sebelumnya
        best_loss_head = checkpoint['best_loss_head']  # head dengan loss terbaik
        normal_label = checkpoint['normal_label']  # label normal yang teridentifikasi

    else:
        # Mulai training dari awal (tidak ada checkpoint)
        print(colored('-- No checkpoint file at {} -- new model initialised'.format(p['classification_checkpoint']), 'green'))
        start_epoch = 0
        best_loss = 1e4  # nilai awal loss tinggi
        best_loss_head = None
        normal_label = 0  # label normal default


    # === INISIALISASI VARIABEL TRAINING ===
    best_f1 = -1 * np.inf  # F1 score terbaik (mulai dari nilai sangat rendah)
    print(colored('\n- Training:', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('-- Epoch %d/%d' %(epoch+1, p['epochs']), 'blue'))

        lr = adjust_learning_rate(p, optimizer, epoch)
        self_sup_classification_train(train_dataloader, model, criterion, optimizer, epoch,
                                      p['update_cluster_head_only'])

        if (epoch == p['epochs']-1):
            # Di akhir, minta prediksi dengan logit + head
            tst_dl = get_val_dataloader(p, train_dataset)
            predictions, _ = get_predictions(p, tst_dl, model, True, True)
        else:
            # Selama epoch, prediksi tanpa logit/head penuh untuk efisiensi
            tst_dl = get_val_dataloader(p, train_dataset)
            predictions = get_predictions(p, tst_dl, model, False, False)

        label_counts = torch.bincount(predictions[0]['predictions'])
        majority_label = label_counts.argmax()

        classification_stats = classification_evaluate(predictions)  # hitung metrik dan loss per head
        lowest_loss_head = classification_stats['lowest_loss_head']
        lowest_loss = classification_stats['lowest_loss']
        predictions = get_predictions(p, val_dataloader, model, False, False)  # prediksi pada val

        rep_f1 = pr_evaluate(predictions, compute_confusion_matrix=False, majority_label=majority_label)  # evaluasi F1 berbasis PR

        if rep_f1 > best_f1:
            best_f1 = rep_f1
            nomral_label = majority_label
            # print('New Checkpoint ...')
            # Simpan bobot terbaik dan checkpoint kemajuan training
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head, 'normal_label': normal_label}, p['classification_model'])
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': normal_label},
                       p['classification_checkpoint'])


    # Muat kembali model terbaik untuk evaluasi akhir pada set validasi
    model_checkpoint = torch.load(p['classification_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': p['epochs'], 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': normal_label},
               p['classification_checkpoint'])
    normal_label = model_checkpoint['normal_label']
    tst_dl = get_val_dataloader(p, val_dataset)
    predictions, _ = get_predictions(p, tst_dl, model, True)

if __name__ == "__main__":
    # Entry point skrip
    main()
