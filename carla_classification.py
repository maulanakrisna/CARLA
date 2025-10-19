
import argparse
import os
import torch
import pandas
import numpy as np
from utils.mypath import MyPath  # util path dataset/model
from termcolor import colored  # pewarnaan output log di terminal
from utils.config import create_config  # muat konfigurasi env/eksperimen
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_val_transformations1, \
                                get_train_dataset, get_train_dataloader, get_aug_train_dataset,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate  # util evaluasi klasifikasi
from utils.train_utils import self_sup_classification_train  # loop train tahap klasifikasi self-supervised
from statsmodels.tsa.stattools import adfuller
import random

def set_seed(seed):
    # Set seed untuk reprodusibilitas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gunakan GPU jika tersedia

# Parser argumen baris perintah untuk tahap klasifikasi
FLAGS = argparse.ArgumentParser(description='classification Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--fname', help='Config the file name of Dataset')

def main():
    global best_f1
    args = FLAGS.parse_args()
    # Buat konfigurasi gabungan dari env + eksperimen + nama file
    p = create_config(args.config_env, args.config_exp, args.fname)
    print(colored('CARLA Self-supervised Classification stage --> ', 'yellow'))

    # CUDNN
   # torch.backends.cudnn.benchmark = True

    # Data
    print(colored('\n- Get dataset and dataloaders for ' + p['train_db_name'] + ' dataset - timeseries ' + p['fname'], 'green'))
    # Transformasi/augmentasi dan injeksi sub-anomali
    train_transformations = get_train_transformations(p)
    sanomaly = inject_sub_anomaly(p)
    val_transformations = get_val_transformations1(p)
    # Dataset training dengan augmentasi untuk membentuk pasangan tetangga
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)

    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        if p['fname'] == 'All':
            # Gabungkan seluruh channel untuk MSL/SMAP
            with open(os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv'), 'r') as file:
                csv_reader = pandas.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == p['train_db_name']]
            ii = 0
            for file_name in data_info['chan_id']:
                p['fname'] = file_name
                if ii == 0 :
                    # Inisialisasi dataset dasar dan validasi pertama kali
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
                    val_dataset.concat_ds(new_val_dataset)
                    base_dataset.concat_ds(new_base_dataset)
                ii+=1
        else:
            #base_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
            # Mode satu channel saja: ambil statistik dari train lalu bangun val
            info_ds = get_train_dataset(p, train_transformations, sanomaly, to_neighbors_dataset=False)
            val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, info_ds.mean, info_ds.std)

    elif p['train_db_name'] == 'yahoo':
        # Khusus dataset Yahoo (CSV per seri waktu) + normalisasi test berdasarkan mean/std train
        filename = os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'Yahoo/', p['fname'])
        dataset = []
        # print(filename)
        df = pandas.read_csv(filename)
        dataset.append({
            'value': df['value'].tolist(),
            'label': df['is_anomaly'].tolist()
        })

        ts = dataset[0]
        data = np.array(ts['value'])
        labels = np.array(ts['label'])
        l = len(data) // 2

        n = 0
        # Diferensiasi hingga segmen awal stasioner (uji ADF)
        while adfuller(data[:l], 1)[1] > 0.05 or adfuller(data[:l])[1] > 0.05:
            data = np.diff(data)
            labels = labels[1:]
            n += 1
        l -= n

        all_train_data = data[:l]
        all_test_data = data[l:]
        all_train_labels = labels[:l]
        all_test_labels= labels[l:]

        mean, std = all_train_data.mean(), all_train_data.std()
        all_test_data = (all_test_data - mean) / std

        TRAIN_TS = all_train_data
        train_label = all_train_labels
        TEST_TS = all_test_data
        test_label = all_test_labels

        base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                          to_augmented_dataset=True, data=TRAIN_TS, label=train_label)
        val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, base_dataset.mean, base_dataset.std,
                                        TEST_TS, test_label)

    elif p['train_db_name'] == 'smd' or p['train_db_name'] == 'kpi' or p['train_db_name'] == 'swat' \
        or p['train_db_name'] == 'swan' or p['train_db_name'] == 'gecco' or p['train_db_name'] == 'wadi' or p['train_db_name'] == 'ucr':
        # Dataset yang didukung langsung loader util
        base_dataset = get_train_dataset(p, train_transformations, sanomaly, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, base_dataset.mean,
                                      base_dataset.std)

    val_dataloader = get_val_dataloader(p, val_dataset)

    print(colored('-- Train samples size: %d - Test samples size: %d' %(len(train_dataset), len(val_dataset)), 'green'))

    # Model
    model = get_model(p, p['pretext_model'])  # inisialisasi model klasifikasi dari bobot pretext
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])  # opsi update hanya head cluster

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: classification will only update the cluster head', 'red'))

    # Loss function
    criterion = get_criterion(p)  # loss untuk klasifikasi self-supervised
    criterion.to(device)

    print(colored('\n- Model initialisation', 'green'))
    # Checkpoint
    if os.path.exists(p['classification_checkpoint']):
        # Lanjut dari checkpoint sebelumnya
        print(colored('-- Model initialised from last checkpoint: {}'.format(p['classification_checkpoint']), 'green'))
        checkpoint = torch.load(p['classification_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        normal_label = checkpoint['normal_label']

    else:
        # Mulai model baru
        print(colored('-- No checkpoint file at {} -- new model initialised'.format(p['classification_checkpoint']), 'green'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        normal_label = 0


    best_f1 = -1 * np.inf
    # best_loss = np.inf
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
