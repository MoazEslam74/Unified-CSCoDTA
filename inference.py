import os
import argparse
import torch
import json
import warnings
import torch.nn as nn
from collections import OrderedDict
from itertools import chain
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
# === [NEW] أضفنا استدعاء مكتبات الـ Confusion Matrix ===
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

# Important graph-related imports
from torch_geometric.data import Data, Batch
from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph, smile_to_graph
from util import GraphDataset, collate, model_evaluate

from models import Unified_CSCoDTA, PredictModule
from utils.config_tools import get_defaults_yaml_args, update_args

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# ==========================================
class DDIDataset(torch.utils.data.Dataset):
    def __init__(self, ddi_csv_path):
        self.ddi_df = pd.read_csv(ddi_csv_path)
        
        print("Extracting unique drugs for caching...")
        unique_smiles = set(self.ddi_df['SMILES1']).union(set(self.ddi_df['SMILES2']))
        print(f"Found {len(unique_smiles)} unique drugs. Pre-processing graphs...")
        
        self.smiles_cache = {}
        for smi in tqdm(unique_smiles, desc="Caching SMILES Graphs"):
            try:
                _, f, e = smile_to_graph(smi)
                data = Data(x=torch.tensor(f, dtype=torch.float32), 
                             edge_index=torch.tensor(e, dtype=torch.long).t().contiguous())
            except:
                data = Data(x=torch.zeros((1, 78), dtype=torch.float32), 
                             edge_index=torch.empty((2, 0), dtype=torch.long))
            self.smiles_cache[smi] = data
            
        print("Caching complete! Ready for blazing fast training 🚀")

    def __getitem__(self, idx):
        row = self.ddi_df.iloc[idx]
        data1 = self.smiles_cache[row['SMILES1']]
        data2 = self.smiles_cache[row['SMILES2']]
        label = torch.tensor(row['Label'], dtype=torch.float32)
        return data1, data2, label

    def __len__(self):
        return len(self.ddi_df)
    
def ddi_collate(data_list):
    batch1 = Batch.from_data_list([d[0] for d in data_list])
    batch2 = Batch.from_data_list([d[1] for d in data_list])
    labels = torch.stack([d[2] for d in data_list])
    return batch1, batch2, labels
# ==========================================

def train(model, predictor, device, train_loader, ddi_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch, batch_size, drug_pos, target_pos, ddi_pos):
    model.train()
    predictor.train()
    LOG_INTERVAL = 100
    
    loss_mse = nn.MSELoss() 
    loss_bce = nn.BCEWithLogitsLoss() 
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=1e-4)
    
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))

    ddi_iter = iter(ddi_loader)

    total_loss_sum = 0.0
    num_batches = 0
    
    train_dti_preds, train_dti_labels = [], []
    train_ddi_preds, train_ddi_labels = [], []

    for batch_idx, data in enumerate(train_loader):
        try:
            ddi_data = next(ddi_iter)
        except StopIteration:
            ddi_iter = iter(ddi_loader)
            ddi_data = next(ddi_iter)

        optimizer.zero_grad()

        # DTI Path
        try:
            ssl_dti_loss = model.get_dti_contrastive_loss(drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)
        except Exception:
            ssl_dti_loss = torch.tensor(0.0, device=device)

        out_drug = model.shared_drug_graph_conv(drug_graph_batchs)
        drug_embedding = out_drug[-1] if isinstance(out_drug, list) else out_drug
        out_target = model.target_graph_conv(target_graph_batchs)
        target_embedding = out_target[-1] if isinstance(out_target, list) else out_target

        dti_output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        dti_lbls = data.y.view(-1, 1).float().to(device)
        loss_dti = loss_mse(dti_output, dti_lbls) + ssl_dti_loss

        # DDI Path
        batch1, batch2, ddi_labels = ddi_data
        batch1, batch2, ddi_labels = batch1.to(device), batch2.to(device), ddi_labels.to(device)

        try:
            ssl_ddi_loss = model.get_ddi_contrastive_loss(drug_graph_batchs, drug_graph_batchs, ddi_pos)
        except Exception:
            ssl_ddi_loss = torch.tensor(0.0, device=device)

        out1 = model.shared_drug_graph_conv([batch1])
        emb1 = out1[-1] if isinstance(out1, list) else out1
        out2 = model.shared_drug_graph_conv([batch2])
        emb2 = out2[-1] if isinstance(out2, list) else out2

        ddi_output = model.predict(task='ddi', input1=emb1, input2=emb2)
        ddi_lbls = ddi_labels.view(-1, 1).float()
        loss_ddi = loss_bce(ddi_output, ddi_lbls) + ssl_ddi_loss

        total_loss = (1.0 * loss_dti) + (1.0 * loss_ddi)
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        num_batches += 1
        
        train_dti_preds.extend(dti_output.detach().cpu().numpy().flatten())
        train_dti_labels.extend(dti_lbls.detach().cpu().numpy().flatten())
        train_ddi_preds.extend(torch.sigmoid(ddi_output).detach().cpu().numpy().flatten())
        train_ddi_labels.extend(ddi_lbls.detach().cpu().numpy().flatten())

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), total_loss.item()))
            
    train_dti_mse = mean_squared_error(train_dti_labels, train_dti_preds)
    
    train_ddi_preds_bin = (np.array(train_ddi_preds) > 0.5).astype(int)
    try:
        train_ddi_auc = roc_auc_score(train_ddi_labels, train_ddi_preds)
        train_ddi_acc = accuracy_score(train_ddi_labels, train_ddi_preds_bin)
    except:
        train_ddi_auc, train_ddi_acc = 0, 0

    return total_loss_sum / num_batches, train_dti_mse, train_ddi_auc, train_ddi_acc

def test_dti(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    
    with torch.no_grad():
        for data in loader:
            out_drug = model.shared_drug_graph_conv(drug_graph_batchs)
            drug_embedding = out_drug[-1] if isinstance(out_drug, list) else out_drug
            
            out_target = model.target_graph_conv(target_graph_batchs)
            target_embedding = out_target[-1] if isinstance(out_target, list) else out_target
            
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# === [NEW] أضفنا return_preds عشان نرجع التوقعات النهائية ونرسم الماتريكس ===
def test_ddi(model, device, loader, return_preds=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        for batch1, batch2, ddi_labels in loader:
            batch1, batch2 = batch1.to(device), batch2.to(device)

            out1 = model.shared_drug_graph_conv([batch1])
            emb1 = out1[-1] if isinstance(out1, list) else out1

            out2 = model.shared_drug_graph_conv([batch2])
            emb2 = out2[-1] if isinstance(out2, list) else out2

            ddi_output = model.predict(task='ddi', input1=emb1, input2=emb2)
            
            total_preds = torch.cat((total_preds, ddi_output.cpu()), 0)
            total_labels = torch.cat((total_labels, ddi_labels.cpu()), 0)
            
    probs = torch.sigmoid(total_preds).numpy().flatten()
    labels = total_labels.numpy().flatten()
    preds = (probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
    except:
        auc, acc = 0, 0

    if return_preds:
        return auc, acc, labels, preds
    return auc, acc
# ========================================================

def train_predict(new_args):
    print("Data preparation in progress...")
    num_drug, num_target = 2111, 228
    affinity_mat = load_data(args.dataset)
    train_data, test_data, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    ddi_pos = drug_pos 
    
    #Cold-start DDI dataset preparation
    print("Loading Cold Start Datasets...")
    ddi_train_dataset = DDIDataset('/kaggle/working/Unified-CSCoDTA/data/ddi_train_cold.csv')
    ddi_test_dataset = DDIDataset('/kaggle/working/Unified-CSCoDTA/data/ddi_test_cold.csv')
    
    ddi_train_loader = torch.utils.data.DataLoader(ddi_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ddi_collate)
    ddi_test_loader = torch.utils.data.DataLoader(ddi_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ddi_collate)

    drug_graphs_dict = get_drug_molecule_graph(json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=num_drug)
    
    target_graphs_dict = get_target_molecule_graph(json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=num_target)

    print("Model preparation...")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    
    model = Unified_CSCoDTA(tau=args.tau, lam=args.lam, ns_dims=[num_drug + num_target + 2, 512, 256],
                    d_ms_dims=[78, 78, 78 * 2, 256], t_ms_dims=[54, 54, 54 * 2, 256],
                    embedding_dim=128, dropout_rate=args.edge_dropout_rate, args=new_args)
                    
    predictor = PredictModule()
    
    drug_pos, target_pos, ddi_pos = drug_pos.to(device), target_pos.to(device), ddi_pos.to(device)
    model.to(device)
    predictor.to(device)

    print(f"Training on {len(train_loader.dataset)} DTI samples and {len(ddi_train_loader.dataset)} DDI samples...")
    print("Start training...")
    
    history_df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Train_DTI_MSE', 'Val_DTI_MSE', 'Val_DTI_CI', 'Train_DDI_AUC', 'Val_DDI_AUC', 'Train_DDI_ACC', 'Val_DDI_ACC'])
    
    # === متغيرات للاحتفاظ بنتائج آخر دورة ===
    final_labels = None
    final_preds = None

    for epoch in range(args.epochs):
        # training and computing training accuracy
        avg_total, train_dti_mse, train_ddi_auc, train_ddi_acc = train(model, predictor, device, train_loader, ddi_train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1, args.batch_size, drug_pos, target_pos, ddi_pos)
              
        # testing and computing validation accuracy
        G, P = test_dti(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader)
        val_dti_metrics = model_evaluate(G, P) # [MSE, CI, ...]
        val_dti_mse = val_dti_metrics[0]
        val_dti_ci = val_dti_metrics[1]
        
        # === [NEW] جلب التوقعات في الدورة الأخيرة فقط لرسم الماتريكس ===
        if epoch == args.epochs - 1:
            val_ddi_auc, val_ddi_acc, final_labels, final_preds = test_ddi(model, device, ddi_test_loader, return_preds=True)
        else:
            val_ddi_auc, val_ddi_acc = test_ddi(model, device, ddi_test_loader)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"DTI Task -> Train MSE: {train_dti_mse:.4f} | Val MSE: {val_dti_mse:.4f} | Val CI: {val_dti_ci:.4f}") 
        print(f"DDI AUC -> Train: {train_ddi_auc:.4f} | Val: {val_ddi_auc:.4f}")
        print(f"DDI ACC -> Train: {train_ddi_acc:.4f} | Val: {val_ddi_acc:.4f}\n")
        
        # save results
        history_df.loc[epoch] = [epoch+1, avg_total, train_dti_mse, val_dti_mse, val_dti_ci, train_ddi_auc, val_ddi_auc, train_ddi_acc, val_ddi_acc]
        history_df.to_csv('learning_curves_log.csv', index=False) 

    # === [NEW] اللوحة الجديدة (4 رسومات مع الـ CI) ===
    plt.figure(figsize=(24, 5)) # عرضناها عشان تسع 4 رسومات
    
    # 1. DTI MSE
    plt.subplot(1, 4, 1)
    plt.plot(history_df['Epoch'], history_df['Train_DTI_MSE'], label='Train DTI MSE', color='blue', linestyle='--')
    plt.plot(history_df['Epoch'], history_df['Val_DTI_MSE'], label='Val DTI MSE', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Score')
    plt.title('DTI MSE (Train vs Validation)')
    plt.grid(True)
    plt.legend()

    # 2. DTI CI (Validation Only)
    plt.subplot(1, 4, 2)
    plt.plot(history_df['Epoch'], history_df['Val_DTI_CI'], label='Val DTI CI', color='purple', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle=':', label='Random Guess (0.5)') # خط الـ 0.5 الاسترشادي
    plt.xlabel('Epochs')
    plt.ylabel('Concordance Index (CI)')
    plt.title('DTI CI Progression')
    plt.grid(True)
    plt.legend()

    # 3. DDI AUC
    plt.subplot(1, 4, 3)
    plt.plot(history_df['Epoch'], history_df['Train_DDI_AUC'], label='Train DDI AUC', color='orange', linestyle='--')
    plt.plot(history_df['Epoch'], history_df['Val_DDI_AUC'], label='Val DDI AUC', color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.title('DDI AUC (Train vs Validation)')
    plt.grid(True)
    plt.legend()

    # 4. DDI Accuracy
    plt.subplot(1, 4, 4)
    plt.plot(history_df['Epoch'], history_df['Train_DDI_ACC'], label='Train DDI ACC', color='green', linestyle='--')
    plt.plot(history_df['Epoch'], history_df['Val_DDI_ACC'], label='Val DDI ACC', color='green', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('DDI Accuracy (Train vs Validation)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves_overfit_check.png', dpi=300)
    print("Saved Learning Curves (with CI) to 'learning_curves_overfit_check.png'")

    # === [NEW] رسم وتصدير مصفوفة الارتباك (Confusion Matrix) ===
    if final_labels is not None and final_preds is not None:
        cm = confusion_matrix(final_labels, final_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe (0)', 'Interact (1)'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d') # values_format='d' عشان يطبع الأرقام صحيحة مش علمية
        plt.title('DDI Confusion Matrix (Cold Start - Final Epoch)')
        plt.tight_layout()
        plt.savefig('ddi_confusion_matrix.png', dpi=300)
        print("Saved Confusion Matrix to 'ddi_confusion_matrix.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kiba')
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    parser.add_argument("--load_config", type=str, default="")
    parser.add_argument("--algo", type=str, default="default")
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    new_args = vars(args)
    if new_args["load_config"] != "":
        with open(new_args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        new_args["algo"] = all_config["main_args"]["algo"]
        new_args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
    else:
        algo_args = get_defaults_yaml_args(new_args["algo"])
    new_args.update(algo_args)
    update_args(unparsed_dict, new_args)


    train_predict(new_args)
