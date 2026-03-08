import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random

dB_file = 'Datasets/full database.xml'
smiles_file = 'GCDTA/drugbank_smiles.csv'
saveFile = 'GCDTA/data/ddi_interactions.csv' # تأكد من وجود مجلد data

# 1. قراءة ملف الـ SMILES وتحويله لقاموس (لربط الـ ID بالـ SMILES لاحقاً)
print("Loading SMILES dictionary...")
df_smiles = pd.read_csv(smiles_file)
df_smiles = df_smiles.dropna(subset=['smiles'])
id_to_smiles = dict(zip(df_smiles['dg_id'], df_smiles['smiles']))
valid_ids = list(id_to_smiles.keys())

# مجموعة لتخزين التفاعلات الإيجابية بدون تكرار
positive_pairs = set()
ns = '{http://www.drugbank.ca}' # الـ Namespace الذي يتماشى مع طريقتك

# 2. استخدام iterparse لمعالجة الملف تدفقياً (Stream)
print("Parsing XML for Drug Interactions...")
context = ET.iterparse(dB_file, events=("end",))

for event, elem in tqdm(context):
    # نحن نبحث عن وسوم 'drug' فقط
    if elem.tag.endswith('drug'):
        # استخراج المعرف الأساسي للدواء
        dg_id_elem = elem.find(f'{ns}drugbank-id')
        dg_id = dg_id_elem.text if dg_id_elem is not None else None
        
        # إذا كان الدواء موجوداً في قاموسنا (له SMILES)
        if dg_id and dg_id in id_to_smiles:
            
            # البحث عن قسم التفاعلات الدوائية بناءً على هيكل الـ XML
            interactions_node = elem.find(f'{ns}drug-interactions')
            if interactions_node is not None:
                # المرور على كل تفاعل داخل القسم
                for interaction in interactions_node.findall(f'{ns}drug-interaction'):
                    interact_id_elem = interaction.find(f'{ns}drugbank-id')
                    
                    if interact_id_elem is not None:
                        interact_id = interact_id_elem.text
                        
                        # التأكد أن الدواء المتفاعل له SMILES أيضاً في القاموس
                        if interact_id in id_to_smiles:
                            # ترتيب المعرفين أبجدياً لمنع التكرار (A, B) هو نفسه (B, A)
                            pair = tuple(sorted([dg_id, interact_id]))
                            positive_pairs.add(pair)
        
        # أهم خطوة: تنظيف الذاكرة بعد كل عقار
        elem.clear()

num_positives = len(positive_pairs)
print(f"\nFound {num_positives} positive drug interactions.")

# 3. توليد العينات السلبية (Negative Samples) للحفاظ على توازن البيانات
print("Generating negative samples (safe pairs)...")
negative_pairs = set()

pbar = tqdm(total=num_positives)
while len(negative_pairs) < num_positives:
    # اختيار دواءين عشوائياً من الأدوية المتاحة
    d1, d2 = random.sample(valid_ids, 2)
    pair = tuple(sorted([d1, d2]))
    
    # التأكد أنهما لا يتفاعلان ولم يتم إضافتهما مسبقاً
    if pair not in positive_pairs and pair not in negative_pairs:
        negative_pairs.add(pair)
        pbar.update(1)
pbar.close()

# 4. تجميع البيانات وحفظها في CSV
print("Saving to CSV...")
rows = []

# إضافة التفاعلات الإيجابية (Label = 1)
for d1, d2 in positive_pairs:
    rows.append({
        'SMILES1': id_to_smiles[d1],
        'SMILES2': id_to_smiles[d2],
        'Label': 1
    })

# إضافة التفاعلات السلبية (Label = 0)
for d1, d2 in negative_pairs:
    rows.append({
        'SMILES1': id_to_smiles[d1],
        'SMILES2': id_to_smiles[d2],
        'Label': 0
    })

df_final = pd.DataFrame(rows)

# خلط الصفوف عشوائياً (Shuffle) لكي يكون التدريب سليماً
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv(saveFile, index=False)
print(f"Done! Saved {len(df_final)} total pairs to {saveFile}.")