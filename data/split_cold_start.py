import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("1. Loading original dataset...")
df = pd.read_csv('data/ddi_interactions.csv') # تأكد من المسار الصحيح

print("2. Extracting unique drugs...")
# استخراج كل الأدوية الفريدة في قاعدة البيانات
unique_drugs = list(set(df['SMILES1']).union(set(df['SMILES2'])))
print(f"Total unique drugs found: {len(unique_drugs)}")

print("3. Splitting drugs (80% Seen, 20% Unseen/Cold)...")
# نعزل 20% من الأدوية تماماً لتكون هي "الأدوية الجديدة"
seen_drugs, unseen_drugs = train_test_split(unique_drugs, test_size=0.20, random_state=42)

seen_set = set(seen_drugs)
unseen_set = set(unseen_drugs)

print("4. Routing interactions based on drug visibility...")
# بيانات التدريب: يجب أن يكون كلا الدواءين في قائمة الـ Seen
train_df = df[df['SMILES1'].isin(seen_set) & df['SMILES2'].isin(seen_set)]

# بيانات الاختبار (Cold Start): التفاعلات التي يظهر فيها أي دواء من الـ Unseen
# هذا يشمل حالة (Cold-Warm) وحالة (Cold-Cold)
test_df = df[~df.index.isin(train_df.index)]

print("5. Saving new datasets...")
train_df.to_csv('data/ddi_train_cold.csv', index=False)
test_df.to_csv('data/ddi_test_cold.csv', index=False)

print("\n✅ Splitting Complete!")
print("-" * 30)
print(f"Train Dataset (Warm Start): {len(train_df)} interactions")
print(f"Test Dataset  (Cold Start): {len(test_df)} interactions")
print("-" * 30)