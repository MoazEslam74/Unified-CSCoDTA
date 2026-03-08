import pandas as pd

print("1. Loading all datasets...")
# قم بتغيير أسماء الملفات لتطابق الأسماء الحقيقية على جهازك
drugbank_df = pd.read_csv('drugbank_smiles.csv') # ملف القاموس
ddinter_df = pd.read_csv('DDInter_Full_Dataset.csv')          # ملف الخطورة
my_ddi_df = pd.read_csv('data/ddi_interactions.csv')  # ملف مشروعك الأصلي

# ==========================================
# الخطوة الأولى: إنشاء قاموس (الاسم -> SMILES) من DrugBank
# ==========================================
print("2. Building DrugBank Dictionary...")
# تحويل أسماء الأدوية لحروف صغيرة لإلغاء الفروق، ثم ربطها بالـ SMILES
name_to_smiles = dict(zip(
    drugbank_df['dg_name'].astype(str).str.lower().str.strip(), 
    drugbank_df['smiles'].astype(str)
))

# ==========================================
# الخطوة الثانية: ترجمة ملف DDInter إلى SMILES
# ==========================================
print("3. Translating DDInter names to SMILES...")
ddinter_df['SMILES_A'] = ddinter_df['Drug_A'].astype(str).str.lower().str.strip().map(name_to_smiles)
ddinter_df['SMILES_B'] = ddinter_df['Drug_B'].astype(str).str.lower().str.strip().map(name_to_smiles)

# حساب عدد الأدوية التي لم نجد لها SMILES
missing_smiles = ddinter_df['SMILES_A'].isna().sum() + ddinter_df['SMILES_B'].isna().sum()
print(f"   -> Successfully mapped DDInter. Missing/Unmatched drugs: {missing_smiles}")

# التخلص من الصفوف التي لم نجد لها SMILES (لتجنب الأخطاء)
ddinter_clean = ddinter_df.dropna(subset=['SMILES_A', 'SMILES_B'])

# ==========================================
# الخطوة الثالثة: تجهيز قاموس الخطورة (SMILES Pair -> Level)
# ==========================================
print("4. Building Severity Dictionary...")
severity_map = {}
for index, row in ddinter_clean.iterrows():
    # ترتيب الـ SMILES أبجدياً لضمان تطابق الدواءين بغض النظر عن من الأول ومن الثاني
    smiles_pair = tuple(sorted([str(row['SMILES_A']), str(row['SMILES_B'])]))
    severity_map[smiles_pair] = row['Level'] # Major, Moderate, Minor

# ==========================================
# الخطوة الرابعة: دمج مستوى الخطورة في ملفك الأصلي (الدفاعي)
# ==========================================
print("5. Injecting Severity Level into original dataset...")

def assign_severity(row):
    # إذا كان الـ Label القديم 0 (لا يوجد تفاعل)، إذن الخطورة Safe
    if row['Label'] == 0:
        return 'Safe'
    
    # ترتيب الـ SMILES للبحث في القاموس
    smiles_pair = tuple(sorted([str(row['SMILES1']), str(row['SMILES2'])]))
    
    # إرجاع درجة الخطورة، وإذا لم يكن التفاعل موجوداً في DDInter نكتب Unknown
    return severity_map.get(smiles_pair, 'Unknown')

# إنشاء العمود الجديد
my_ddi_df['Severity_Level'] = my_ddi_df.apply(assign_severity, axis=1)

# ==========================================
# الخطوة الأخيرة: حفظ الملف النهائي
# ==========================================
output_name = 'ddi_interactions_with_severity.csv'
my_ddi_df.to_csv(output_name, index=False)

print(f"\n✅ Done! File saved as: {output_name}")
print("\n--- Severity Distribution in your new dataset ---")
print(my_ddi_df['Severity_Level'].value_counts())