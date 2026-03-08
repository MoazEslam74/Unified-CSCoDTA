import pandas as pd
import glob
import os

# 1. حدد المسار الفعلي (استخدم '.' إذا كانت الملفات في نفس مجلد السكريبت)
path = 'DDInter'  
# تأكد من أن أسماء الملفات تبدأ فعلياً بـ ddinter
file_pattern = os.path.join(path, "ddinter_downloads_code_*.csv")
all_files = glob.glob(file_pattern)

# تحقق من وجود ملفات قبل المتابعة
if not all_files:
    print(f"Error: No files found in path {os.path.abspath(path)}")
    print("Be sure that the ddinter_downloads_code_*.csv files exist in the DDInter directory.")
else:
    li = []
    for filename in all_files:
        try:
            # قراءة الملف
            df = pd.read_csv(filename, index_col=None, header=0)
            
            # استخراج حرف الفئة من اسم الملف (A, B, L, إلخ)
            category = os.path.basename(filename).split('_')[-1].replace('.csv', '')
            df['ATC_Category'] = category
            
            li.append(df)
            print(f"تمت قراءة الملف: {filename}")
        except Exception as e:
            print(f"حدث خطأ أثناء قراءة {filename}: {e}")

    # 2. الدمج فقط إذا كانت القائمة تحتوي على جداول
    if li:
        full_ddi_data = pd.concat(li, axis=0, ignore_index=True)
        # حفظ الملف النهائي في مسارك الحالي بمجلد Moaz Eslam
        output_name = 'DDInter_Full_Dataset.csv'
        full_ddi_data.to_csv(output_name, index=False)
        print("-" * 30)
        print(f"✅ تم بنجاح! إجمالي التفاعلات المدمجة: {len(full_ddi_data)}")
        print(f"تم حفظ الملف باسم: {output_name}")
    else:
        print("فشل الدمج: القائمة لا تحتوي على بيانات.")