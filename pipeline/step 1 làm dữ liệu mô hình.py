# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:03:56 2026

@author: tamph1
"""


#
import pandas as pd
from timeit import default_timer as timer
#from sklearn.model_selection import train_test_split
S = timer()
import numpy as np

import glob
##

start = timer()
folder_path = "D:\model\SKV parquet"
file_list = glob.glob(folder_path + "\*.parquet")
# In ra file đầu tiên trước khi đọc
print("Đang xử lý file:", file_list[0])

main_dataframe = pd.read_parquet(file_list[0],  columns = [
    'BILLING_ACCTNUM',
    'CIF',
    'ISSUE_DATE',
    'OPEN_CIF',
    'CAP_1',
    'CAP_2',
    'DM',
    'NGAY_BC',
    'LAI_SUAT',
    'SO_THE',
    'NHOM_NO',
    'DVKD_OLD',
    'TDVKD',
    'TSBD',
    'CAP_PD',
    'TEN_CPD',
    'THOIHAN',
    'CIB_SME_KHCN',
    'LTV_CIF',
    'Vung mien',
    'CAP_PD1',
    'TENNGANH_KH',
    'TENNGANH',
    'TEN_MD',
    'FLAG_LHINH',
    'TENLHINH2',
    'TENMUCDICH2',
    'DM2',
    'FIRST_BOOK_DATE',
    'CAP_PD2',
    'CAP_PD3',
    'DU_NO_ADJUST_CIF',
    'BDS',
    'DS',
    'PL_BDS_DS'
    ], engine="pyarrow")

main_dataframe['BILLING_ACCTNUM'] = main_dataframe['BILLING_ACCTNUM'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(13) if len(x) <= 13 and x != '' else x)
 
main_dataframe['CIF'] = main_dataframe['CIF'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(8) if len(x) <= 8 and x != '' else x)

main_dataframe['ISSUE_DATE'] = pd.to_datetime(main_dataframe['ISSUE_DATE'], format='%d/%m/%Y')

main_dataframe['OPEN_CIF'] = pd.to_datetime(main_dataframe['OPEN_CIF'],  format='%d/%m/%Y')

main_dataframe['NGAY_BC'] = pd.to_datetime(main_dataframe['NGAY_BC'], format='%d/%m/%Y')

main_dataframe['FIRST_BOOK_DATE'] = pd.to_datetime(main_dataframe['FIRST_BOOK_DATE'], format='%d/%m/%Y')


main_dataframe['LAI_SUAT']  = np.where(
    main_dataframe['LAI_SUAT']  < 0,
    0,
        main_dataframe['LAI_SUAT'] 
    )
        
main_dataframe['NX']  = np.where(
    main_dataframe['NHOM_NO']  >= 3,
    1,0)

del main_dataframe['NHOM_NO']
#main_dataframe = main_dataframe.drop(columns=['DU_NO_ADJUST', 'TSBD'])
main_dataframe['THOIHAN'] = main_dataframe['THOIHAN'].replace(r'^\s*$', pd.NA, regex=True).fillna('NGANHAN').astype(str)
main_dataframe['CAP_PD'] = main_dataframe['CAP_PD'].replace(r'^\s*$', pd.NA, regex=True).fillna('TRUONG DVKD').astype(str)
main_dataframe['CAP_PD1'] = main_dataframe['CAP_PD1'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
main_dataframe['CAP_PD2'] = main_dataframe['CAP_PD2'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
main_dataframe['CAP_PD3'] = main_dataframe['CAP_PD3'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
main_dataframe['TEN_CPD'] = main_dataframe['TEN_CPD'].replace(r'^\s*$', pd.NA, regex=True).fillna('Trưởng PGD').astype(str)


main_dataframe= pd.DataFrame(main_dataframe, columns = [
    'BILLING_ACCTNUM',
    'CIF',
    'ISSUE_DATE',
    'OPEN_CIF',
    'CAP_1',
    'CAP_2',
    'DM',
    'NGAY_BC',
    'LAI_SUAT',
    'SO_THE',
    'NX',
    'DVKD_OLD',
    'TDVKD',
    'TSBD',
    'CAP_PD',
    'TEN_CPD',
    'THOIHAN',
    'CIB_SME_KHCN',
    'LTV_CIF',
    'Vung mien',
    'CAP_PD1',
    'TENNGANH_KH',
    'TENNGANH',
    'TEN_MD',
    'FLAG_LHINH',
    'TENLHINH2',
    'TENMUCDICH2',
    'DM2',
    'FIRST_BOOK_DATE',
    'CAP_PD2',
    'CAP_PD3',
    'DU_NO_ADJUST_CIF',
    'BDS',
    'DS',
    'PL_BDS_DS',
    ])


for i in range(1,len(file_list)):
    # In ra file đầu tiên trước khi đọc
    print("Đang xử lý file:", file_list[i])
    data = pd.read_parquet(file_list[i],  columns = [
        'BILLING_ACCTNUM',
        'CIF',
        'ISSUE_DATE',
        'OPEN_CIF',
        'CAP_1',
        'CAP_2',
        'DM',
        'NGAY_BC',
        'LAI_SUAT',
        'SO_THE',
        'NHOM_NO',
        'DVKD_OLD',
        'TDVKD',
        'TSBD',
        'CAP_PD',
        'TEN_CPD',
        'THOIHAN',
        'CIB_SME_KHCN',
        'LTV_CIF',
        'Vung mien',
        'CAP_PD1',
        'TENNGANH_KH',
        'TENNGANH',
        'TEN_MD',
        'FLAG_LHINH',
        'TENLHINH2',
        'TENMUCDICH2',
        'DM2',
        'FIRST_BOOK_DATE',
        'CAP_PD2',
        'CAP_PD3',
        'DU_NO_ADJUST_CIF',
        'BDS',
        'DS',
        'PL_BDS_DS'
        ], engine="pyarrow")
    
    data['BILLING_ACCTNUM'] = data['BILLING_ACCTNUM'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(13) if len(x) <= 13 and x != '' else x)
     
    data['CIF'] = data['CIF'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(8) if len(x) <= 8 and x != '' else x)
    
    data['ISSUE_DATE'] = pd.to_datetime(data['ISSUE_DATE'], format='%d/%m/%Y')
    
    data['OPEN_CIF'] = pd.to_datetime(data['OPEN_CIF'],  format='%d/%m/%Y')
    
    data['NGAY_BC'] = pd.to_datetime(data['NGAY_BC'], format='%d/%m/%Y')
    
    data['FIRST_BOOK_DATE'] = pd.to_datetime(data['FIRST_BOOK_DATE'], format='%d/%m/%Y')
    
    
    data['LAI_SUAT']  = np.where(
        data['LAI_SUAT']  < 0,
        0,
            data['LAI_SUAT'] 
        )
            
    data['NX']  = np.where(
        data['NHOM_NO']  >= 3,
        1,0)
    
    del data['NHOM_NO']
    #data = data.drop(columns=['DU_NO_ADJUST', 'TSBD'])
    
    data['THOIHAN'] = data['THOIHAN'].replace(r'^\s*$', pd.NA, regex=True).fillna('NGANHAN').astype(str)
    data['CAP_PD'] = data['CAP_PD'].replace(r'^\s*$', pd.NA, regex=True).fillna('TRUONG DVKD').astype(str)
    data['CAP_PD1'] = data['CAP_PD1'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
    data['CAP_PD2'] = data['CAP_PD2'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
    data['CAP_PD3'] = data['CAP_PD3'].replace(r'^\s*$', pd.NA, regex=True).fillna('DVKD').astype(str)
    data['TEN_CPD'] = data['TEN_CPD'].replace(r'^\s*$', pd.NA, regex=True).fillna('Trưởng PGD').astype(str)
    
    
    data= pd.DataFrame(data, columns = [
        'BILLING_ACCTNUM',
        'CIF',
        'ISSUE_DATE',
        'OPEN_CIF',
        'CAP_1',
        'CAP_2',
        'DM',
        'NGAY_BC',
        'LAI_SUAT',
        'SO_THE',
        'NX',
        'DVKD_OLD',
        'TDVKD',
        'TSBD',
        'CAP_PD',
        'TEN_CPD',
        'THOIHAN',
        'CIB_SME_KHCN',
        'LTV_CIF',
        'Vung mien',
        'CAP_PD1',
        'TENNGANH_KH',
        'TENNGANH',
        'TEN_MD',
        'FLAG_LHINH',
        'TENLHINH2',
        'TENMUCDICH2',
        'DM2',
        'FIRST_BOOK_DATE',
        'CAP_PD2',
        'CAP_PD3',
        'DU_NO_ADJUST_CIF',
        'BDS',
        'DS',
        'PL_BDS_DS',
        ])


    df = pd.DataFrame(data)

    main_dataframe = pd.concat([main_dataframe,df],axis=0)
    
    main_dataframe = pd.DataFrame(main_dataframe, columns  = [
        'BILLING_ACCTNUM',
        'CIF',
        'ISSUE_DATE',
        'OPEN_CIF',
        'CAP_1',
        'CAP_2',
        'DM',
        'NGAY_BC',
        'LAI_SUAT',
        'SO_THE',
        'NX',
        'DVKD_OLD',
        'TDVKD',
        'TSBD',
        'CAP_PD',
        'TEN_CPD',
        'THOIHAN',
        'CIB_SME_KHCN',
        'LTV_CIF',
        'Vung mien',
        'CAP_PD1',
        'TENNGANH_KH',
        'TENNGANH',
        'TEN_MD',
        'FLAG_LHINH',
        'TENLHINH2',
        'TENMUCDICH2',
        'DM2',
        'FIRST_BOOK_DATE',
        'CAP_PD2',
        'CAP_PD3',
        'DU_NO_ADJUST_CIF',
        'BDS',
        'DS',
        'PL_BDS_DS',
        ])


    del df
    del data
    
data_sktv =  main_dataframe
del main_dataframe

data_sktv = pd.DataFrame(data_sktv, columns  = [
    'BILLING_ACCTNUM',
    'CIF',
    'ISSUE_DATE',
    'OPEN_CIF',
    'CAP_1',
    'CAP_2',
    'DM',
    'NGAY_BC',
    'LAI_SUAT',
    'SO_THE',
    'NX',
    'DVKD_OLD',
    'TDVKD',
    'TSBD',
    'CAP_PD',
    'TEN_CPD',
    'THOIHAN',
    'CIB_SME_KHCN',
    'LTV_CIF',
    'Vung mien',
    'CAP_PD1',
    'TENNGANH_KH',
    'TENNGANH',
    'TEN_MD',
    'FLAG_LHINH',
    'TENLHINH2',
    'TENMUCDICH2',
    'DM2',
    'FIRST_BOOK_DATE',
    'CAP_PD2',
    'CAP_PD3',
    'DU_NO_ADJUST_CIF',
    'BDS',
    'DS',
    'PL_BDS_DS',
    ])

data_sktv['BILLING_ACCTNUM'] = data_sktv['BILLING_ACCTNUM'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(13) if len(x) <= 13 and x != '' else x)

data_sktv['CIF'] = data_sktv['CIF'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).apply(lambda x: x.zfill(8) if len(x) <= 8 and x != '' else x)



# ====== CHUẨN HÓA KIỂU DỮ LIỆU ======
data_sktv['ISSUE_DATE'] = pd.to_datetime(data_sktv['ISSUE_DATE'], format='%d/%m/%Y')

data_sktv['OPEN_CIF'] = pd.to_datetime(data_sktv['OPEN_CIF'], format='%d/%m/%Y')


data_sktv['FIRST_BOOK_DATE'] = pd.to_datetime(data_sktv['FIRST_BOOK_DATE'], format='%d/%m/%Y')

data_sktv['NGAY_BC'] = pd.to_datetime(data_sktv['NGAY_BC'], format='%d/%m/%Y')

# ====== BƯỚC 1: ĐIỀN FIRST_BOOK_DATE ======

data_sktv['OPEN_CIF'] = data_sktv.groupby(['CIF']).OPEN_CIF.transform('min')
data_sktv['FIRST_BOOK_DATE'] = data_sktv['FIRST_BOOK_DATE'].fillna(data_sktv['OPEN_CIF'])  # Công thức: FIRST_BOOK_DATE trống thì lấy OPEN_CIF cùng dòng

data_sktv['FIRST_BOOK_DATE'] = data_sktv.groupby(['CIF']).FIRST_BOOK_DATE.transform('min')

##
#thêm phần lọc dữ liệu TT14 tại đây

###



data_sktv['DURATION_ISSUE_DATE'] = (pd.DatetimeIndex(data_sktv['NGAY_BC']).year - pd.DatetimeIndex(data_sktv['ISSUE_DATE']).year) * 12 + (pd.DatetimeIndex(data_sktv['NGAY_BC']).month - pd.DatetimeIndex(data_sktv['ISSUE_DATE']).month)

data_sktv['DURATION_ISSUE_DATE'] = np.where(
data_sktv['DURATION_ISSUE_DATE'] < 0,
0,
data_sktv['DURATION_ISSUE_DATE'])


data_sktv['DURATION_FIRST_BOOK_DATE'] = (pd.DatetimeIndex(data_sktv['NGAY_BC']).year - pd.DatetimeIndex(data_sktv['FIRST_BOOK_DATE']).year) * 12 + (pd.DatetimeIndex(data_sktv['NGAY_BC']).month - pd.DatetimeIndex(data_sktv['FIRST_BOOK_DATE']).month)

data_sktv['DURATION_FIRST_BOOK_DATE'] = np.where(
data_sktv['DURATION_FIRST_BOOK_DATE'] < 0,
0,
data_sktv['DURATION_FIRST_BOOK_DATE'])


data_sktv['DURATION_OPEN_CIF'] = (pd.DatetimeIndex(data_sktv['NGAY_BC']).year - pd.DatetimeIndex(data_sktv['OPEN_CIF']).year) * 12 + (pd.DatetimeIndex(data_sktv['NGAY_BC']).month - pd.DatetimeIndex(data_sktv['OPEN_CIF']).month)

data_sktv['DURATION_OPEN_CIF'] = np.where(
data_sktv['DURATION_OPEN_CIF'] < 0,
0,
data_sktv['DURATION_OPEN_CIF'])

data_sktv['DU_NO_ADJUST_CIF'] = data_sktv['DU_NO_ADJUST_CIF'].fillna(0).apply(float)
data_sktv['TSBD'] = data_sktv['TSBD'].fillna(0).apply(float)
data_sktv['BDS'] = data_sktv['BDS'].fillna(0).apply(float)
data_sktv['DS'] = data_sktv['DS'].fillna(0).apply(float)


data_sktv['CAP_PD'] = data_sktv['CAP_PD'].fillna('').astype(str)
data_sktv['TEN_CPD'] = data_sktv['TEN_CPD'].fillna('').astype(str)
data_sktv['CAP_PD1'] = data_sktv['CAP_PD1'].fillna('').astype(str)
data_sktv['CAP_PD2'] = data_sktv['CAP_PD2'].fillna('').astype(str)
data_sktv['CAP_PD3'] = data_sktv['CAP_PD3'].fillna('').astype(str)

data_sktv["LTV_CIF"] = data_sktv["LTV_CIF"].apply(
            lambda x: float(x)
            if pd.notna(x) and x != "tín chấp"
            else x
        )


##
#thêm phần lọc dữ liệu TT14 tại đây
#...
###






#######chỉnh từ vùng này
####### chỉnh từ vùng này
#....
data_sktv['train_test']  =''
####### chỉnh đến vùng này





del data_sktv['ISSUE_DATE']
del data_sktv['FIRST_BOOK_DATE']
del data_sktv['OPEN_CIF']
#del data_sktv['FIRST_BOOK_DATE_HISTORY_MONTHS']
#del data_sktv['MAX_NGAY_BC_CIF']

data_sktv['LTV_CIF'] = data_sktv['LTV_CIF'].astype('string')
#data_sktv =  data_sktv.drop_duplicates(keep='last')

data_sktv = pd.DataFrame(data_sktv, columns = ['BILLING_ACCTNUM',
'CIF',
'CAP_1',
'CAP_2',
'DM',
'NGAY_BC',
'LAI_SUAT',
'SO_THE',
'NX',
'DVKD_OLD',
'TDVKD',
'TSBD',
'CAP_PD',
'TEN_CPD',
'THOIHAN',
'CIB_SME_KHCN',
'LTV_CIF',
'Vung mien',
'CAP_PD1',
'TENNGANH_KH',
'TENNGANH',
'TEN_MD',
'FLAG_LHINH',
'TENLHINH2',
'TENMUCDICH2',
'DM2',
'CAP_PD2',
'CAP_PD3',
'DU_NO_ADJUST_CIF',
'BDS',
'DS',
'PL_BDS_DS',
 'DURATION_ISSUE_DATE',
 'DURATION_OPEN_CIF',
 'DURATION_FIRST_BOOK_DATE',
 'train_test'])

data_sktv.to_parquet(
    r"D:\model\credit_data.parquet",
    index=False,
    compression="brotli"
)



from pathlib import Path
import math

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path(r"D:\model")
MAX_SIZE_MB = 20
COMPRESSION = "brotli"

OUTPUT_FILE_1 = OUTPUT_DIR / "credit_data1.parquet"
OUTPUT_FILE_2 = OUTPUT_DIR / "credit_data2.parquet"

# =========================
# CHIA DỮ LIỆU THÀNH 2 PHẦN
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

split_index = math.ceil(len(data_sktv) / 2)

data_sktv_1 = data_sktv.iloc[:split_index]
data_sktv_2 = data_sktv.iloc[split_index:]

# =========================
# XUẤT FILE PARQUET
# =========================
data_sktv_1.to_parquet(
    OUTPUT_FILE_1,
    index=False,
    compression=COMPRESSION
)

data_sktv_2.to_parquet(
    OUTPUT_FILE_2,
    index=False,
    compression=COMPRESSION
)

# =========================
# KIỂM TRA DUNG LƯỢNG
# =========================
size_file_1_mb = OUTPUT_FILE_1.stat().st_size / (1024 ** 2)
size_file_2_mb = OUTPUT_FILE_2.stat().st_size / (1024 ** 2)

print(f"File 1: {OUTPUT_FILE_1}")
print(f"Số dòng: {len(data_sktv_1):,}")
print(f"Dung lượng: {size_file_1_mb:.2f} MB")

print(f"\nFile 2: {OUTPUT_FILE_2}")
print(f"Số dòng: {len(data_sktv_2):,}")
print(f"Dung lượng: {size_file_2_mb:.2f} MB")

if size_file_1_mb <= MAX_SIZE_MB and size_file_2_mb <= MAX_SIZE_MB:
    print("\nHoàn tất: Cả 2 file đều không vượt quá 20 MB.")
else:
    print(
        "\nCảnh báo: Có file vượt quá 20 MB. "
        "Dữ liệu cần được chia thành nhiều hơn 2 file."
    )


del data_sktv











