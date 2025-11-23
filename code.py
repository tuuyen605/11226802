# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Sklearn Modules
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

# XGBoost
from xgboost import XGBClassifier

# --- 2. THU THẬP DỮ LIỆU ---
file_name = 'KLTN.xlsx' 
df = pd.read_excel(file_name)
print("Dữ liệu đã được import thành công!")
print(df.head())

# --- 3. TIỀN XỬ LÝ DỮ LIỆU & LÀM SẠCH ---

# 3.1. Loại bỏ các cột không cần thiết cho mô hình
df.drop(['customer_id', 'year', 'phone_no'], axis=1, inplace=True, errors='ignore')
df.info()

# 3.2. Xử lý giá trị khuyết (fillna)
print("\nKiểm tra giá trị khuyết trước khi xử lý:")
null_df = df.isnull().sum().to_frame(name='Số lượng khuyết')
null_df