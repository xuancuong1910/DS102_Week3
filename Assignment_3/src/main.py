from data_prep import get_data_info 
from preprocessing import load_data 
from model import SVM
from evaluate import evaluate_metrics
import pandas as pd
from sklearn.svm import LinearSVC


if __name__ == "__main__":
    base_raw_path = r"D:\24520242_DS102_Week3\Assignment_3\data"
    df = get_data_info(base_raw_path)
    
    
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    
    print("Đang tải và xử lý ảnh (vui lòng đợi)...")

    
    DATA_DIR = r"D:\24520242_DS102_Week3\Assignment_3\data" 

    
    X_train, y_train = load_data(DATA_DIR, split="train")
    X_test, y_test = load_data(DATA_DIR, split="test")
    
    
    print("Bắt đầu huấn luyện SVM bằng SGD...")
    
    model = SVM(C=1.0, lr=0.001, epochs=50) 
    model.fit(X_train, y_train)
    
    
    print("Đang dự đoán trên tập Test...")
    y_pred = model.predict(X_test)
    
   
    p, r, f1 = evaluate_metrics(y_test, y_pred)
    
    print("\n--- KẾT QUẢ ASSIGNMENT 1 ---")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    print("\n[Sklearn] Bắt đầu huấn luyện...")
    
    sklearn_model = LinearSVC(C=1.0, max_iter=1000, dual=False)
    sklearn_model.fit(X_train, y_train)
    
    print("[Mô hình Sklearn] Đang dự đoán...")
    y_pred_sk = sklearn_model.predict(X_test)
    p_sk, r_sk, f1_sk = evaluate_metrics(y_test, y_pred_sk)
    
    print("\n--- KẾT QUẢ ASSIGNMENT 2 ---")
    print(f"Precision: {p_sk:.4f}")
    print(f"Recall:    {r_sk:.4f}")
    print(f"F1-score:  {f1_sk:.4f}")

    comparison_data = {
        "Chỉ số": ["Precision", "Recall", "F1-score"],
        "Soft-Margin SVM": [p, r, f1],
        "Sklearn SVM": [p_sk, r_sk, f1_sk]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    print("\n" + "="*40)
    print("SO SÁNH KẾT QUẢ PHÂN LOẠI")
    print("="*40)
    print(df_compare.to_string(index=False))
      
