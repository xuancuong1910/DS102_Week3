from pathlib import Path
import pandas as pd
from typing import Union

def get_data_info(base_path: Union[str, Path]) -> pd.DataFrame:
    data = []
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Error: The path {base_dir} does not exist.")
        return pd.DataFrame()

    for split in ['train', 'test', 'val']:
        split_path = base_dir / split
        
        for label in ['NORMAL', 'PNEUMONIA']:
            class_path = split_path / label
            
            if class_path.exists():
                
                for img_path in class_path.iterdir():
                    if img_path.is_file():
                        
                        data.append({
                            'path': str(img_path),
                            'label': 1 if label == 'PNEUMONIA' else -1,
                            'split': split
                        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    base_raw_path = Path(r"D:\24520242_DS102_Week3\Assignment_3\data")
    df = get_data_info(base_raw_path)
    
    print("--- Thống kê dữ liệu ---")
    if not df.empty:
        print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
        print(f"\nTổng cộng: {len(df)} ảnh.")
    else:
        print("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại đường dẫn base_raw_path.")