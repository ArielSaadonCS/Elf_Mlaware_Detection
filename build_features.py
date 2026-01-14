"""
Build Features for all three pipelines
Creates separate CSV files for each pipeline for training and comparison.
"""
import os
import pandas as pd
from pipeline_a import extract_pipeline_a_vector, get_pipeline_a_feature_names
from pipeline_b import extract_pipeline_b_vector, get_pipeline_b_feature_names
from pipeline_c import extract_pipeline_c_vector, get_pipeline_c_feature_names


def build_dataset(basedir="elf_dataset"):
    """
    Build feature datasets for all three pipelines.
    Creates:
    - features_pipeline_a.csv
    - features_pipeline_b.csv
    - features_pipeline_c.csv
    """
    
    rows_a = []
    rows_b = []
    rows_c = []
    
    for label_dir, label in [("benign", 0), ("malware", 1)]:
        folder = os.path.join(basedir, label_dir)
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
            continue
        
        files = os.listdir(folder)
        print(f"Processing {len(files)} files from {label_dir}...")
        
        for i, fname in enumerate(files):
            fpath = os.path.join(folder, fname)
            
            try:
                with open(fpath, 'rb') as f:
                    data = f.read()
                
                # Check ELF magic
                if data[:4] != b'\x7fELF':
                    print(f"  Skipping {fname}: Not an ELF file")
                    continue
                
                # Extract features for each pipeline
                feats_a = extract_pipeline_a_vector(data)
                feats_b = extract_pipeline_b_vector(data)
                feats_c = extract_pipeline_c_vector(data)
                
                rows_a.append(feats_a + [label])
                rows_b.append(feats_b + [label])
                rows_c.append(feats_c + [label])
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(files)} files...")
                    
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue
    
    # Create DataFrames and save
    cols_a = get_pipeline_a_feature_names() + ['label']
    cols_b = get_pipeline_b_feature_names() + ['label']
    cols_c = get_pipeline_c_feature_names() + ['label']
    
    df_a = pd.DataFrame(rows_a, columns=cols_a)
    df_b = pd.DataFrame(rows_b, columns=cols_b)
    df_c = pd.DataFrame(rows_c, columns=cols_c)
    
    df_a.to_csv("features_pipeline_a.csv", index=False)
    df_b.to_csv("features_pipeline_b.csv", index=False)
    df_c.to_csv("features_pipeline_c.csv", index=False)
    
    print("\n" + "=" * 50)
    print("Feature extraction complete!")
    print("=" * 50)
    print(f"\nPipeline A (Structural): {len(df_a)} samples, {len(cols_a)-1} features")
    print(f"  -> features_pipeline_a.csv")
    print(f"\nPipeline B (Statistical): {len(df_b)} samples, {len(cols_b)-1} features")
    print(f"  -> features_pipeline_b.csv")
    print(f"\nPipeline C (Hybrid):      {len(df_c)} samples, {len(cols_c)-1} features")
    print(f"  -> features_pipeline_c.csv")
    
    print(f"\nClass distribution:")
    print(f"  Benign:  {len(df_a[df_a['label']==0])}")
    print(f"  Malware: {len(df_a[df_a['label']==1])}")


if __name__ == "__main__":
    build_dataset()
