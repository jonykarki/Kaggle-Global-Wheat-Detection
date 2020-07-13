import sys, os
if os.path.abspath(os.pardir) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.pardir))
import CONFIG

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA_DIR = CONFIG.CFG.DATA.BASE

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
bboxes = np.stack(train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    train_df[column] = bboxes[:, i]
train_df.drop(columns=['bbox'], inplace=True)

print(train_df.head())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = train_df[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:,'source'] = train_df[['image_id','source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

df_folds.to_csv("folds.csv")

print(f"Created folds.csv with shape {df_folds.shape}")