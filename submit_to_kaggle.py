import os
import pandas as pd


SS_CSV   = os.path.join('E://hubmap-organ-segmentation', "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)

TEST_CSV = os.path.join('E://hubmap-organ-segmentation', "test.csv")
test_df = pd.read_csv(TEST_CSV)
sub=pd.read_csv('UneXt.csv')
ss_df['rle']=sub['rle']
ss_df = ss_df[["id", "rle"]]
ss_df.to_csv("submission.csv", index=False)