
## 初回実行

copy_files_to_pattern(
    file_list_source="./output/candidate_metrics.csv",  # 採用のみなら CURATED_CSV_PATH
    dest_root=RECOPY_DEST_ROOT,
    pattern_col="facing_direction",   # front / side_left / side_right / three_quarter_* へ自動振り分け
    only_selected=False,              # 採用のみなら True
    clear_dest=True,                  # やり直し時にコピー先の古いファイルを自動で空にする
    dry_run=True,                     # ← 最初は True で予定だけ確認
)
