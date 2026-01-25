# make a 10k-row sample (keeps header)
head -n 1 /tmp/financial_fraud_detection_dataset.csv > /tmp/sample_10k.csv
tail -n +2 /tmp/financial_fraud_detection_dataset.csv | head -n 10000 >> /tmp/sample_10k.csv

# run pipeline on sample and capture logs
python -u run_pipeline.py artifacts/lgb_baseline_1f0b62ecf771.pkl artifacts/ae_anomaly_b1aa1157ad73.pkl /tmp/sample_10k.csv > pipeline_sample.log 2>&1

# follow logs
tail -n 200 pipeline_sample.log
# inspect results
ls -lh artifacts/results.parquet