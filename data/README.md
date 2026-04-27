# Dataset Setup

This project uses the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity.

## Download

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Request access and download the CSV files
3. Place all CSV files in `data/raw/`

## Expected Files

```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Preprocessing

```bash
python scripts/preprocess.py --input data/raw/ --output data/processed/
```

This will:
- Clean inf/NaN values
- Encode labels
- Apply SMOTE oversampling on the training split
- Fit and save a MinMaxScaler
- Create sliding windows (size=50) for LSTM
- Save X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy

## Citation

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
Toward generating a new intrusion detection dataset and intrusion traffic characterization.
*ICISSP 2018*.
