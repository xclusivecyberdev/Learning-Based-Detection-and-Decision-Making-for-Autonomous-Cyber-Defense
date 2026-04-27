#!/usr/bin/env bash
# download_dataset.sh
# Downloads CICIDS2017 dataset from the University of New Brunswick server.
# Total size: ~1.2 GB compressed, ~2.8 GB uncompressed.
#
# Usage:
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh --dir data/raw

set -e

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "  CICIDS2017 Dataset Download"
echo "  Target directory: $DATA_DIR"
echo "============================================================"
echo ""
echo "NOTE: The CICIDS2017 dataset requires a request to the University"
echo "of New Brunswick. Please visit:"
echo ""
echo "  https://www.unb.ca/cic/datasets/ids-2017.html"
echo ""
echo "After obtaining access, download the CSV files and place them in:"
echo "  $DATA_DIR/"
echo ""
echo "Expected files:"
FILES=(
    "Monday-WorkingHours.pcap_ISCX.csv"
    "Tuesday-WorkingHours.pcap_ISCX.csv"
    "Wednesday-workingHours.pcap_ISCX.csv"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)
for f in "${FILES[@]}"; do
    echo "  - $f"
done

echo ""
echo "Alternatively, a mirror may be available at:"
echo "  https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/"
echo ""

# Check if files already exist
count=$(ls "$DATA_DIR"/*.csv 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "Found $count CSV file(s) already in $DATA_DIR"
    echo "Run: python scripts/preprocess.py --input $DATA_DIR"
else
    echo "No CSV files found in $DATA_DIR. Please download manually."
fi
