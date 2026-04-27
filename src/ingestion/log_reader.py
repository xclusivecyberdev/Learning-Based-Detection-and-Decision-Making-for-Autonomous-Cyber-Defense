"""
log_reader.py
Reads raw network log files (syslog, CSV, pcap-derived) from disk.
Used in live mode for real-time threat detection.
"""

import os
import csv
import logging
import pandas as pd
from typing import Iterator

logger = logging.getLogger(__name__)


class LogReader:
    """
    Reads CSV-format network flow logs (compatible with CICIDS2017 schema).
    Yields rows as dicts for the feature extractor.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def tail_csv(self, filepath: str, chunk_size: int = 64) -> Iterator[pd.DataFrame]:
        """
        Continuously tail a CSV file, yielding new chunks as they arrive.
        For live-mode monitoring of a network capture tool's output.
        """
        position = 0
        header = None

        while True:
            with open(filepath, "r") as f:
                f.seek(position)
                reader = csv.DictReader(f)
                if header is None:
                    header = reader.fieldnames

                rows = list(reader)
                position = f.tell()

            if rows:
                df = pd.DataFrame(rows)
                # Convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass
                yield df

            import time
            time.sleep(0.5)

    def read_csv_batch(self, filepath: str) -> pd.DataFrame:
        """Read a complete CSV file into a DataFrame."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Log file not found: {filepath}")
        logger.info(f"Reading log file: {filepath}")
        return pd.read_csv(filepath, low_memory=False)

    def list_log_files(self, extension: str = ".csv") -> list:
        """List all log files in the configured directory."""
        files = []
        for fname in os.listdir(self.log_dir):
            if fname.endswith(extension):
                files.append(os.path.join(self.log_dir, fname))
        return sorted(files)
