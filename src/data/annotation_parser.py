from pathlib import Path
from typing import List, Dict
import pandas as pd


def load(csv_path: str) -> List[Dict]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df = df[["operation", "start", "end"]].copy()
    df = df.dropna(subset=["operation", "start", "end"])

    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    df = df[df["end"] > df["start"]]

    df = df.sort_values("start").reset_index(drop = True)

    operation = []

    for i, row in df.iterrows():
        operation.append(
            {
                "operation":row["operation"],
                "start" : row["start"],
                "end" : row["end"],
            }
        )

    return operation

# if __name__ == "__main__":
#     ops = load(
#         "C:/Users/Chetak/Documents/GitHub/projects/VLM Challenge/Dataset/U0101/annotation/openpack-operations/S0100.csv"
#     )

#     print(f"Total operations: {len(ops)}")
#     print(ops[:3])