import pandas as pd
from modules.m1_profiling.metadata import extract_metadata

df = pd.read_csv("data/raw/broadband_customers.csv")
for col, m in extract_metadata(df).items():
    print(col, m["inferred_type"], m["semantic_type"])

def test_detects_mixed_type_column():
    df = pd.DataFrame({"charges": [29.99, "£44.99", 54.99]})
    result = extract_metadata(df)
    assert result["charges"]["mixed_type"]["is_mixed"] is True