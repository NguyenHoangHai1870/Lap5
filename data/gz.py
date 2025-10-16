import gzip
import shutil

input_file = "c4-train.00000-of-01024-30K.json"
output_file = input_file + ".gz"

with open(input_file, "rb") as f_in:
    with gzip.open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"✅ Đã nén thành {output_file}")
