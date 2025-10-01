import gzip
import shutil

# Tên file gốc (JSON)
input_file = "c4-train.00000-of-01024-30K.json"
# Tên file sau khi nén
output_file = "c4-train.00000-of-01024-30K.json.gz"

# Nén file
with open(input_file, "rb") as f_in:
    with gzip.open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Đã nén xong: {output_file}")
