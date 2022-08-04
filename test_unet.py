import os
import load_and_predict as lp

model_name = "average_1"
test_file = "external"

PATH = "/data/gcm49/experiment3"

model_path = f"{PATH}/models/{model_name}.h5"
test_path = f"{PATH}/hdf5_files/{test_file}.h5"
out_path = f"{PATH}/predictions/{model_name}/"

if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating new directory: {out_path}")

out_path += f"{test_file}_predictions.h5"

lp.load_and_predict(model_path, test_path, out_path)