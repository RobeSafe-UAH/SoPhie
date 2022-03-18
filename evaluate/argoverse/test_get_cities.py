import numpy as np
import os
import csv
import pdb

split = "val"

root_folder = "data/datasets/argoverse/motion-forecasting/"
root_file_name = root_folder + split + "/data"

filename = root_folder + split + "/data_processed/" + "num_seq_list" + ".npy"
with open(filename, 'rb') as my_file: num_seq_list = np.load(my_file)

city_ids = []
seq_len = 50

def read_file(_path):
    data = csv.DictReader(open(_path))
    aux = []
    id_list = []

    num_agent = 0
    for row in data:
        values = list(row.values())
        # object type 
        values[2] = 0 if values[2] == "AV" else 1 if values[2] == "AGENT" else 2
        if values[2] == 1:
            num_agent += 1
        # city
        values[-1] = 0 if values[-1] == "PIT" else 1
        # id
        id_list.append(values[1])
        # numpy_sequence
        aux.append(values)
 
    id_list, id_idx = np.unique(id_list, return_inverse=True)
    data = np.array(aux)
    data[:, 1] = id_idx

    return data.astype(np.float64)

for i, file_id in enumerate(num_seq_list):
    print(f"File {file_id} -> {i}/{len(num_seq_list)}")

    path = os.path.join(root_file_name,str(file_id)+".csv")
    data = read_file(path) 

    frames = np.unique(data[:, 0]).tolist() 
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :]) # save info for each frame

    idx = 0

    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)

    city_id = curr_seq_data[0,5]

    city_ids.append(city_id)

filename2 = root_folder + split + "/data_processed/" + "city_ids" + ".npy"
with open(filename2, 'wb') as my_file: np.save(my_file, city_ids)