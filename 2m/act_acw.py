import os
import csv
import datetime as dt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from transformers import T5Model, T5Config
import torch
import torch.nn as nn
import random
import sys
from scipy import fftpack

random.seed(0)
np.random.seed(1)
torch.manual_seed(2)

frame_size = 3*1
activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

act_path = '/home/mex/data/act/'
acw_path = '/home/mex/data/acw/'
results_file = '/home/mex/results_lopo/2m/chronos_ac_2m_transformer.csv'

frames_per_second = 100
window = 5
increment = 3  # Changed to 3-second stride
embedding_dim = 512  # Embedding dimension for Chronos-T5-Base
ac_min_length = 95*window
ac_max_length = 100*window
fusion = int(sys.argv[1])  # Should be 0 for early fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()

def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0]+'.000000'
        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
        _temp = [float(f) for f in row[1:]]
        temp.extend(_temp)
        _data.append(temp)
    return _data

def read(path, _sensor):
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].replace(_sensor, '')
            activity_id = sensor.split('_')[0]
            sensor_index = sensor.split('_')[1]
            _data = _read(os.path.join(subject_path, activity))
            if activity_id in allactivities:
                allactivities[activity_id][sensor_index] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor_index] = _data
        alldata[subject] = allactivities
    return alldata

def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]

def trim(_data):
    _length = len(_data)
    _inc = _length/(window*frames_per_second)
    _new_data = []
    for i in range(window*frames_per_second):
        idx = int(i*_inc)
        _new_data.append(_data[idx])
    return _new_data

def frame_reduce(_features):
    if frames_per_second == 0:
        return _features
    new_features = {}
    for subject in _features:
        _activities = {}
        activities = _features[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                new_item = []
                new_item.append(trim(item[0]))
                new_item.append(trim(item[1]))
                time_windows.append(new_item)
            _activities[activity] = time_windows
        new_features[subject] = _activities
    return new_features

def split_windows(act_data, acw_data):
    outputs = []
    start = act_data[0][0]
    end = act_data[len(act_data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    act_frames = [a[1:] for a in act_data[:]]
    act_frames = np.array(act_frames)
    act_length = act_frames.shape[0]
    act_frames = np.reshape(act_frames, (act_length*frame_size))
    act_frames = act_frames/(max(act_frames)-min(act_frames))
    act_frames = [float("{0:.5f}".format(f)) for f in act_frames.tolist()]
    act_frames = np.reshape(np.array(act_frames), (act_length, frame_size))

    acw_frames = [a[1:] for a in acw_data[:]]
    acw_frames = np.array(acw_frames)
    acw_length = acw_frames.shape[0]
    acw_frames = np.reshape(acw_frames, (acw_length*frame_size))
    acw_frames = acw_frames/(max(acw_frames)-min(acw_frames))
    acw_frames = [float("{0:.5f}".format(f)) for f in acw_frames.tolist()]
    acw_frames = np.reshape(np.array(acw_frames), (acw_length, frame_size))

    start_times = []  # Store start times for positional encoding
    while start + _window < end:
        _end = start + _window
        act_start_index = find_index(act_data, start)
        act_end_index = find_index(act_data, _end)
        acw_start_index = find_index(acw_data, start)
        acw_end_index = find_index(acw_data, _end)
        act_instances = [a[:] for a in act_frames[act_start_index:act_end_index]]
        acw_instances = [a[:] for a in acw_frames[acw_start_index:acw_end_index]]
        start_times.append((start - act_data[0][0]).total_seconds())  # Relative start time
        start = start + _increment
        instances = [act_instances, acw_instances]
        outputs.append(instances)
    return outputs, start_times

def extract_features(act_data, acw_data):
    _features = {}
    for subject in act_data:
        _activities = {}
        act_activities = act_data[subject]
        for act_activity in act_activities:
            time_windows = []
            start_times = []
            activity_id = activity_id_dict.get(act_activity)
            act_activity_data = act_activities[act_activity]
            acw_activity_data = acw_data[subject][act_activity]
            for item in act_activity_data.keys():
                windows, times = split_windows(act_activity_data[item], acw_activity_data[item])
                time_windows.extend(windows)
                start_times.extend(times)
            _activities[activity_id] = (time_windows, start_times)
        _features[subject] = _activities
    return _features

def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data

def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data

def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items, start_times = activities[act]
            new_items = []
            new_times = []
            for idx, item in enumerate(items):
                new_item = []
                act_len = len(item[0])
                acw_len = len(item[1])
                if act_len < ac_min_length or acw_len < ac_min_length:
                    continue
                if act_len > ac_max_length:
                    new_item.append(reduce(item[0], act_len - ac_max_length))
                elif act_len < ac_max_length:
                    new_item.append(pad(item[0], ac_max_length - act_len))
                else:
                    new_item.append(item[0])

                if acw_len > ac_max_length:
                    new_item.append(reduce(item[1], acw_len - ac_max_length))
                elif acw_len < ac_max_length:
                    new_item.append(pad(item[1], ac_max_length - acw_len))
                else:
                    new_item.append(item[1])
                new_items.append(new_item)
                new_times.append(start_times[idx])
            new_activities[act] = (new_items, new_times)
        new_features[subject] = new_activities
    return new_features

def get_sinusoidal_positional_encoding(num_patches, embedding_dim, start_times):
    # Positional encoding based on start times (in seconds)
    positions = torch.tensor(start_times, dtype=torch.float32).unsqueeze(1)  # [num_patches, 1]
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embedding_dim))
    pe = torch.zeros(num_patches, embedding_dim)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe  # [num_patches, embedding_dim]

def flatten(_data):
    flatten_data = []
    flatten_labels = []
    flatten_times = []
    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data, start_times = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for _ in range(len(activity_data))])
            flatten_times.extend(start_times)
    return flatten_data, flatten_labels, flatten_times

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
    
    def forward(self, x):
        # x: [batch_size, num_patches, num_frames, frame_size]
        x = x.view(x.size(0), x.size(1), -1)  # Flatten to [batch_size, num_patches, num_frames*frame_size]
        x = self.projection(x)  # [batch_size, num_patches, embedding_dim]
        return x

class ChronosActivityClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(ChronosActivityClassifier, self).__init__()
        self.chronos = T5Model.from_pretrained("amazon/chronos-t5-base").to(device)
        self.patch_embedding_act = PatchEmbedding(window*frames_per_second*frame_size, embedding_dim)
        self.patch_embedding_acw = PatchEmbedding(window*frames_per_second*frame_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim*2, num_classes)  # Early fusion doubles embedding_dim
        self.embedding_dim = embedding_dim
    
    def forward(self, act_patches, acw_patches, act_times, acw_times):
        # act_patches, acw_patches: [batch_size, num_patches, num_frames, frame_size]
        # act_times, acw_times: [batch_size, num_patches]
        
        # Convert to tensors
        act_patches = torch.tensor(act_patches, dtype=torch.float32).to(device)
        acw_patches = torch.tensor(acw_patches, dtype=torch.float32).to(device)
        
        # Get patch embeddings
        act_embeddings = self.patch_embedding_act(act_patches)  # [batch_size, num_patches, embedding_dim]
        acw_embeddings = self.patch_embedding_acw(acw_patches)  # [batch_size, num_patches, embedding_dim]
        
        # Add positional encodings
        act_pe = get_sinusoidal_positional_encoding(act_embeddings.size(1), self.embedding_dim, act_times).to(device)
        acw_pe = get_sinusoidal_positional_encoding(acw_embeddings.size(1), self.embedding_dim, acw_times).to(device)
        act_embeddings = act_embeddings + act_pe.unsqueeze(0)  # [batch_size, num_patches, embedding_dim]
        acw_embeddings = acw_embeddings + acw_pe.unsqueeze(0)  # [batch_size, num_patches, embedding_dim]
        
        # Early fusion: concatenate act and acw embeddings
        fused_embeddings = torch.cat([act_embeddings, acw_embeddings], dim=-1)  # [batch_size, num_patches, embedding_dim*2]
        
        # Process with Chronos-T5-Base
        outputs = self.chronos(inputs_embeds=fused_embeddings).last_hidden_state  # [batch_size, num_patches, hidden_dim]
        pooled_output = outputs.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        return logits

def _run_(_train_features, _train_labels, _train_times, _test_features, _test_labels, _test_times):
    _train_features = np.array(_train_features)
    _test_features = np.array(_test_features)
    _train_labels = np.array(_train_labels)
    _test_labels = np.array(_test_labels)
    
    # Separate act and acw
    _train_features_act = _train_features[:, 0]  # [batch_size, num_patches, num_frames, frame_size]
    _train_features_acw = _train_features[:, 1]
    _test_features_act = _test_features[:, 0]
    _test_features_acw = _test_features[:, 1]
    
    # Convert labels to tensor
    _train_labels = torch.tensor(_train_labels, dtype=torch.float32).to(device)
    _test_labels = torch.tensor(_test_labels, dtype=torch.float32).to(device)
    
    # Initialize model
    model = ChronosActivityClassifier(num_classes=len(activity_list), embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(_train_features_act, _train_features_acw, _train_times, _train_times)  # Same times for act and acw
        loss = criterion(outputs, _train_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/30, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        _predict_logits = model(_test_features_act, _test_features_acw, _test_times, _test_times)
        _predict_labels = torch.argmax(_predict_logits, dim=1).cpu().numpy()
        _test_labels_np = np.argmax(_test_labels.cpu().numpy(), axis=1)
        
        f_score = metrics.f1_score(_test_labels_np, _predict_labels, average='macro')
        accuracy = metrics.accuracy_score(_test_labels_np, _predict_labels)
        results = 'chronos_ac_2m_transformer' + ',' + str(fusion) + ',' + str(sys.argv[2]) + ',' + str(accuracy) + ',' + str(f_score)
        print(results)
        write_data(results_file, str(results))

# Main execution
act_data = read(act_path, '_act')
acw_data = read(acw_path, '_acw')

all_features = extract_features(act_data, acw_data)
all_features = pad_features(all_features)
all_features = frame_reduce(all_features)
all_users = list(all_features.keys())

i = sys.argv[2]
train_features, test_features = train_test_split(all_features, [i])

train_features, train_labels, train_times = flatten(train_features)
test_features, test_labels, test_times = flatten(test_features)

from keras.utils import np_utils
train_labels = np_utils.to_categorical(train_labels, len(activity_list))
test_labels = np_utils.to_categorical(test_labels, len(activity_list))

_run_(train_features, train_labels, train_times, test_features, test_labels, test_times)
