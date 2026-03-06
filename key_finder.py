import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import os
import argparse
from glob import glob
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--predict-key", action="store_true")
parser.add_argument("--song")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test-batch", action="store_true")
parser.add_argument("--epochs", default=10)
parser.add_argument("--load-model")
parser.add_argument("--data-size", default=None)
parser.add_argument("--cqt", action="store_true")

args = parser.parse_args()

print(args.predict_key, args.train, args.epochs, args.load_model)

key_class = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

torch.cuda.empty_cache()

def load_dataset(root):
    audio_dir = os.path.join(root, "audio")
    data = []
    key_files = glob(os.path.join(root, "annotations", "key", "*.key"))

    if args.data_size != None:
        key_files = key_files[:int(args.data_size)]

    for key in key_files:
        name = os.path.splitext(os.path.basename(key))[0]
        with open(key, "r", encoding="utf-8") as f:
            key_label = f.read().strip()
        audio_path = os.path.join(audio_dir, name + ".wav")
        data.append({
            "path": audio_path,
            "key": key_label
        })

    X_list = []
    y = []

    for index, track in enumerate(data):
        if not args.cqt:
            X_list.append(logmel(track["path"], n_fft=8192, n_mels=105, hop_length=8820))
        else:
            X_list.append(cqt_log_spec(track["path"]))
        y.append(key_decode(track["key"]))
        print(f"Loaded Training Data {index+1}/{len(data)}")
    X = np.stack(X_list)
    return X, y

def load_testset(root):
    audio_dir = os.path.join(root, "audio")
    data = []
    key_files = glob(os.path.join(root, "annotations", "key", "*.key"))

    if args.data_size != None:
        key_files = key_files[:int(args.data_size)]

    for key in key_files:
        name = os.path.splitext(os.path.basename(key))[0]
        with open(key, "r", encoding="utf-8") as f:
            key_label = f.read().strip()
        audio_path = os.path.join(audio_dir, name + ".wav")
        data.append({
            "path": audio_path,
            "key": key_label
        })

    X_list = []
    y = []

    for index, track in enumerate(data):
        if not args.cqt:
            X_list.append(logmel(track["path"], n_fft=8192, n_mels=105, hop_length=8820))
        else:
            X_list.append(cqt_log_spec(track["path"]))
        y.append(key_decode(track["key"]))
        print(f"Loaded Training Data {index+1}/{len(data)}")
    X = np.stack(X_list)
    return X, y

def key_decode(giantsteps_key):
    tonic, mode = giantsteps_key.strip().split(" ")

    if mode.lower() == "major":
        return key_class.index(tonic)
    if mode.lower() == "minor":
        return key_class.index(tonic) + 12 # maps keys to 24 classes
    
def key_return(index):
    if index > 11:
        return f"{key_class[index-12]} minor"
    return f"{key_class[index]} major"

def decode_key_rel(k):
    if k < 12:
        return k, "major"
    return (k - 12), "minor"

def is_relative(pred, actual):
    pred_i, pred_mode = decode_key_rel(pred)
    actual_i, actual_mode = decode_key_rel(actual)

    if pred_mode == actual_mode:
        return False
    elif actual_mode == "major":
        rel_minor = (actual_i - 3) % 12
        return pred_mode == "minor" and pred_i == rel_minor
    else:
        rel_major = (actual_i + 3) % 12
        return pred_mode == "major" and pred_i == rel_major

def cqt_log_spec(file_path, target_length=600):
    y, sr = librosa.load(file_path)
    C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=24, n_bins=120, hop_length=sr//5))
    spec = np.log1p(C)

    F, T = spec.shape
    if T < target_length:
        spec = np.pad(spec, ((0,0),(0,target_length-T)))
    else:
        start = (T - target_length)//2
        spec = spec[:, start:start+target_length]

    return spec

def logmel(file_path, n_fft, n_mels, hop_length, target_length=600, plot=False):
    y, sr = librosa.load(file_path, sr=44100)
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    spec_db = librosa.power_to_db(spectogram, ref=np.max)

    length = spec_db.shape[1]

    if length < target_length:
        pad = target_length - length
        spec_db = np.pad(spec_db, ((0, 0), (0, pad)))
    elif length > target_length:
        start = np.random.randint(0, length-target_length)
        spec_db = spec_db[:, start:start+target_length]

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    return spec_db

# ----------- CNN Model -----------

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        if args.cqt:
            self.bins = 120
        else:
            self.bins = 105

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5 = nn.Conv2d(8, 8, 5, padding=2)

        self.dropout2d = nn.Dropout2d(0.2)

        self.freq_dense = nn.Linear(8*self.bins, 48)
        self.classifier = nn.Linear(48, 24)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        x = self.dropout2d(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = F.elu(self.freq_dense(x))
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
    
# ----------- Training Loop -----------
    
def train_model(model, dataloader_train, optimizer, criterion, device,
                epochs=20, grad_clip=None):

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader_train:
            xb = xb.to(device, non_blocking=True)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            yb = yb.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # ---- stats ----
            bs = xb.size(0)
            running_loss += loss.item() * bs
            total += bs

        train_loss = running_loss / total

        print(f"Epoch {epoch:3d} | Trianing Loss {train_loss:.4f}")

# ----------- Classifier -----------

model = FeatureExtractor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(args.predict_key):
    if not args.test_batch and not os.path.exists(args.song):
        print(f"Could not find song at file path {args.song}")
        sys.exit()
    elif not os.path.exists(args.load_model):
        print("Could not find existing model, please train model first")
        sys.exit()
    model.load_state_dict(torch.load(args.load_model))
    model.to(device)
    if args.test_batch:
        x, y = load_testset("./giantsteps-key-dataset")
        x_tensor = torch.tensor(x).unsqueeze(1).to(device)
    else:
        if args.cqt:
            spectrogram = cqt_log_spec(args.song)
        else:
            spectrogram = logmel(args.song, n_fft=8192, n_mels=105, hop_length=8820)
        x_tensor = torch.tensor(spectrogram)[None, None, :, :].to(device)
    with torch.no_grad():
        if args.test_batch:
            correct = 0
            total = 0
            relative = 0
            preds = model(x_tensor).argmax(dim=1)
            for pred, yb in zip(preds, y):
                print(f"Predicted Key: {key_return(pred)} | Actual Key: {key_return(yb)}")
                if pred == yb:
                    correct += 1
                elif is_relative(pred, yb):
                    correct += 1
                    relative += 1
                total += 1
            print(f"{correct}/{total}")
            print(f"Relative Majors/Minors predicted instead: {relative}")
            print(f"{(correct/total*100):.1f}% Accuracy")
        else:
            pred = model(x_tensor).argmax(dim=1)
            print(f"Predicted Key: {key_return(pred)}")
else:
    print("Loading Dataset")
    # X, y = load_dataset("./giantsteps-key-dataset")
    X, y = load_dataset("./augmented-data")

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42, stratify=y)

    X_mean = X_train.mean()
    X_std  = X_train.std()

    print(f"X Shape: {X.shape}")
    print(f"y Shape: {len(y)}")

    X_train = (X_train - X_mean) / (X_std + 1e-6)
    X_test  = (X_test  - X_mean) / (X_std + 1e-6)

    print(f"Train size: {X_train.shape}")
    print("X min/max:", X_train.min(), X_train.max())
    print("X mean/std:", X_train.mean(), X_train.std())

    print(f"Test size: {X_test.shape}")

    counts = np.bincount(y_train, minlength=24)
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


    X_tensor_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_tensor_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    y_tensor_train = torch.tensor(y_train, dtype=torch.long)
    y_tensor_test = torch.tensor(y_test, dtype=torch.long)

    dataset_train = TensorDataset(X_tensor_train, y_tensor_train)
    dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler)

    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    if(args.train):
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)

        train_model(model, dataloader_train, optimizer, criterion, device,
                    epochs=int(args.epochs), grad_clip=1.0)

        torch.save(model.state_dict(), f"models/key_model_{args.epochs}_epochs.pt")
    elif not os.path.exists(args.load_model):
        print("Could not find existing model, please train model first")
        sys.exit()
    else:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model.to(device)

    # test training
    model.eval()
    correct = 0
    total = 0

    guesses = []
    actual = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader_test:

            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device).long()

            pred_batch = model(x_batch).argmax(dim=1)

            guesses.extend(pred_batch.cpu().tolist())
            actual.extend(y_batch.cpu().tolist())

            correct += (pred_batch == y_batch).sum().item()
            total += y_batch.size(0)

    print(guesses)
    print(actual)

    accuracy = correct / total
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Test Accuracy: {accuracy}")

