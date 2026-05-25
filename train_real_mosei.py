cd C:\Users\student\Desktop\sentiment_Analysis
.venv\Scripts\python.exe -m scene_motion_llm.main_train --dataset-type ucf101 --dataset-path scene_motion_llm/dataset/UCF101 --batch-size 4 --epochs 10import h5py
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path('dataset') / 'CMU-MOSEI'

FEATURE_FILES = {
    'facet': (ROOT / 'visuals' / 'CMU_MOSEI_VisualFacet42.csd', 'FACET 4.2'),
    'openface': (ROOT / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd', 'OpenFace_2'),
    'audio': (ROOT / 'acoustics' / 'CMU_MOSEI_COVAREP.csd', 'COVAREP')
}
LABEL_FILE = ROOT / 'labels' / 'CMU_MOSEI_Labels.csd'

CLASS_NAMES = ['Positive', 'Neutral', 'Negative']


def load_labels(labels_path):
    with h5py.File(labels_path, 'r') as f:
        root = f.get('All Labels') or f
        data_group = root['data']
        video_ids = sorted(list(data_group.keys()))
        labels = []
        for vid in video_ids:
            features = data_group[vid]['features'][()]
            score = float(features[0][0])
            if score > 0.5:
                labels.append(0)
            elif score < -0.5:
                labels.append(2)
            else:
                labels.append(1)
    return video_ids, np.array(labels, dtype=np.int64)


def load_feature_modalities(video_ids):
    modality_data = {}
    requested_ids = list(video_ids)
    available_ids = set(requested_ids)

    # Determine the common set of video IDs present in all available modalities.
    for name, (path, rootkey) in FEATURE_FILES.items():
        if not path.exists():
            print(f' WARNING: modality file missing: {path}')
            continue
        with h5py.File(path, 'r') as f:
            g = f[rootkey]
            data = g['data']
            ids = set(data.keys())
            available_ids &= ids

    if not available_ids:
        raise ValueError('No common video IDs found across the selected modality files.')

    if len(available_ids) < len(requested_ids):
        missing_ids = sorted(set(requested_ids) - available_ids)
        print(f' WARNING: {len(missing_ids)} video IDs are missing from one or more modalities.')
        print(f' Using {len(available_ids)} common video IDs for training / evaluation.')

    common_video_ids = [vid for vid in requested_ids if vid in available_ids]

    for name, (path, rootkey) in FEATURE_FILES.items():
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            g = f[rootkey]
            data = g['data']
            X = []
            invalid_entries = 0
            total_entries = 0
            for vid in common_video_ids:
                vid_group = data[vid]
                feats = vid_group['features'][()]
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                total_entries += feats.size
                invalid_entries += np.count_nonzero(np.isnan(feats))
                invalid_entries += np.count_nonzero(np.isinf(feats))
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                mu = np.nanmean(feats, axis=0)
                sigma = np.nanstd(feats, axis=0)
                X.append(np.concatenate([mu, sigma], axis=0))
        modality_data[name] = np.stack(X, axis=0)
        if invalid_entries > 0:
            print(f' WARNING: sanitized {invalid_entries} invalid values in {name} modality')
        print(f' loaded modality {name}: shape {modality_data[name].shape}')

    return modality_data, common_video_ids


def build_dataset(video_ids, modality_data):
    arrays = [modality_data[m] for m in sorted(modality_data.keys())]
    X = np.concatenate(arrays, axis=1)
    return X


def print_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('Label distribution:')
    for u, c in zip(unique, counts):
        print(f'  {CLASS_NAMES[u]:>8}: {c}')


def train_and_eval(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
    )
    print(f' splits: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print('\nTraining logistic regression...')
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('\nLogistic Regression results:')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 macro:', f1_score(y_test, y_pred, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    print('\nTraining small MLP classifier...')
    mlp = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    print('\nMLP results:')
    print(classification_report(y_test, y_pred_mlp, target_names=CLASS_NAMES, digits=4))
    print('Accuracy:', accuracy_score(y_test, y_pred_mlp))
    print('F1 macro:', f1_score(y_test, y_pred_mlp, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_mlp))

    return {
        'lr': {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        },
        'mlp': {
            'accuracy': accuracy_score(y_test, y_pred_mlp),
            'f1_macro': f1_score(y_test, y_pred_mlp, average='macro')
        }
    }


if __name__ == '__main__':
    print('Loading labels from', LABEL_FILE)
    video_ids, labels = load_labels(LABEL_FILE)
    print('Total samples:', len(video_ids))
    print_distribution(labels)
    modality_data, common_video_ids = load_feature_modalities(video_ids)

    if len(common_video_ids) < len(video_ids):
        label_map = {vid: lbl for vid, lbl in zip(video_ids, labels)}
        labels = np.array([label_map[vid] for vid in common_video_ids], dtype=np.int64)
        video_ids = common_video_ids
        print('Adjusted dataset size after modality alignment:', len(video_ids))
        print_distribution(labels)

    X = build_dataset(video_ids, modality_data)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print('Combined feature shape:', X.shape)
    results = train_and_eval(X, labels)
    print('\nFinal metrics:')
    for model_name, metrics in results.items():
        print(f' {model_name}: acc={metrics["accuracy"]:.4f}, f1_macro={metrics["f1_macro"]:.4f}')
