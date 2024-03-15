from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

dataset_names_uci = [
    "Wine", "Contraceptive", "Pen-Based",
    "Dermatology", "Balance Scale",
    "Glass", "Heart (Cleveland)", "Car Evaluation",
    "Yeast", "Shuttle"
]

dataset_names_openML = [
    "New Thyroid", "FARS", 'Hayes-Roth', 'Vertebra Column', 'Page blocks'
]

ids_openML = [
    40682, 40672, 329, 1523, 30
]

ids_uci = [
    109, 30, 81,
    33, 12,
    42, 45, 19,
    110, 148
]

datasets_uci_dict = dict(zip(dataset_names_uci, ids_uci))
datasets_openML_dict = dict(zip(dataset_names_openML, ids_openML))


def download_data(dataset_name)->tuple:
    if dataset_name in datasets_uci_dict:
        dataset = fetch_ucirepo(id=datasets_uci_dict[dataset_name])
        X, y = dataset.data.features, dataset.data.targets

    elif dataset_name in datasets_openML_dict:
        dataset = fetch_openml(data_id=datasets_openML_dict[dataset_name], as_frame=True)
        X, y = dataset.data, dataset.target

    else:
        raise ValueError("Dataset not found in UCI or OpenML dictionaries.")
    return X, y