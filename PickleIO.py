import pickle
def save_pkl(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
def load_pkl(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data