from src.preprocess import load_data, preprocess_data, final_dataset
from src.train import train_model

if __name__ == "__main__":
    data_dir = "data"
    X, Y = load_data(data_dir)
    dataset = preprocess_data(X)
    new_data, labels = final_dataset(dataset, Y, ch=[0, 1, 2])
    train_model(new_data, labels, output_path="models/cnn_model")
