from src.data_pipeline import main as build_data
from src.train_model import main as train_model
from src.build_report import main as build_report


if __name__ == "__main__":
    build_data()
    train_model()
    build_report()

