import numpy as np
import pandas as pd

from src.data_pipeline import haversine_km
from src.train_model import FEATURES


def test_haversine_zero_distance():
    distance = haversine_km(pd.Series([41.0]), pd.Series([29.0]), pd.Series([41.0]), pd.Series([29.0]))
    assert np.isclose(distance.iloc[0], 0.0)


def test_model_feature_list_excludes_outcomes():
    forbidden = {"is_late", "late_days", "delivery_days", "review_score", "order_delivered_customer_date"}
    assert forbidden.isdisjoint(FEATURES)

