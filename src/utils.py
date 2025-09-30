import numpy as np
from sklearn.metrics import precision_recall_curve

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def make_engine_from_env():

    load_dotenv()

    return create_engine(
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}",
        pool_pre_ping=True
    )


def threshold_for_min_precision(y_true, scores, min_prec=0.80):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    idx = np.where(prec[:-1] >= min_prec)[0]
    if len(idx):
        return float(thr[idx[0]])
    return float(thr[-1]) if len(thr) else 0.5
