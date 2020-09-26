import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _dataframe_handler(df):
    return df.apply(lambda x: ' '.join([str(x_i) for x_i in x]), axis=1).values


class TfidfVectorizerDataFrames(TfidfVectorizer):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def fit(self, df, y=None):
        raw_documents = _dataframe_handler(df=df)
        return super().fit(raw_documents, y)

    def fit_transform(self, df, y=None):
        raw_documents = _dataframe_handler(df=df)
        return super().fit_transform(raw_documents, y)

    def transform(self, df, copy="deprecated"):
        raw_documents = _dataframe_handler(df=df)
        return super().transform(raw_documents, copy)

    def vectorize(self, df):
        raw_documents = _dataframe_handler(df=df)
        return np.array([self.transform([material_i]).data for material_i in raw_documents])
