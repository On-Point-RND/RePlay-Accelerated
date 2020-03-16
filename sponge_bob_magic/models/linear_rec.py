"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from typing import Dict, Optional

from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit, udf, when
from pyspark.sql.types import DoubleType, FloatType

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import func_get, get_feature_cols, get_top_k_recs


class LinearRec(Recommender):
    """ Рекомендатель на основе линейной модели и эмбеддингов. """
    _model: LogisticRegressionModel
    augmented_data: DataFrame

    def __init__(
            self,
            lambda_param: float = 0.0,
            elastic_net_param: float = 0.0,
            num_iter: int = 100):
        self.lambda_param: float = lambda_param
        self.elastic_net_param: float = elastic_net_param
        self.num_iter: int = num_iter

    def get_params(self) -> Dict[str, object]:
        return {"lambda_param": self.lambda_param,
                "elastic_net_param": self.elastic_net_param,
                "num_iter": self.num_iter}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        # TODO: добавить проверку, что в логе есть только нули и единицы
        self.augmented_data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features")
        ).cache()

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        self._model = (
            LogisticRegression(
                maxIter=self.num_iter,
                regParam=self.lambda_param,
                elasticNetParam=self.elastic_net_param)
            .fit(self.augmented_data)
        )
        spark = SparkSession(log.rdd.context)
        model_path = os.path.join(spark.conf.get("spark.local.dir"),
                                  "linear.model")
        self._model.write().overwrite().save(model_path)
        self._model = self._model.read().load(model_path)

    @staticmethod
    def _augment_data(
            log: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame
    ) -> DataFrame:
        """
        Обогащает лог фичами пользователей и объектов.

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый спарк-датайрейм, в котором к каждой строчке лога
            добавлены фичи пользователя и объекта, которые в ней встречаются
        """
        user_feature_cols, item_feature_cols = get_feature_cols(
            user_features, item_features)

        return VectorAssembler(
            inputCols=user_feature_cols + item_feature_cols,
            outputCol="features"
        ).transform(
            log
            .join(user_features.drop("timestamp"), on="user_id", how="inner")
            .join(item_features.drop("timestamp"), on="item_id", how="inner")
        )

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: Optional[DataFrame] = None,
                 items: Optional[DataFrame] = None,
                 context: Optional[str] = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        data = (
            self._augment_data(
                users.crossJoin(items), user_features, item_features
            )
            .select("features", "item_id", "user_id")
        )

        if filter_seen_items:
            data = data.join(log, on=["user_id", "item_id"], how="left_anti")

        recs = (
            self._model
            .transform(data)
            .select(
                "user_id",
                "item_id",
                udf(func_get, DoubleType())("probability", lit(1))
                .alias("relevance")
                .cast(FloatType())
            )
        )

        recs = get_top_k_recs(recs, k)
        recs = recs.withColumn(
            "relevance",
            when(recs["relevance"] < 0, 0).otherwise(recs["relevance"])
        )
        return recs