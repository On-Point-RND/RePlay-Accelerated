"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.als_rec import ALSRec


class ALSRecTestCase(PySparkTest):
    def setUp(self):
        self.als_rec = ALSRec(1)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i1", self.some_date, 1.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u2", "i3", self.some_date, 2.0],
                ["u3", "i4", self.some_date, 2.0],
                ["u1", "i4", self.some_date, 2.0]
            ],
            schema=LOG_SCHEMA
        )
        self.als_rec._seed = 42

    def test_fit(self):
        self.als_rec.fit(self.log)
        item_factors = np.array(
            self.als_rec.model.itemFactors
            .toPandas()["features"].tolist()
        )
        self.assertEqual(item_factors.shape, (3, 1))

    def test_predict(self):
        recs = self.als_rec.fit_predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            user_features=None,
            item_features=None
        )
        # проверяем только формат ответа
        self.assertEqual(recs.schema, REC_SCHEMA)

    def test_get_params(self):
        self.assertEqual(self.als_rec.get_params(), {"rank": 1})