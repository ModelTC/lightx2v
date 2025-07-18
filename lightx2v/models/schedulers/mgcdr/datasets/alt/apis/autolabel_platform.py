# Standard Library
import os

# Import from third library
import boto3
import json

# Import from alt
import pymysql.cursors
from loguru import logger


class AutolabelerPlatformHelper(object):
    AWS_ENDPOINT_URL = "http://sdc-oss.iagproxy.senseauto.com"
    AWS_ACCESS_KEY_ID = "45DJ69SBDGDFITJT9PCD"
    AWS_SECRET_ACCESS_KEY = "oiWI7NP3O3g1UOkEzuzJSZLWkBJpv2a5mcuShCRd"

    # prod
    DB_HOST = "lg.paas.sensetime.com"
    DB_PORT = 37970
    DB_USER = "senseauto"
    DB_PASSWORD = "2022Senseauto"
    DB_NAME = "auto_runner"

    @classmethod
    def get_result(cls, task_id, job_source_type=1):
        task_name = task_id

        s3 = boto3.client(
            "s3",
            endpoint_url=cls.AWS_ENDPOINT_URL,
            aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY,
        )
        sdc3 = boto3.client(
            "s3",
            endpoint_url="http://auto-business.st-sh-01.sensecoreapi-oss.cn",
            aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY,
        )

        tasks = []
        with pymysql.connect(
            host=cls.DB_HOST,
            port=cls.DB_PORT,
            user=cls.DB_USER,
            password=cls.DB_PASSWORD,
            database=cls.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "select * from task_info ti where job_id = %s and status = 2 and job_source_type = %s",
                    (task_name, job_source_type),
                )
                tasks = cursor.fetchall()

        logger.info(f"{task_name}: len(tasks)={len(tasks)}")

        # download task meta and get meta result
        results = []
        for task in tasks:
            meta_path = task["meta_path"]
            if meta_path.startswith("s3://"):
                meta_path = meta_path[5:]
            bucket, key = meta_path.split("/", 1)
            if bucket.startswith("sdc3-"):
                obj = sdc3.get_object(Bucket=bucket, Key=key)
            else:
                obj = s3.get_object(Bucket=bucket, Key=key)
            meta = json.loads(obj["Body"].read().decode("utf-8"))

            data_annotation = meta["data_annotation"]
            bucket_mapper = meta["bucket_mapper"]

            if data_annotation.startswith("s3://"):
                data_annotation = data_annotation[5:]

            bucket, key = data_annotation.split("/", 1)
            if bucket.startswith("{"):
                bucket = bucket[1:-1]
                bucket = bucket_mapper[bucket]["name"]
            results.append(
                {
                    "bucket": bucket,
                    "key": key,
                }
            )
        meta_jsons = []
        for r in results:
            cur_meta = "s3://{}/{}meta.json".format(r["bucket"], r["key"].rstrip("/").rstrip(".gt_labels"))
            meta_jsons.append(str(cur_meta))

        return meta_jsons


if __name__ == "__main__":
    res = AutolabelerPlatformHelper.get_result(task_id=10395)
