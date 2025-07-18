# Standard Library
import os

# Import from alt
import requests
import time
from loguru import logger

tokens = {'SFQ': "fffc06d2cd0a4ab9b9803ee7042bb2f8",
          "QL": "91d5d3660d634c6e88d21b0923e88cce",
          "GCB": "a5cfea2dc58142f19235411dbc549cdb"}


class BeeHelper:
    """
    docs: https://i1gmfjq19i.feishu.cn/docx/IFOUd4eLvogpC8xkHTucOgeJnzf
    """

    URL = "https://bee.sensetime.com/api/task/get"
    RESULT_URL = "https://bee.sensetime.com/api/task/requestExportResult"

    GET_TESULT_URL = "https://bee.sensetime.com/api/task/getExportResult"

    ATTRIBUTES = [
        "id",
        "name",
        "projectID",
        "projectName",
        "creator",
        "creatorUsername",
        "client",
        "tenant",
        "phone",
        "fileNumber",
        "trialFileNumber",
        "result",
        "expectedAt",
        "documentation",
        "documentationName",
        "storageType",
        "storageCluster",
        "taskStepList",
        "preStorageType",
        "preStorageCluster",
        "preAnnotationPath",
        "existsTaskID",
        "existsTaskStep",
        "dataSource",
        "preAnnotation",
        "operator",
        "packageMethod",
        "dataType",
        "passRate",
        "packageSize",
        "preAnnotationTool",
        "correction",
        "datalistID",
        "datalistName",
        "tags",
        "v2TagList",
        "customV2TagList",
        "v2CustomTagList",
        "preResultType",
        "datalistResultID",
        "datalistResultName",
        "datasetID",
        "datasetName",
        "auditStatus",
        "round",
        "isAcceptance",
        "reworkType",
        "rejectReason",
        "auditedNumber",
        "auditFailedNumber",
        "acceptedNumber",
        "acceptanceFailedNumber",
        "taskBillID",
        "filePathListObjectKey",
        "createdAt",
        "toolType",
        "toolID",
        "toolName",
        "toolConfig",
        "budget",
        "unitType",
        "unitDescName",
        "unitDescObjectKey",
        "shareInfo",
        "fileOrder",
        "resultSuffix",
        "resultSuffixArray",
        "lastUpload",
        "skipAudit",
        "customRate",
        "lastTaskStepUpdaterID",
        "lastTaskStepUpdaterName",
        "lastTaskStepUpdaterLoginType",
        "isTrialTask",
        "trialType",
        "supportOperator",
        "supportOperatorID",
        "authBatchID",
        "authInfoTotal",
        "authInfoList",
        "authIDList",
        "authWay",
        "relationDatasetID",
        "relationDatasetName",
        "relationDatalistID",
        "relationDatalistName",
        "relationDatasetType",
        "packageCount",
        "supportOperatorSupplier",
    ]

    STATUS_MAPPING = {
        1: "文件传输中",
        2: "运营审核中",
        3: "试标注",
        4: "试标注验收",
        5: "标注中",
        6: "验收中",
        8: "任务终止",
        9: "已完成",
        11: "任务暂停",
        12: "返工确认中",
        13: "重新编辑",
    }

    def __init__(self, sensebee_id=22956, token="91d5d3660d634c6e88d21b0923e88cce") -> None:
        if isinstance(sensebee_id, str):
            sensebee_id = int(sensebee_id)

        self.headers = {"Content-Type": "application/json", "sensebee-api-token": token}

        self.sensebee_id = sensebee_id
        response = requests.get(self.URL, headers=self.headers, params={"taskID": sensebee_id})

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        self.data = response.json()

        if self.data.get("id") != sensebee_id:
            raise Exception(f"Task ID mismatch: expected {sensebee_id}, got {self.data.get('id')}")

    def __getattr__(self, item):
        if item in self.ATTRIBUTES:
            return self.data.get(item)
        raise AttributeError(f"'BeeHelper' object has no attribute '{item}'")

    @property
    def root(self):
        return self.data.get("storagePath")

    @property
    def status(self):
        cur_status = self.data.get("status")
        return self.STATUS_MAPPING[cur_status]

    def get_result(self, save_path="./", skip_exist=True):
        response = requests.post(self.RESULT_URL, headers=self.headers, json={"taskID": self.sensebee_id})
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        result_res = response.json()

        filename = result_res["filename"]

        os.makedirs(save_path, exist_ok=True)
        save_zip_file = os.path.join(save_path, filename.split("/")[-1])

        if os.path.exists(save_zip_file):
            if skip_exist:
                logger.info("{} exist, skip!".format(save_zip_file))
                return save_zip_file
            else:
                os.remove(save_zip_file)

        if "url" not in result_res:  # 还在标注中的状态
            cnt = 600
            while cnt > 0:
                response = requests.get(self.GET_TESULT_URL, headers=self.headers, params={'taskID': self.sensebee_id, 'filename': filename})
                if response.json()['status'] == 2:
                    result_res["url"] = response.json()['url']
                    break
                elif response.json()['status'] == 1:
                    break
                cnt -= 1
                time.sleep(2)
                logger.info("waiting", self.sensebee_id)

        url_resp = requests.get(result_res["url"])
        assert url_resp.status_code == 200

        with open(save_zip_file, "wb") as result_zip:
            result_zip.write(url_resp.content)
        assert os.path.exists(save_zip_file)

        return save_zip_file


class BeeUploader:
    COPY_TASK_URL = "https://bee.sensetime.com/api/task/copyTask"
    UPLOAD_URL = "https://bee.sensetime.com/api/oss/uploadUrl"
    token = tokens["SFQ"]

    headers = {"Content-Type": "application/json", "sensebee-api-token": token}

    PVB_TEMPLATE_TASK_ID = 22547
    GOP_TEMPLATE_TASK_ID = 24076

    @classmethod
    def get_cluster(cls, root):
        if "s3://sdc_gt_label" in root:
            return "sdc-iag-pilot"
        elif "s3://sdc3_gt_label" in root:
            return "sdc3-iag-pilot"
        elif "s3://sdc3-gt-label-2" in root or "s3://sdc3-adas-3" in root:
            return "auto-oss"
        else:
            raise NotImplementedError(root)

    @classmethod
    def get_aksk(cls, cluster):
        return {
            "sdc-iag-pilot": {"accessID": "O5B34FE51MMZRZGEY1Z5", "accessSecret": "Xh6hOX5xnZ6BrsNnCcCt8NzNCbWeCSYwtABErKJ6"},
            "sdc3-iag-pilot": {"accessID": "45DJ69SBDGDFITJT9PCD", "accessSecret": "oiWI7NP3O3g1UOkEzuzJSZLWkBJpv2a5mcuShCRd"},
            "auto-oss": {"accessID": "45DJ69SBDGDFITJT9PCD", "accessSecret": "oiWI7NP3O3g1UOkEzuzJSZLWkBJpv2a5mcuShCRd"},
        }[cluster]

    @classmethod
    def upload_file(cls, upload_url, file_path):
        assert os.path.exists(file_path)
        with open(file_path, "rb") as file:
            response = requests.put(upload_url, data=file)

        if response.status_code == 200 or response.status_code == 201:
            logger.info("File uploaded successfully")
        else:
            logger.error(f"Failed to upload file. Status code: {response.status_code}")
            logger.error(response.text)

    @classmethod
    def upload_gop(cls, task_name, root, label_json, file_txt, pre_sensebee=None):
        cls.headers['sensebee-api-token'] = tokens["GCB"]
        task_id = cls.upload_task(cls.GOP_TEMPLATE_TASK_ID, task_name, root, label_json, file_txt, pre_sensebee)
        return task_id

    @classmethod
    def upload_pvb(cls, task_name, root, label_json, file_txt, pre_sensebee=None):
        cls.headers['sensebee-api-token'] = tokens["SFQ"]
        task_id = cls.upload_task(cls.PVB_TEMPLATE_TASK_ID, task_name, root, label_json, file_txt, pre_sensebee)
        return task_id

    @classmethod
    def upload_task(cls, template_task, task_name, root, label_json, file_txt, pre_sensebee=None):
        url_response = requests.get(cls.UPLOAD_URL, params={"suffix": "jsonl", "temp": False})
        label_json_objectKey, label_json_url = url_response.json()["objectKey"], url_response.json()["url"]
        cls.upload_file(label_json_url, label_json)

        url_response = requests.get(cls.UPLOAD_URL, params={"suffix": "tmp.txt", "temp": False})
        label_txt_objectKey, label_txt_url = url_response.json()["objectKey"], url_response.json()["url"]
        cls.upload_file(label_txt_url, file_txt)

        body = {
            "taskID": template_task,
            "name": task_name,
            "storageType": 2,  # 2 是ceph
            "storageCluster": cls.get_cluster(root),
            "akSkConfig": cls.get_aksk(cls.get_cluster(root)),
            "storagePath": root,
            "filePathListObjectKey": label_txt_objectKey,
            "unitDescObjectKey": label_json_objectKey,
            "unitDescName": label_json.split('/')[-1],
            "unitType": 2,
            "preAnnotation": 2,
        }

        if pre_sensebee:
            body.update(
                {
                    "preAnnotation": 1,
                    "preStorageType": 2,
                    "preStorageCluster": cls.get_cluster(root),
                    "preAnnotationPath": pre_sensebee,
                    "preakSkConfig": cls.get_aksk(cls.get_cluster(root)),
                }
            )

        response = requests.post(cls.COPY_TASK_URL, headers=cls.headers, json=body)
        assert response.status_code == 200

        return response.json()['taskID']


if __name__ == "__main__":
    bee = BeeHelper(sensebee_id=22956)

    print(bee.root)
    print(bee.status)

    zip_file = bee.get_result()
