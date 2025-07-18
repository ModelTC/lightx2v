# Standard Library
from dataclasses import dataclass

# Import from third library
import numpy as np

# Import from alt
from alt.utils.token_helper import generate_token
from loguru import logger


class LabelMapper:
    LABELS = {
        "": None,  # 异常标签
        # PVB
        "小型车辆": "VEHICLE_CAR",
        "运动型多用途轿车": "VEHICLE_SUV",
        "大型货车": "VEHICLE_TRUCK",
        "皮卡": "VEHICLE_PICKUP",
        "公交车": "VEHICLE_BUS",
        "三轮车": "VEHICLE_TRIKE",
        "异型车": "VEHICLE_SPECIAL",
        "警车": "VEHICLE_POLICE",
        "消防车": "VEHICLE_FIRE_TRUCK",
        "救护车": "VEHICLE_AMBULANCE",
        "垃圾车": "VEHICLE_RUBBISH",
        "畜牧车": "VEHICLE_LIVESTOCK",
        "多段车": "VEHICLE_MULTI_STAGE",
        "行人": "PEDESTRIAN_NORMAL",
        "普通行人": "PEDESTRIAN_NORMAL",
        "交警": "PEDESTRIAN_TRAFFIC_POLICE",
        "自行车骑手": "CYCLIST_BICYCLE",
        "机动车骑手": "CYCLIST_MOTOR",
        "机动车骑行者": "CYCLIST_MOTOR",
        "其他骑手": "CYCLIST_OTHERS",
        # GOP
        "锥形桶": "CONE",
        "隔离桶": "ISOLATION_BARRER",
        "隔离柱": "POLE",
        "隔离石墩": "STONE POLE",
        "水马": "BARRIER",
        "三角标": "TRIANGLE_WARNING",
        "施工标志牌": "CONSTRUCTION_SIGN",
        "地锁": "PARKING_LOCK",
        "道闸闸杆": "BARRIER_GATE",
        "打开的道闸闸杆": "BARRIER_GATE",
        "关闭的道闸闸杆": "BARRIER_GATE",
        "闭合的道闸闸杆": "BARRIER_GATE",
        "开启中的道闸闸杆": "BARRIER_GATE",
        "推车": "CART",
        "动物": "ANIMAL",
        "开启地锁": "PARKING_LOCK",
        "关闭地锁": "PARKING_LOCK",
        "锥桶": "CONE",
        # GAC-GOP
        "临时路栏": "TEMPORARY BARRICADE",
        "永久路栏": "PERMANENT BARRICADE",
        "其他障碍物": "OBSTACLES",
        "地面标志牌": "GROUND SIGN",  # 和施工标志其实是存在歧义的
        "伸缩门": "RETRACTABLE DOOR",
        "减速带": "SPEED BUMP",
        "限位器": "PARKING LIMITER",
        "柱子": "COLUMN",  # 只标停车场的柱子
        "充电桩": "CHARGING PILE",
    }

    E2C_LABELS = {
        "PEDESTRIAN": "行人",
        "BIKE_BICYCLE": "自行车骑手",
        "VEHICLE": "小型车辆",
        "BIKE_BIKEBIG": "自行车骑手",
        "BIKE": "自行车骑手",
    }
    for key, value in LABELS.items():
        if key in E2C_LABELS:
            continue
        E2C_LABELS[value] = key

    @classmethod
    def to_pap(cls, label):
        if label in ["VEHICLE_FIRE_TRUCK", "VEHICLE_AMBULANCE", "VEHICLE_RUBBISH"]:
            return "VEHICLE_TRUCK"
        elif label in ["VEHICLE_LIVESTOCK"]:
            return "VEHICLE_SUV"
        elif label in ["CYCLIST_OTHERS"]:
            return "CYCLIST_BICYCLE"
        elif label in ["GROUND SIGN", "CONSTRUCTION_SIGN"]:
            return "CONSTRUCTION_SIGN"
        return label

    @classmethod
    def to_gac(cls, label):
        if label in ["VEHICLE_MULTI_STAGE"]:
            # 暂不确定是否需要
            pass
        return label


@dataclass
class Target2D:
    def __post_init__(self):
        self.token = generate_token()

    @classmethod
    def build_from_sensesee(cls, target_meta, mode="pap"):
        target = Target2D()
        target.filename = target_meta["imageName"]

        # assert target.filename != ''
        target.left = target_meta["x"]
        target.top = target_meta["y"]
        target.width = target_meta["width"]
        target.height = target_meta["height"]

        target.visible = True  # 标注默认为可见
        target.score = 1.0  # 标注默认为1.0
        target.gt_quality = 1.0  # 后面补充，暂时设置为None

        if "attribute" in target_meta:
            if target_meta["attribute"] == "":
                target.label = None
                return target
            label = LabelMapper.LABELS[target_meta["attribute"]]
            target.label = getattr(LabelMapper, f"to_{mode}")(label)

        return target

    @property
    def xyxy(self):
        return [self.left, self.top, self.left + self.width, self.top + self.height]

    @property
    def score2d(self):
        return self.score

    @property
    def valid(self):
        if hasattr(self, "label") and self.label is None:
            return False
        if self.width < 5 or self.height < 5:
            return False
        if self.filename == "":
            return False
        return True

    @property
    def pap_format(self):
        return {
            "token": self.token,
            "visible": self.visible,
            "score2d": self.score2d,
            "bbox2d": self.xyxy,
            "gt_quality": self.gt_quality,
        }

    @property
    def generate_format(self):
        return {
            "token": self.token,
            "box2d": [self.left, self.top, self.width, self.height],
            "format": "xywh",
            "filename": self.filename,
            "camera_name": self.filename.split("/")[-2].split("#")[0],
        }

    def reset_info2d(self, target_2d):
        box_2d, score, token, camera_name, filename = target_2d

        assert self.token == token
        self.filename = filename
        self.left, self.top, self.width, self.height = box_2d
        self.score = score
        # self.camera_name = camera_name

    @property
    def camera_name(self):
        try:
            camera_name = self.filename.split("/")[-2].split("#")[0]
        except Exception as e:  # noqa
            camera_name = ""
        return camera_name


@dataclass
class Target3D:
    def __post_init__(self):
        self.token = generate_token()

    @classmethod
    def build_from_sensenee(cls, target_meta, mode="pap", task="pvb"):
        target = Target3D()
        target.task = task

        label = LabelMapper.LABELS[target_meta["attribute"]]
        target.label = getattr(LabelMapper, f"to_{mode}")(label)
        target.location = [target_meta["center"]["x"], target_meta["center"]["y"], target_meta["center"]["z"]]

        target.num_pts = target_meta.get("count", None)
        target.length, target.width, target.height = target_meta["width"], target_meta["height"], target_meta["depth"]
        target.yaw = target_meta["rotation"]

        target.attribute = cls.build_attributes(
            target_meta.get("subAttribute", {}), task=task, label=target_meta["attribute"], location=target.location
        )
        target.score = 1.0  # sensebee 标注默认分数为0

        target.info2d = []

        caches = []
        for rect in target_meta.get("rects", []):
            try:
                cur_target_2d = Target2D.build_from_sensesee(rect)
            except Exception as e:
                continue
            cur_target_2d.token = target.token

            camera_name = cur_target_2d.camera_name

            if camera_name not in caches:  # 工具BUG
                caches.append(camera_name)
            else:
                continue

            if cur_target_2d.valid:
                target.info2d.append(cur_target_2d)
                assert cur_target_2d.filename != ""

        target.velocity = [0, 0, 0]
        target.track_id = None

        return target

    @classmethod
    def build_attributes(cls, attributes, task="pvb", label=None, location=None):
        if task in ["pvb"]:
            # 自车道归属的补丁
            if location[1] > 5 or location[1] < -5:
                if attributes["自车道归属"] == "是":
                    attributes["自车道归属"] = "否"

            attrs = {
                "OPEN_STATUS": None,  # PVB 无关 ['Open', 'Close', 'Halfway']
                "SELF_LANE_OWNERSHIP": {"是": 0, "否": 1, "不清楚": -1}[attributes["自车道归属"]],
                "SELF_LANE_CROSS": {"是": 0, "否": 1, "不清楚": -1}[attributes["自车道压线"]],
                "SHIELD": {"是": True, "否": False}[attributes["遮挡"]],
                "MULTI_BBOX": None,  # TODO
                "VEHICLE_DOOR_STARUS": attributes["车舱门开启方位"] != "关闭",
                "VEHICLE_DOOR_OPEN_SIDE": {
                    "关闭": False,
                    "左侧车舱门开启": "LEFT",
                    "右侧车舱门开启": "RIGHT",
                    "左右均开启": "LEFTnRIGHT",
                }[attributes["车舱门开启方位"]],
                "VEHICLE_TAILGATE_STARUS": {"是": True, "否": False}[attributes["车尾门(后备箱)开合"]],
                "ACCIDENTA_STATUS": {"是": True, "否": False}[attributes["事故车属性"]],
                "LIGHT_STATUS": {
                    "关闭": False,
                    "左转灯亮": "TRUN_LEFT",
                    "右转灯亮": "TURN_RIGHT",
                    "制动(刹车)灯亮": "BRAKE",
                    "双闪灯亮": "DOUBLE FLASH",
                }[attributes.get("车灯", "关闭")],
                "VEHICLE_TRIKE_MANNED_STATUS": {"是": True, "否": False}[attributes["三轮车是否有人"]],
            }

            return attrs
        elif task in ["gop"]:
            open_sets = ["开启地锁", "开启道闸闸杆", "打开的道闸闸杆"]
            close_sets = ["关闭地锁", "道闸闸杆", "关闭道闸闸杆", "关闭的道闸闸杆", "闭合的道闸闸杆"]
            halfway_sets = ["开启中的道闸闸杆"]

            if label in open_sets:
                status = "open"
            elif label in close_sets:
                status = "close"
            elif label in halfway_sets:
                status = "halfway"
            else:
                status = None

            attrs = {
                "OPEN_STATUS": status,
                "SELF_LANE_OWNERSHIP": None,
                "SELF_LANE_CROSS": None,
                "SHIELD": False,
                "MULTI_BBOX": None,
                "VEHICLE_DOOR_STARUS": None,
                "VEHICLE_DOOR_OPEN_SIDE": None,
                "VEHICLE_TAILGATE_STARUS": None,
            }
            return attrs

    @property
    def corse_label(self):
        if self.task == "gop":
            return "GOP"  # 统一一类去处理
        else:
            corse = self.label.split("_")[0]
            if corse == "VEHICLE":
                if self.label != "VEHICLE_CAR":
                    corse = "TRUCK"
        return corse

    @property
    def score3d(self):
        return self.score

    @property
    def id(self):
        return self.track_id

    @property
    def global_location(self):
        ego_location = np.array(self.location + [1])[np.newaxis, :].transpose()
        enu_location = np.array(self.ego2world @ ego_location)[:3, 0]
        return enu_location.tolist()

    @property
    def data(self):
        boxes = self.global_location + [self.length, self.width, self.height, self.yaw]
        boxes.extend([self.token, self.timestamp])
        return boxes

    @property
    def bbox3d(self):
        return self.location + [self.length, self.width, self.height, 0.0, 0.0, self.yaw]

    @property
    def valid(self):
        if self.label is None:
            logger.warning("Label is None")
            return False
        if None in self.location:
            logger.warning("location is None")
            return False
        if self.yaw is None:
            logger.warning("yaw is None")
            return False
        return True

    def reset_info2d(self, targets_2d):
        def find_match(filename):
            src_camera_name = filename.split("/")[-2].split("#")[0]
            for target in targets_2d:
                box_2d, score, token, camera_name, filename = target
                if src_camera_name == camera_name:
                    return target
            raise NotImplementedError

        for idx, target_2d in enumerate(self.info2d):
            box_2d, score, token, camera_name, filename = find_match(target_2d.filename)

            assert self.info2d[idx].token == token

            self.info2d[idx].filename = filename
            self.info2d[idx].left, self.info2d[idx].top, self.info2d[idx].width, self.info2d[idx].height = box_2d
            self.info2d[idx].score = score

    def reset_info3d(self, target_3d):
        assert self.token == target_3d["token"]
        self.location = target_3d["location"]
        self.length = target_3d["length"]
        self.width = target_3d["width"]
        self.height = target_3d["height"]
        self.yaw = target_3d["yaw"]

    def set_location(self, location):
        assert len(location) == 3
        self.location = location

    def set_dimension(self, dimension):
        assert len(dimension) == 3
        self.length, self.width, self.height = dimension

    def set_rotation(self, rotation):
        self.yaw = rotation


@dataclass
class BundleFrameTargets:
    CAMERA_NAMES = [
        "center_camera_fov30",
        "center_camera_fov120",
        "left_front_camera",
        "left_rear_camera",
        "right_front_camera",
        "right_rear_camera",
        "front_camera_fov195",
        "rear_camera_fov195",
        "left_camera_fov195",
        "right_camera_fov195",
        "rear_camera",  # 一定要放最后
    ]

    @classmethod
    def build_from_sensebee(cls, sensebee_meta, task="pvb"):
        BFT = BundleFrameTargets()

        cls.valid = sensebee_meta.get("valid", True)
        BFT.bundle_targets_3d = []
        for target in sensebee_meta["step_1"]["result"]:  # 3D-2D框
            cur_target_3d = Target3D.build_from_sensenee(target, task=task)
            if cur_target_3d.valid:
                BFT.bundle_targets_3d.append(cur_target_3d)

        BFT.bundle_targets_2d = []
        for target in sensebee_meta["step_1"]["resultRect"]:  # 纯2D框
            try:
                cur_target_2d = Target2D.build_from_sensesee(target)
            except Exception as e:
                continue
            if cur_target_2d.valid:
                BFT.bundle_targets_2d.append(cur_target_2d)
        return BFT

    @property
    def targets(self):
        return self.bundle_targets_3d + self.bundle_targets_2d

    def set_ego2world(self, ego2world):
        self.ego2world = ego2world
        for idx, _ in enumerate(self.bundle_targets_3d):
            self.bundle_targets_3d[idx].ego2world = ego2world

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        for idx, _ in enumerate(self.bundle_targets_3d):
            self.bundle_targets_3d[idx].timestamp = timestamp
            for jdx, _ in enumerate(self.bundle_targets_3d[idx].info2d):
                self.bundle_targets_3d[idx].info2d[jdx].timestamp = timestamp
        for idx, _ in enumerate(self.bundle_targets_2d):
            self.bundle_targets_2d[idx].timestamp = timestamp

    def set_calibration(self, calibs):
        self.cache_calibs = calibs
        self.calibrations = dict()
        for camera_meta in calibs["cameras"]:
            camera_name = camera_meta["calib"]["calName"]

            lidar2camera_rt = np.eye(4)
            lidar2camera_rt[:3, :] = np.array(camera_meta["calib"]["T"])

            camera_dist = camera_meta["calib"].get("SourceFisheyeDistortion", [])

            self.calibrations[camera_name] = {
                "data_path": camera_meta["image"],
                "camera_intrinsic": np.array(camera_meta["calib"]["P"])[:3, :3].tolist(),
                "camera_dist": camera_dist,
                "lidar2camera_rt": lidar2camera_rt.tolist(),
            }

    def set_ego_vels(self, ego_vels):
        self.ego_vels = ego_vels

    @classmethod
    def empty_format(cls, timestamp, ego_vels, ego2global_transformation_matrix):
        return {
            "timestamp": timestamp * 1000 * 1000,
            "ego2global_transformation_matrix": ego2global_transformation_matrix.tolist(),
            "ego_velocity": ego_vels.tolist(),
            "sensors": {},  # 需要外部补充
            "Objects": None,
            "Pure2DObjects": None,
            "vehicle_id": None,
            "case_name": None,
        }

    def pap_format(self):
        def find_camera_name(filename):
            for name in self.CAMERA_NAMES:
                if name in filename:
                    return name
            raise NotImplementedError(filename)

        objects_3d, objects_2d = [], {}
        for target in self.bundle_targets_3d:
            info2d = {}
            for box2d in target.info2d:
                camera_name = find_camera_name(box2d.filename)
                info2d[camera_name] = box2d.pap_format
            data = {
                "token": target.token,
                "bbox3d": target.bbox3d,
                "score3d": target.score3d,
                "velocity": target.vels,
                "id": target.track_id,
                "label": target.label,
                "attribute": target.attribute,
                "info2d": info2d if info2d else None,
            }
            objects_3d.append(data)

        for target in self.bundle_targets_2d:
            camera_name = find_camera_name(target.filename)
            if camera_name not in objects_2d:
                objects_2d[camera_name] = {"label2d": [], "score2d": [], "bbox2d": []}

            objects_2d[camera_name]["label2d"].append(target.label)
            objects_2d[camera_name]["score2d"].append(target.score2d)
            objects_2d[camera_name]["bbox2d"].append(target.xyxy)

        return {
            "timestamp": self.timestamp * 1000 * 1000,
            "ego2global_transformation_matrix": self.ego2global_transformation_matrix.tolist(),
            "ego_velocity": self.ego_vels.tolist(),
            "sensors": {},  # 需要外部补充
            "Objects": objects_3d,
            "Pure2DObjects": objects_2d,
            "vehicle_id": None,
            "case_name": None,
        }
