import warnings
import numpy as np
import cv2
from matplotlib import cm

CAMERA_MODELS = frozenset({ 'PINHOLE', 'EQUIDISTANT', 'SCARAMUZZA' })

class Camera(object):
    def __init__(self, img_width=-1, img_height=-1, instrinsic=np.array([1, 1, 0, 0]), distortion=np.zeros(4)):
        self.img_width = img_width
        self.img_height = img_height

        self.camera_model = 'PINHOLE'
        self.distortion = np.array(distortion)

        self.set_instrinsic(instrinsic)
        self.Rwc = np.eye(3, 3, dtype=np.float64)
        self.Rcw = np.eye(3, 3, dtype=np.float64)
        self.pwc = np.array([[0, 0, 0]], dtype=np.float64).T
        self.pcw = np.array([[0, 0, 0]], dtype=np.float64).T
        self.rvec = np.array([[0, 0, 0]], dtype=np.float64).T
        self.tvec = np.array([[0, 0, 0]], dtype=np.float64).T
        self.camera_scale = 1.0
        self.pose_scale = 1.0

    # def set_distortion(self, k1, k2, k3, p1, p2=0, k4=None, k5=None ,k6=None, camera_model='PINHOLE'):
    #     self.camera_model = camera_model
    #     if self.camera_model == 'PINHOLE':
    #         if k4 is None or k5 is None or k6 is None:
    #             self.distortion = np.array([k1, k2, p1, p2, k3], dtype=np.float64).reshape((5, 1))
    #         else:
    #             self.distortion = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64).reshape((8, 1))
    #     elif self.camera_model == 'EQUIDISTANT':
    #         self.distortion = np.array([k1, k2, k3, p1], dtype=np.float64).reshape((4, 1))

    def set_distortion(self, distortion, camera_model='PINHOLE'):
        '''
        Camera models:
            *PINHOLE poly-4: k1, k2, p1, p2 for cv
            *PINHOLE poly-5: k1, k2, p1, p2, k3 for cv
            *EQUIDISTANT fisheye: k1, k2, k3, k4 for cv fisheye
            *SCARAMUZZA fisheye: k1, ... k11
        '''
        self.camera_model = camera_model
        self.distortion = np.array(distortion)
    
    def set_instrinsic(self, instrinsic):
        instrinsic = np.array(instrinsic)
        if instrinsic.shape == (3, 3):
            self.K = instrinsic
            self.instrinsic = np.array([
                instrinsic[0, 0], instrinsic[1, 1],
                instrinsic[0, 2], instrinsic[1, 2]
            ])
        else:
            self.instrinsic = np.array(instrinsic)

            self.K = np.array([[instrinsic[0], 0, instrinsic[2]],
                               [0, instrinsic[1], instrinsic[3]],
                               [0, 0, 1]])

    def get_width(self):
        return self.img_width

    def get_height(self):
        return self.img_height

    def get_K(self):
        return self.K

    def get_instrinsic(self):
        return self.instrinsic

    def get_distortion(self):
        return self.distortion

    def get_camera_model(self):
        return self.camera_model

    def set_cxcy_half(self):
        self.instrinsic[2] = self.img_width // 2
        self.instrinsic[3] = self.img_height // 2
        self.K[0, 2] = self.img_width // 2
        self.K[1, 2] = self.img_height // 2

    def set_Rwc_pwc(self, Rwc, pwc, pose_scale=1.0):
        self.Rwc = Rwc
        pwc_col = np.array(pwc).reshape((3, 1))
        self.pwc = pwc_col

        self.Rcw = np.transpose(Rwc)
        self.pcw = -np.dot(self.Rcw, pwc_col)

        self.pose_scale = pose_scale
        self._to_r_t()

    def set_Rcw_pcw(self, Rcw, pcw, pose_scale=1.0):
        pcw_col = np.array(pcw).reshape((3, 1))
        self.Rcw = Rcw
        self.pcw = pcw_col

        self.Rwc = np.transpose(Rcw)
        self.pwc = -np.dot(self.Rwc, pcw_col)

        self.pose_scale = pose_scale

        self._to_r_t()

    def set_Rcw_pwc(self, Rcw, pwc, pose_scale=1.0):
        pwc_col = np.array(pwc).reshape((3, 1))
        self.Rcw = Rcw
        self.pwc = pwc_col

        self.Rwc = np.transpose(Rcw)
        self.pcw = -np.dot(self.Rwc, pwc_col)

        self.pose_scale = pose_scale

        self._to_r_t()

    def _to_r_t(self):
        self.rvec = cv2.Rodrigues(self.Rcw)[0]
        self.tvec = self.pcw

    def get_Rcw(self):
        return self.Rcw

    def get_pcw(self):
        return self.pcw

    def get_pwc(self):
        return self.pwc

    def get_Rwc(self):
        return self.Rwc

    def get_pose_scale(self):
        return self.pose_scale

    def get_Tcw(self):
        Tcw = np.eye(4, dtype=self.Rcw.dtype)
        Tcw[:3, :3] = self.Rcw
        Tcw[0, 3] = self.pcw[0, 0]
        Tcw[1, 3] = self.pcw[1, 0]
        Tcw[2, 3] = self.pcw[2, 0]
        return Tcw
    
    def undistort(self, image):
        if self.camera_model == 'PINHOLE':
            return cv2.undistort(image, self.K, self.distortion)
        elif self.camera_model == 'EQUIDISTANT':
            [h, w] = image.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.distortion.reshape((4, 1)),
                  np.eye(3), self.K, (w, h), cv2.CV_16SC2)
            return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) 
        elif self.camera_model == 'SCARAMUZZA':
            warnings.warn("SCARAMUZZA is not supported now!", stacklevel=2)
            return
        return

    def undistort_points(self, points):
        pointss = np.array([points])
        if self.camera_model == 'PINHOLE':
            return cv2.undistortPoints(pointss, self.K, self.distortion, P=self.K)
        elif self.camera_model == 'EQUIDISTANT':
            return cv2.fisheye.undistortPoints(pointss, self.K, self.distortion, P=self.K)
        elif self.camera_model == 'SCARAMUZZA':
            warnings.warn("SCARAMUZZA is not supported now!", stacklevel=2)
            return
        return

    def image_to_world(self, points):
        undistort_point2ds = self.undistort_points(points)[0]
        norm_pts = np.ones((undistort_point2ds.shape[0], 3), dtype=np.float64)
        norm_pts[:, 0] = (undistort_point2ds[:, 0] -  self.K[0, 2]) / self.K[0, 0]
        norm_pts[:, 1] = (undistort_point2ds[:, 1] -  self.K[1, 2]) / self.K[1, 1]

        return norm_pts

    # https://blog.csdn.net/yanglusheng/article/details/52268234
    def calc_project_matrix(self, n=0.001, f=1000):
        project_matrix = np.zeros((4, 4))
        project_matrix[0, 0] = 2 * self.K[0, 0] / self.img_width
        project_matrix[0, 2] = 1 - 2 * self.K[0, 2] / self.img_width
        project_matrix[1, 1] = 2 * self.K[1, 1] / self.img_height
        project_matrix[1, 2] = 2 * self.K[1, 2] / self.img_height - 1
        project_matrix[2, 2] = -(f + n) / (f - n)
        project_matrix[2, 3] = -(2 * f * n) / (f - n)
        project_matrix[3, 2] = -1
        return project_matrix

    def points_to_camera(self, p3d):
        p3d_col = np.array(p3d).reshape((3, 1))
        return np.dot(self.Rcw, p3d_col) + self.pcw

    def project_to_image_one(self, p3d):
        p3d_col = np.array(p3d).reshape((3, 1))
        pt = np.dot(self.Rcw, p3d_col) + self.pcw
        pt = pt.tolist()
        p3ds = np.array([pt], dtype=np.float64)
        p2ds, _ = cv2.projectPoints(p3ds, (0, 0, 0), (0, 0, 0), self.K, self.distortion)
        return p2ds[0][0], pt[2]

    '''
    p3d = [1, 1, 1], or np.array([1, 1, 1])
    output = np.array([[[1, 1, 1]], [[1, 1, ]]]
    '''

    def project_to_image(self, p3ds, has_distortion):
        if self.camera_model == 'PINHOLE':
            has_distortion = False

        p3ds = np.array(p3ds)
        if p3ds.shape[1] != 3:
            return [], [], []
        pcw = self.pcw.reshape((3,))
        cam_p3ds = np.dot(p3ds, self.Rcw.T) + pcw
        z = cam_p3ds[:, 2]
        # print(self.K)
        
        p3dss = np.array([cam_p3ds], dtype=np.float64)
        camera_model = self.camera_model

        if has_distortion:
            distortion = self.distortion
        else:
            distortion = np.zeros((5,), dtype=np.float64)
            camera_model = 'PINHOLE'
            
        if camera_model == 'PINHOLE':
            p2ds, _ = cv2.projectPoints(p3dss, (0, 0, 0), (0, 0, 0), self.K, distortion)
            p2ds = p2ds[:, 0]
        elif camera_model == 'EQUIDISTANT':
            p2ds, _ = cv2.fisheye.projectPoints(p3dss, (0, 0, 0), (0, 0, 0), self.K, distortion)
            p2ds = p2ds[0, :, :]
        elif camera_model == 'SCARAMUZZA':
            p2ds = self.project_scara_fisheye_points(cam_p3ds)
        elif camera_model == 'KB':
            p2ds = self.project_kb_fisheye_points(cam_p3ds)
        # print(p2ds.shape)
        return p2ds, z, cam_p3ds
    
    def project_scara_fisheye_points(self, objectPoints: np.ndarray) -> np.ndarray:
        # assert and reshape
        shape = list(objectPoints.shape)
        assert 3 == shape[-1]
        objectPoints = objectPoints.reshape((-1, 3))

        # project
        imagePoints = self.scara_fisheye_camera_to_image(
            p3ds=objectPoints,
            camera_intrinsic=self.K,
            camera_dist=self.distortion
        )

        # reshape
        shape[-1] = 2
        imagePoints = imagePoints.reshape(shape)

        return imagePoints
    
    def project_kb_fisheye_points(self, objectPoints):
        camera_intrinsic = self.K
        kb_param = self.distortion[0]
        r = np.linalg.norm(objectPoints[:, :2], axis=1).reshape(-1, 1)
        theta = np.arctan2(r, objectPoints[:, 2:3])
        d_theta = theta + np.power(theta, 3)*kb_param[0] \
            + np.power(theta, 5)*kb_param[1] \
            + np.power(theta, 7)*kb_param[2] \
            + np.power(theta, 9)*kb_param[3]
        p2ds = camera_intrinsic @ np.concatenate([d_theta*objectPoints[:, 0:1]/r, d_theta*objectPoints[:, 1:2]/r, np.ones_like(objectPoints[:, 0:1])], 1).transpose()

        image_points = p2ds[:2]

        return image_points.T

    def scara_poly_val(self, param, x):
        p = np.poly1d(param[::-1])
        res = p(x)
        return res

    def scara_fisheye_camera_to_image(self, p3ds, camera_intrinsic, camera_dist):
        aff_ = np.array([
            camera_intrinsic[0][0], camera_intrinsic[0][1],
            camera_intrinsic[1][0], camera_intrinsic[1][1]
        ]).reshape(2, 2)
        xc_ = camera_intrinsic[0][2]
        yc_ = camera_intrinsic[1][2]
        inv_poly_param_ = camera_dist[1]

        norm = np.linalg.norm(p3ds[:, :2], axis=1)

        invNorm = 1 / norm

        theta = np.arctan2(-p3ds[:, 2], norm)

        rho = self.scara_poly_val(inv_poly_param_, theta)

        xn = np.ones((2, p3ds.shape[0]))
        xn[0] = p3ds[:, 0] * invNorm * rho
        xn[1] = p3ds[:, 1] * invNorm * rho

        p2ds = (np.dot(aff_, xn) +
                np.tile(np.array([xc_, yc_]).reshape(2, -1), (xn.shape[1]))).T

        return p2ds

    def project_to_image_filter(self, points, image_shape, has_distortion):
        [h,w] = image_shape[:2]
        img_pts, z, cam_p3ds = self.project_to_image(points, has_distortion)
        for idx, pt in enumerate(img_pts):
            if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h:
                z[idx] = -1.0
        # u_all = img_pts[:, 1]
        # v_all = img_pts[:, 0]
        # uu = np.logical_or((u_all < 0), (u_all > h-1))
        # vv = np.logical_or((v_all < 0), (v_all > w-1))
        # in_image = uu | vv
        # z[in_image] = -1.0
        return img_pts, z, cam_p3ds

    def project_orth_to_image_one(self, p3d):
        pt = np.dot(self.Rcw, np.array(p3d)) + self.pcw.T
        return np.array([pt[0, 0] * self.pose_scale, pt[0, 1] * self.pose_scale]), pt[0, 2]

    '''
    p3d = [1, 1, 1], or np.array([1, 1, 1])
    output = np.array([[[1, 1, 1]], [[1, 1, ]]]
    '''

    def project_orth_to_image(self, p3ds):
        p2ds = []
        z = []
        for p3d in p3ds:
            pt = np.dot(self.Rcw, np.array(p3d)) + self.pcw.T
            p2ds.append([pt[0, 0] * self.pose_scale, self.img_height - pt[0, 1] * self.pose_scale])
            z.append(pt[0, 2])
        return np.array(p2ds), z
# vertices3d: N * 3
def project_to_image(vertices3d, camera, type='persp'):
    # type = [perspective, orthogonal]
    p3ds = vertices3d.reshape((-1, 3))
    if type == 'persp':
        p2ds, z, p3ds_local = camera.project_to_image(p3ds)
    elif type == 'orth':
        p2ds, z = camera.project_orth_to_image(p3ds)
    else:
        p2ds, z = camera.project_to_image(p3ds)
    return p2ds, z, p3ds_local
