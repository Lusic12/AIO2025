import numpy as np
from scipy.linalg import block_diag
import cv2
import glob



class KalmanFilter(object):
    """
    Kalman Filter để theo dõi bounding box

    State vector (6D):
        x = [l, t, w, h, dl, dt]^T

    Trong đó:
        (l, t) : tọa độ góc trái-trên của bbox
        (w, h) : width / height của bbox (theo dõi trực tiếp, không log-scale)
        (dl, dt): vận tốc theo trục l và t

    Mô hình động học:
        - (l, t): constant velocity
        - (w, h): random walk (không có vận tốc)
    """

    def __init__(self, z_init, dt, std_meas):
        """
        Parameters
        ----------
        z_init : ndarray (4,)
            Bounding box ban đầu [l, t, w, h]
        dt : float
            Bước thời gian giữa hai frame
        std_meas : float
            Độ lệch chuẩn của measurement noise (pixel)
        """
        self.dt = dt

        # =====================================================
        # Transition Matrix F (6x6)
        # =====================================================
        # Mô hình:
        #   l_{k+1} = l_k + dl_k * dt
        #   t_{k+1} = t_k + dt_k * dt
        #   w_{k+1} = w_k
        #   h_{k+1} = h_k
        #   dl_{k+1} = dl_k
        #   dt_{k+1} = dt_k
        #
        # => constant velocity cho vị trí
        # => w, h giữ nguyên (chỉ thay đổi nhờ process noise)
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0],
            [0, 1, 0, 0, 0, self.dt],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # =====================================================
        # Measurement Matrix H (4x6)
        # =====================================================
        # Detector / GT đo trực tiếp:
        #   z = [l, t, w, h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # =====================================================
        # Process Noise Covariance Q
        # =====================================================
        # Giả sử:
        #   - Chuyển động có gia tốc ngẫu nhiên Gaussian
        #   - Gia tốc ảnh hưởng lên (l, dl) và (t, dt)
        #
        # Với 1D:
        #   p_{k+1} = p_k + v_k dt + 0.5 a dt^2
        #   v_{k+1} = v_k + a dt
        #
        # Gia tốc a ~ N(0, std_meas^2)
        #
        # => hệ số tác động của a lên state [p, v]:
        #   G = [0.5*dt^2, dt]^T
        B0 = std_meas * np.array([
            [(dt ** 2) / 2],
            [dt]
        ])

        # Noise cho w, h (random walk):
        #   w_{k+1} = w_k + noise
        #   h_{k+1} = h_k + noise
        B1 = np.diag([std_meas, std_meas])

        # Ghép block noise cho:
        #   - (l, dl)
        #   - (t, dt)
        #   - (w, h)
        #
        # np.kron(I2, B0) tạo 2 block cho l và t
        B = block_diag(
            *[np.kron(np.eye(2, dtype='f8'), B0), B1]
        )

        # Q = B * B^T
        # (covariance sinh ra từ gia tốc + random walk)
        self.Q = np.dot(B, B.T)

        # =====================================================
        # Measurement Noise Covariance R
        # =====================================================
        # Detector đo [l, t, w, h] trực tiếp
        # Giả sử noise độc lập, Gaussian
        D0 = np.diag([std_meas, std_meas, std_meas, std_meas])
        self.R = np.dot(D0, D0.T)

        # =====================================================
        # Initial State Covariance P
        # =====================================================
        # - l, t, w, h: tin vừa phải (std_meas^2)
        # - dl, dt: không biết vận tốc ban đầu => variance lớn hơn
        self.P = np.array([
            [std_meas ** 2, 0, 0, 0, 0, 0],
            [0, std_meas ** 2, 0, 0, 0, 0],
            [0, 0, std_meas ** 2, 0, 0, 0],
            [0, 0, 0, std_meas ** 2, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Mean state vector
        self.x = np.zeros(self.F.shape[1])
        self.x[0:4] = z_init

    def predict(self):
        """
        Prediction step:
            x_k|k-1 = F x_{k-1|k-1}
            P_k|k-1 = F P F^T + Q
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Update step với measurement z = [l, t, w, h]
        """
        # Innovation (residual)
        z_pred = np.dot(self.H, self.x)
        y = z - z_pred

        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Posterior mean
        self.x = self.x + np.dot(K, y)

        # Posterior covariance
        # NOTE: must be matrix multiplication, not element-wise.
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, self.H)) @ self.P

        return np.dot(self.H, self.x)





def clamp_bbox(bbox, min_size=0.1):
    """
    Đảm bảo w, h >= min_size
    (tránh bbox âm do noise)
    """
    bbox = bbox.astype('float64').copy()
    bbox[2] = max(float(bbox[2]), float(min_size))
    bbox[3] = max(float(bbox[3]), float(min_size))
    return bbox



def draw_bbox(bbox, image, color):
    l, t, w, h = bbox.astype('int')
    r, b = l + w, t + h
    image = cv2.rectangle(image, (l, t), (r, b), color=color, thickness=2)
    return image


def main_bbox_tracking():
    z_init = np.array([150, 65, 48, 60], dtype='float64')
    std_meas = 5  # and standard deviation of the measurement in pixel
    dt = 1
    # create KalmanFilter object
    kf = KalmanFilter(z_init, dt, std_meas)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (320, 240)
    out = cv2.VideoWriter('gt.mp4', fourcc, 15.0, frame_size)

    # https://www.votchallenge.net/vot2022/dataset.html
    image_paths = sorted(glob.glob("./vot2022_hand/images/*.jpg"))
    gt = np.loadtxt("./vot2022_hand/groundtruth.txt", delimiter=",")
    for idx, (image_pth, bbox) in enumerate(zip(image_paths, gt)):
        img = cv2.imread(image_pth)
        bbox = clamp_bbox(bbox)
        img = draw_bbox(bbox, img, (255, 0, 0))
        bbox_n = clamp_bbox(bbox + np.random.normal(0, std_meas, size=4))  # adding noise to bbox
        img = draw_bbox(bbox_n, img, (255, 0, 255))

        kf.predict()
        x_est = kf.update(np.array([bbox_n[0], bbox_n[1], bbox_n[2], bbox_n[3]]))
        x_est = clamp_bbox(x_est)
        img = draw_bbox(x_est, img, (255, 255, 255))
        img = cv2.putText(img, " Frame " + str(idx), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       color=(0, 0, 255), thickness=1)
        out.write(img)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img)
        cv2.waitKey(20)
    out.release()


if __name__ == '__main__':
    main_bbox_tracking()
