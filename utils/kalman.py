import numpy as np
from numpy.linalg import inv

# why not the one from opencv? this implementation more flexible and could be easily changed.

class Kalman:
    # np.eye( int, int, ... )
    #     but
    # np.zeros( tuple(int, int), ... )
    def __init__(self, dynam_params, measure_params, control_params):

        dp = dynam_params       # Dimensionality of the state
        mp = measure_params     # Dimensionality of the measurement
        cp = control_params     # Dimensionality of the the control vector

        # zeros                                                         shapes
        self.state_pre = self.zeros(dp, 1)              # x(k|k-1)      dp, 1
        self.state_post = self.zeros(dp, 1)             # x(k|k)        dp, 1

        self.error_cov_pre = self.zeros(dp, dp)         # P(k|k-1)      dp, dp
        self.gain = self.zeros(dp, mp)                  # K(k)          dp, mp
        self.error_cov_post = self.zeros(dp, dp)        # P(k|k)        dp, dp
        self.control_matrix = self.zeros(dp, cp)        # B(k)          dp, cp
        self.measurement_matrix = self.zeros(mp, dp)    # H(k)          mp, dp

        # eye
        self.transition_matrix = self.eye(dp, dp)       # F(k)          dp, dp
        self.process_noise_cov = self.eye(dp, dp)       # Q(k)          dp, dp
        self.measurement_noise_cov = self.eye(mp, mp)   # R(k)          mp, mp

        self.predicted = self.measurement_matrix.dot(self.state_pre)
        self.corrected = self.measurement_matrix.dot(self.state_post)

    # NOTICE that coordinates measured like z = measurement_matrix * state_
    # That means if we return state_post to the global file like state[:1]
    # we cant compute z because of wrong sizes of matrices!

    @staticmethod
    def zeros(param_1, param_2):
        return np.zeros((param_1, param_2), dtype=np.float32)

    @staticmethod
    def eye(param_1, param_2):
        return np.eye(param_1, param_2, dtype=np.float32)

    def pre_fit_cov(self):
        # S(k) = R(k) + H(k) * P(k|k-1) * Ht (k)
        # (mp, mp) + (mp, dp) * (dp, dp) * (dp, mp).t =
        # (mp, mp) + (mp, dp) * (dp, mp) =
        # (mp, mp) + (mp, mp) =
        # (mp, mp)
        ret = self.measurement_noise_cov + \
               self.measurement_matrix.dot(self.error_cov_pre).dot(self.measurement_matrix.transpose())
        return ret

    def predict(self, control=None):

        # x(k|k-1) = F(k) * x(k-1|k-1)
        # (dp, 1) = (dp, dp) * (dp, 1)
        self.state_pre = self.transition_matrix.dot(self.state_post)

        if control is not None:
            # x(k|k-1) = x(k|k-1) + B(k) * u(k)
            # (dp, 1) += (dp, cp) * (cp, 1)
            self.state_pre += self.control_matrix.dot(control)

        # P(k|k-1) = F(k) * P(k-1|k-1) Ft(k) + Q(k)
        # (dp, dp) = (dp, dp) * (dp, dp) * (dp, dp) + (dp, dp)
        self.error_cov_pre = (self.transition_matrix.dot(self.error_cov_post)).dot(self.transition_matrix.transpose()) \
                             + self.process_noise_cov
        # update
        self.state_post = self.state_pre
        self.predicted = self.measurement_matrix.dot(self.state_pre)

        return self.predicted

    def correct(self, measurement):
        # temp2 = H(k) * P(k|k-1)
        temp2 = self.measurement_matrix.dot(self.error_cov_pre)
        # temp3 = H(k) * P(k|k-1) * Ht(k) + R(k)
        temp3 = temp2.dot(self.measurement_matrix.transpose()) + self.measurement_noise_cov
        # (H(k) * P(k|k-1) * Ht(k) + R(k))_inv * H(k) * P(k|k-1)
        temp4 = inv(temp3).dot(temp2)

        # K(k) = P(k|k - 1) * Ht(k) * S_inv(k)
        self.gain = temp4.transpose()

        # x(k|k) = x(k|k-1) + K(k) (z(k) - H(k) * x(k|k-1))
        # self.state_post = self.state_pre + self.gain.dot(measurement - self.measurement_matrix * self.state_pre)
        self.state_post = self.state_pre + self.gain.dot(measurement - self.measurement_matrix.dot(self.state_pre))
        # P(k|k) = ( I - K(k) * H(k) ) * P(k|k-1)
        self.error_cov_post = self.error_cov_pre - self.gain.dot(temp2)
        del temp2, temp3, temp4

    # def estimate(self, position):
    #     self.predict()
    #     self.correct(position)
    #
    #     return self.measurement_matrix.dot(self.state_pre)

    def distance(self, measurement):
        res = self.predicted - measurement
        cov = inv(self.pre_fit_cov())
        return np.sqrt(res.transpose().dot(cov).dot(res))
