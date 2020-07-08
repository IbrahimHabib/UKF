#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pow;

double angle_cal(double angle) {
  angle = std::fmod(angle + M_PI, 2 * M_PI);  
  if (angle < 0) angle += 2 * M_PI;
  return angle - M_PI;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = 7;
  n_sig = 2 * n_aug_ + 1;
  lambda_ = 3.0 - n_x_;
  double w0 = lambda_ / (lambda_ + n_aug_);
  double w = 1 / (2 * (lambda_ + n_aug_));

  Xsig_pred_ = MatrixXd(n_x_, n_sig);
  Xsig_pred_.fill(0.0);
  weights_ = VectorXd(n_sig);
  weights_.fill(w);
  weights_(0) = w0;
  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if (is_initialized_) {
    double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
      UpdateLidar(meas_package);
    }
    else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
      UpdateRadar(meas_package);
    }
  }
  else {
    x_.fill(0.0);
    x_.head(2) << meas_package.raw_measurements_;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    
    if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
      UpdateLidar(meas_package);
    }
    else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
      UpdateRadar(meas_package);
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  MatrixXd P_Aug = MatrixXd(n_aug_, n_aug_);
  P_Aug.fill(0.0);
  P_Aug.topLeftCorner(n_x_, n_x_) = P_;
  P_Aug(n_x_, n_x_) = pow(std_a_, 2);
  P_Aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);
  MatrixXd X_Sig = MatrixXd(n_aug_, n_sig);
  X_Sig.fill(0.0);
  VectorXd x_Aug = VectorXd(n_aug_);
  x_Aug.fill(0.0);
  x_Aug.head(n_x_) = x_;
  MatrixXd square_aug = P_Aug.llt().matrixL();
  double c_aug = sqrt(lambda_ + n_aug_);
  MatrixXd cA_aug = c_aug * square_aug;
  X_Sig.col(0) = x_Aug;
  for (int i = 0; i< n_aug_; ++i) {
    X_Sig.col(i+1)       = x_Aug + cA_aug.col(i);;
    X_Sig.col(i+1+n_aug_) = x_Aug - cA_aug.col(i);;
  }
 
  double dt = delta_t;
  Xsig_pred_.fill(0.0);

  for (int i = 0; i < n_sig; i++) {
    double px = X_Sig(0, i);
    double py = X_Sig(1, i);
    double v = X_Sig(2, i); 
    double psi = X_Sig(3, i);
    double psid = X_Sig(4, i);
    double nu_a = X_Sig(5, i);
    double nu_psidd = X_Sig(6, i);
    
    if (std::fabs(psid) > 0.001) {
      Xsig_pred_(0, i) = px + v / psid * (sin(psi + psid * dt) - sin(psi)) +
                        pow(dt, 2) / 2 * cos(psi) * nu_a;
      Xsig_pred_(1, i) = py + v / psid * (-cos(psi + psid * dt) + cos(psi)) +
                        pow(dt, 2) / 2 * sin(psi) * nu_a;
    } else {
      Xsig_pred_(0, i) = px + v * dt * cos(psi) +
                         pow(dt, 2) / 2 * cos(psi) * nu_a;
      Xsig_pred_(1, i) = py + v * dt * sin(psi) +
                         pow(dt, 2) / 2 * sin(psi) * nu_a;
    }
    
    Xsig_pred_(2, i) = v + 0 + dt * nu_a;
    Xsig_pred_(3, i) = psi + psid * dt + pow(dt, 2) / 2 * nu_psidd;
    Xsig_pred_(4, i) = psid + 0 + dt * nu_psidd;
  }

  
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  for (int i = 0; i < n_sig; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i < n_sig; i++) {
    P = P + weights_(i) *
            (Xsig_pred_.col(i) - x_) *
            (Xsig_pred_.col(i) - x_).transpose();
  }

  x_ = x;
  P_ = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
   VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();
  
  MatrixXd H = MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  MatrixXd R = MatrixXd(n_z, n_z);
  R << pow(std_laspx_, 2), 0,
       0, pow(std_laspy_, 2);

  VectorXd z_pred = VectorXd(n_z);
  z_pred = x_.head(n_z);

  VectorXd y = z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd S_inv = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * S_inv;

  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();

  MatrixXd Z_sig = MatrixXd(n_z, n_sig);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  for (int i = 0; i < n_sig; i++) {
      double Px = Xsig_pred_(0, i);
      double Py = Xsig_pred_(1, i);
      double V = Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i);
      double yaw_d = Xsig_pred_(4, i);
      double v1 = cos(yaw)*V;
      double v2 = sin(yaw)*V;

      // measurement model
      double r = sqrt(Px*Px + Py*Py); // r
      double phi = std::atan2(Py, Px); // phi
      double r_d = 0.0;
      if (std::fabs(r) > 0.001) {
        r_d = (Px*v1 + Py*v2) / r;  // r_dot
      }

      Z_sig(0, i) = r;
      Z_sig(1, i) = phi;
      Z_sig(2, i) = r_d;
  }

  for (int i = 0; i < n_sig; i++) {
    z_pred = z_pred + weights_(i) * Z_sig.col(i);  
  }
  
  MatrixXd R = MatrixXd(n_z, n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  double std_rho2 = pow(std_radr_, 2);
  double std_phi2 = pow(std_radphi_, 2);
  double std_rhod2 = pow(std_radrd_, 2);
  double mod_angle = 0.0;

  R << std_rho2, 0, 0,
       0, std_phi2, 0,
       0, 0, std_rhod2;

  for (int i = 0; i < n_sig; i++) {
    VectorXd z_diff = Z_sig.col(i) - z_pred;
    z_diff(1) = angle_cal(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i = 0; i < n_sig; i++) {
    VectorXd z_diff = Z_sig.col(i) - z_pred;
    z_diff(1) = angle_cal(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = angle_cal(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = MatrixXd(n_x_, n_z); 
  MatrixXd S_inv = S.inverse();
  VectorXd residuals = z - z_pred;
  residuals(1) = angle_cal(residuals(1));

  K = Tc * S_inv;

  x_ = x_ + K * residuals;
  MatrixXd Kt = K.transpose();
  P_ = P_ - K * S * Kt;
}