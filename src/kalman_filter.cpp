#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
    x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	UpdateInternal(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
	double px = x_[0];
	double py = x_[1];
	double vx = x_[2];
	double vy = x_[3];

	VectorXd z_pred(3);
	double radius = sqrt(px * px + py + py);
	// Check radius for 0.
	double d_rho = (fabs(radius < 0.001))?0:((px * vx + py * vy) / radius);
	z_pred << radius,  atan2(py, px), d_rho;
	VectorXd y = z - z_pred;
	UpdateInternal(y);
}

void KalmanFilter::UpdateInternal(Eigen::VectorXd &y)
{
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd K = P_ * Ht * S.inverse();

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}