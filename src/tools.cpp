#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
  	cout << "Invalid estimation or ground_truth data" << endl;
  	return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i)
  {
  	VectorXd residual = estimations[i] - ground_truth[i];

  	//coefficient-wise multiplication
  	residual = residual.array()*residual.array();
  	rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	if (fabs(px) < 0.0001 && fabs(py) < 0.0001)
	{
		px = 0.0001;
		py = 0.0001;
	}
	float rSq = px * px + py * py;
	if (fabs(rSq) < 0.0000002)
	{
		rSq = 0.0000002;
	}
	float r = sqrt(rSq);

	float r32 = r * rSq; //pow(rSq, 3.0 / 2.0);

	float h00 = px / r;
	float h01 = py / r;
	float h10 = -py / rSq;
	float h11 = px / rSq;
	float h20 = py * (vx * py - vy * px) / r32;
	float h21 = px * (vy * px - vx * py) / r32;
	float h22 = h00;
	float h23 = h01;

	//compute the Jacobian matrix
	Hj << h00, h01, 0.0, 0.0,
    h10, h11, 0.0, 0.0,
    h20, h21, h22, h23;

	return Hj;
}
