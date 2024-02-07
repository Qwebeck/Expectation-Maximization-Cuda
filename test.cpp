#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Eigen::EigenSolver;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

int main()
{
    MatrixXd m(2, 2);
    m(0, 0) = 1.0;
    m(1, 0) = 0.0;
    m(0, 1) = 0.0;
    m(1, 1) = 1.0;
    std::cout << m << std::endl;

    EigenSolver<MatrixXd> eigenValueSolver(m);

    MatrixXcd eigenvalueMatrix = eigenValueSolver.eigenvalues();
    MatrixXcd eigenvectorMatrix = eigenValueSolver.eigenvectors();

    std::cout << "The eigenvectors of m are:" << std::endl
              << eigenvectorMatrix << std::endl;
}