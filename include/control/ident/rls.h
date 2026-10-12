// Copyright 2024 The Control Design Library Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONTROL_IDENT_RLS_H_
#define CONTROL_IDENT_RLS_H_

// Recursive least squares, for identifying while the plant runs.
//
// The batch solution is a matrix inverse; the recursive one updates it in
// O(n^2) per sample without ever forming that inverse, which is what makes it
// usable online. With a forgetting factor below one it also tracks a slowly
// changing plant, at the cost of never converging to a fixed answer - which is
// the whole point when the plant is drifting and a defect when it is not.
//
// The identity worth knowing and the one the test checks: with a forgetting
// factor of one, the recursion and the batch least-squares solution agree
// exactly, up to the influence of the initial covariance. That influence is
// `(Phi'Phi + P0^-1)^-1`, so a large `P0` makes the prior negligible and the
// two agree to round-off - which is the statement the test makes rather than
// "approximately equal".

#include <Eigen/Core>

#include "control/ident/dataset.h"
#include "control/ident/ls.h"

namespace control {
namespace ident {

struct RlsOptions {
    // 1.0 for a time-invariant plant (the estimate converges and stays);
    // 0.95..0.999 to track a drifting one. Below about 0.95 the estimate reacts
    // to noise faster than to the plant.
    double forgetting_factor = 1.0;
    // Initial parameter covariance. Large means "I know nothing", which is the
    // honest starting point and what makes the result match batch least
    // squares.
    double initial_covariance = 1e10;
    // Covariance reinitialisation: when the trace of P falls below this
    // fraction of its initial value the estimator has stopped learning, and
    // with a forgetting factor it may also have gone to sleep. Zero disables
    // it.
    double covariance_reset_threshold = 0.0;
};

struct RlsStep {
    Eigen::VectorXd parameters;     // current estimate
    double prediction_error = 0.0;  // a priori error, before the update
    double parameter_change =
        0.0;  // norm of the update: a convergence indicator
    double covariance_trace =
        0.0;  // how much the estimator is still willing to learn
};

class RecursiveLeastSquares {
 public:
    // `parameter_count` is na + nb; the regressor layout must match
    // `ArxRegressor` so the estimate can be compared with the batch one.
    RecursiveLeastSquares(int parameter_count, const RlsOptions& options = {});

    void Reset();

    // One sample: `regressor` is phi[k], `output` is y[k].
    RlsStep Update(const Eigen::VectorXd& regressor, double output);

    const Eigen::VectorXd& parameters() const { return parameters_; }
    const Eigen::MatrixXd& covariance() const { return covariance_; }
    int samples_processed() const { return samples_; }
    // Largest parameter change over the whole run, for a convergence statement.
    double worst_parameter_change() const { return worst_change_; }

 private:
    RlsOptions options_;
    Eigen::VectorXd parameters_;
    Eigen::MatrixXd covariance_;
    int samples_ = 0;
    double worst_change_ = 0.0;
};

// The regressor layout, shared by the batch and recursive paths so the two can
// be compared without a translation step - which is itself a place bugs hide.
//   phi[k] = [-y[k-1] ... -y[k-na], u[k-d] ... u[k-d-nb+1]]
Eigen::VectorXd ArxRegressor(const Dataset& data, int index,
                             int denominator_order, int numerator_order,
                             int delay);

// Runs the recursion over a whole record, which is what the comparison with the
// batch fit needs. Returns the final estimate.
RlsStep RunRecursiveLeastSquares(const Dataset& data, int denominator_order,
                                 int numerator_order, int delay,
                                 const RlsOptions& options, ArxModel* model);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_RLS_H_
