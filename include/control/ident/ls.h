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

#ifndef CONTROL_IDENT_LS_H_
#define CONTROL_IDENT_LS_H_

// ARX identification: the workhorse, and worth understanding rather than
// dismissing as old.
//
//   y[k] = -a1 y[k-1] - ... - a_na y[k-na]
//          + b0 u[k-d] + ... + b_{nb-1} u[k-d-nb+1] + e[k]
//
// The right-hand side is *linear in the parameters*, so the whole thing is one
// least-squares problem - which is why ARX is the model every other method is
// compared against and the one that never fails to produce an answer. That last
// property is also its trap: it always produces an answer. If the noise is not
// white, or the input is not persistently exciting, or the true system has a
// delay you did not allow for, the answer is still there, still fits the
// training data, and still wrong. Hence the validation module, which is not
// optional.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/ident/dataset.h"

namespace control {
namespace ident {

struct ArxModel {
    // Denominator: 1 + a1 z^-1 + ... + a_na z^-na. `a` holds a1..a_na.
    Eigen::VectorXd a;
    // Numerator: b0 + b1 z^-1 + ... . `b` holds b0..b_{nb-1}, already including
    // the delay as leading entries (so a pure delay of d samples has d zeros at
    // the front when the model is written with the delay folded into `b`).
    Eigen::VectorXd b;
    // Input delay in samples, kept separately from `b` so a report can state
    // it: the delay is the parameter a process engineer wants to see, and
    // burying it in leading zeros of `b` hides it.
    int delay = 1;
    double sample_time = 0.0;

    int DenominatorOrder() const { return static_cast<int>(a.size()); }
    int NumeratorOrder() const { return static_cast<int>(b.size()); }
    int Order() const { return DenominatorOrder(); }
    bool IsValid() const;
    std::string DescribeProblem() const;
    // 1 + a1 z^-1 + ... as a vector of z^-k coefficients, for reporting poles.
    Eigen::VectorXd DenominatorCoefficients() const;
};

struct ArxFitReport {
    ArxModel model;
    FitMetrics training_fit;
    // Condition number of the regressor matrix: the numerical statement of
    // whether the input was rich enough for this many parameters. A large value
    // means the parameters are not separately identifiable even though the fit
    // looks fine, and the notes say so.
    double regressor_condition_number = 0.0;
    bool ill_conditioned = false;
    std::vector<std::string> warnings;
    std::string notes;
    bool ok = false;
};

// Fits by least squares on the training part of the record.
ArxFitReport FitArx(const Dataset& training, int denominator_order,
                    int numerator_order, int delay = 1);

// Free-run simulation: the predicted output is fed back into the regressor
// instead of the measured one. This is the honest way to use an ARX model - the
// one-step-ahead prediction that the least-squares fit optimises uses the
// measured past, and reporting that number as "the fit" flatters the model by a
// wide margin on any system with slow poles.
//
// `history` supplies the first `denominator_order` outputs; pass the measured
// ones to reproduce the record, or zeros to see the transient from rest.
Eigen::VectorXd SimulateArx(const ArxModel& model, const Eigen::VectorXd& input,
                            double sample_time, const Eigen::VectorXd& history);

// One-step-ahead prediction, using measured past outputs. Provided precisely so
// the difference between it and the free-run simulation can be reported: the
// gap between the two is the clearest single indicator that a model is fitting
// noise.
Eigen::VectorXd PredictArxOneStep(const ArxModel& model, const Dataset& data);

// Discrete state-space form of the ARX model, in observer canonical form. No
// dependency on a control library: the four matrices are returned directly, so
// the model can be handed to any design tool, including `control-design`.
struct StateSpaceModel {
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd c;
    Eigen::MatrixXd d;
    double sample_time = 0.0;
    int Order() const { return static_cast<int>(a.rows()); }
};
StateSpaceModel ToStateSpace(const ArxModel& model);

// Poles of the identified model: the eigenvalues of A, which are the roots of
// the denominator. Reported sorted by descending magnitude, because for a
// discrete model the pole nearest the unit circle is the one that governs.
Eigen::VectorXcd ArxPoles(const ArxModel& model);

// Frequency response of the identified model at `omega` (rad/s), evaluated from
// the difference equation directly. The single most useful plot in
// identification is the measured and predicted frequency responses: a model can
// match a step response and still be wrong at the frequency the loop crosses
// over.
std::complex<double> ArxFrequencyResponse(const ArxModel& model, double omega);

// Impulse response of the model, `samples` entries. This is also the Markov
// parameter sequence, which is what the subspace method consumes, so the two
// modules compose.
Eigen::VectorXd ArxImpulseResponse(const ArxModel& model, int samples);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_LS_H_
