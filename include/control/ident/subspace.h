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

#ifndef CONTROL_IDENT_SUBSPACE_H_
#define CONTROL_IDENT_SUBSPACE_H_

// Realization from a Hankel matrix: getting a state-space model from impulse
// response data.
//
// The method is the Eigensystem Realization Algorithm, which is the Ho-Kalman
// construction. Given the Markov parameters H1, H2, ... of a discrete system
// (H_k = C A^(k-1) B for a strictly proper model), the block Hankel matrix
//
//   H = [ H1  H2  H3 ... ]      H_shifted = [ H2  H3 ... ]
//       [ H2  H3  H4 ... ]                  [ H3  H4 ... ]
//       [ ...            ]                  [ ...        ]
//
// factors as H = Gamma * Delta with Gamma the observability matrix and Delta
// the controllability matrix. Its singular values therefore *are* the system's
// Hankel singular values: the number of significant ones is the order, and that
// is an answer to "what order is this system" that does not depend on trying
// orders until the fit stops improving.
//
//   A = Sigma^(-1/2) U' H_shifted V Sigma^(-1/2),   C = first p rows of Gamma,
//   B = first m columns of Delta,                    D = H0.
//
// What this module does *not* do, stated because the omission matters: it does
// not compute the Markov parameters from arbitrary input/output data. That step
// is what separates ERA from N4SID - N4SID replaces the impulse response with
// an oblique projection of the output onto the input and its own past, and so
// identifies from normal operating data without a special experiment. ERA needs
// an impulse (or a pulse) test, or the Markov parameters from a model. The
// `ArxImpulseResponse` function provides the bridge when an ARX model has
// already been fitted, and that composition - ARX for the Markov parameters,
// ERA for the realization - is a legitimate and much-used route.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/ident/ls.h"

namespace control {
namespace ident {

struct EraOptions {
    // Number of block rows in the Hankel matrix. Needs to be at least the
    // order; more rows make the order determination clearer, at the cost of
    // consuming more Markov parameters (the Hankel needs block_rows + 1 of
    // them).
    int block_rows = 10;
    // Requested order. Zero means "decide from the singular values", which uses
    // `singular_value_threshold`.
    int order = 0;
    // A singular value below this fraction of the largest is treated as zero,
    // so the order is the count above it. The gap between consecutive singular
    // values is usually more informative than the absolute level, and the
    // report includes both.
    double singular_value_threshold = 1e-6;
};

struct EraResult {
    StateSpaceModel model;
    // Hankel singular values, descending. The order is decided here.
    Eigen::VectorXd hankel_singular_values;
    int order = 0;
    // Ratio of consecutive singular values: the largest ratio marks the most
    // natural order. Reported because a hand-picked threshold and a gap can
    // disagree, and the disagreement is worth seeing.
    Eigen::VectorXd singular_value_ratios;
    int order_from_largest_gap = 0;
    double largest_singular_value = 0.0;
    std::string notes;
    std::vector<std::string> warnings;
    bool ok = false;
};

// Realizes a state-space model from Markov parameters. `markov` holds H0, H1,
// ... with H0 the direct feedthrough (zero for a strictly proper model).
EraResult RealizeFromMarkovParameters(const Eigen::VectorXd& markov, int inputs,
                                      int outputs, double sample_time,
                                      const EraOptions& options = {});

// The same from an impulse response, which for a strictly proper model is the
// Markov parameters with a zero prepended.
EraResult RealizeFromImpulseResponse(const Eigen::VectorXd& impulse_response,
                                     double sample_time,
                                     const EraOptions& options = {});

// Poles of a realized discrete model: the eigenvalues of A, sorted by
// descending magnitude. The comparison that matters is against the poles of the
// true system that generated the data.
Eigen::VectorXcd StateSpacePoles(const StateSpaceModel& model);

// Impulse response of a state-space model, for comparing a realization against
// what it was built from.
Eigen::VectorXd StateSpaceImpulseResponse(const StateSpaceModel& model,
                                          int samples);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_SUBSPACE_H_
