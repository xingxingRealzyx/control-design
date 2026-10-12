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

#ifndef CONTROL_ANALYSIS_H_
#define CONTROL_ANALYSIS_H_

// What can be done to a model: controllability, observability, and the
// structure behind a "no" answer.
//
// This module exists to keep a design honest *before* any synthesis runs. The
// three outcomes that matter are not "yes/no":
//
//   1. fully controllable/observable - arbitrary pole placement is available;
//   2. not full rank, but every uncontrollable mode is already stable
//      (stabilizable) - a stabilising controller exists, but the offending
//      states cannot be moved, only left alone;
//   3. an unstable mode that cannot be reached by the input - no controller
//      fixes this. It is a physical statement about where the actuator is, and
//      a design tool that grinds ahead anyway produces gains that look fine and
//      a plant that diverges.
//
// Mode classification uses the PBH test rather than the rank of the
// controllability matrix, because the PBH test names *which* mode is at fault:
// "the mode at -0.5 is unreachable" is actionable, "rank 3 < 4" is not.

#include <Eigen/Core>
#include <complex>
#include <vector>

#include "control/types.h"

namespace control {

// Result of a full rank test, with the numerical margin kept for the caller.
// A rank test is thresholded, so `IsFullRank()` being true is not the whole
// story: a system that is controllable "by 1e-12" will produce controller gains
// that are enormous, and the caller deserves to know that before asking for
// them.
struct ControllabilityResult {
    RankResult rank;  // rank of the controllability matrix
    bool fully_controllable = false;
    bool stabilizable = false;  // every uncontrollable mode is stable
    // Modes (eigenvalues of A) that the input cannot influence, with the
    // measure of how badly; sorted by descending real part so the dangerous one
    // is first.
    ComplexVector uncontrollable_modes;
    // Smallest singular value of [A - lambda I, B] per mode: the PBH measure.
    Vector pbh_measures;
    // Set when rank is full but only barely, i.e. the smallest singular value
    // of the controllability matrix is within `warning_ratio` of the tolerance.
    bool marginal = false;
    std::string notes;
};

struct ObservabilityResult {
    RankResult rank;  // rank of the observability matrix
    bool fully_observable = false;
    bool detectable = false;  // every unobservable mode is stable
    ComplexVector unobservable_modes;
    Vector pbh_measures;  // smallest singular value of [A - lambda I; C]
    bool marginal = false;
    std::string notes;
};

struct AnalysisOptions {
    // Threshold for the rank tests, relative to the largest singular value of
    // the matrix being tested. 1e-9 is right for double precision on models
    // that are not pathologically scaled.
    double relative_tolerance = 1e-9;
    // A full-rank result whose smallest singular value is below
    // `marginal_ratio` times the largest is reported as marginal: controllable
    // in exact arithmetic, unusable for design in practice.
    double marginal_ratio = 1e-6;
};

// Controllability matrix C = [B, AB, A^2 B, ..., A^(n-1) B], n x (n*m). Exposed
// because it is occasionally useful to look at, and because the tests check the
// rank computation against it directly.
Matrix ControllabilityMatrix(const LtiSystem& system);

// Observability matrix O = [C; CA; ...; CA^(n-1)], (n*p) x n.
Matrix ObservabilityMatrix(const LtiSystem& system);

ControllabilityResult AnalyseControllability(
    const LtiSystem& system, const AnalysisOptions& options = {});
ObservabilityResult AnalyseObservability(const LtiSystem& system,
                                         const AnalysisOptions& options = {});

// PBH test, per mode: is the mode lambda reachable by the input? Returns the
// smallest singular value of [A - lambda I, B], which is zero (up to round-off)
// exactly when the mode is uncontrollable. Exposed for callers who want to
// interrogate a specific eigenvalue, e.g. one they are about to try to move.
double PbhControllabilityMeasure(const LtiSystem& system,
                                 std::complex<double> pole);
double PbhObservabilityMeasure(const LtiSystem& system,
                               std::complex<double> pole);

// Orthonormal bases of the four Kalman subspaces:
//   controllable and observable, controllable but not observable,
//   observable but not controllable, and neither.
// The columns are orthonormal and the four bases together span the state space:
// the returned matrices have n rows and as many columns as the subspace has
// dimensions, so a pair of them can be composed into a similarity
// transformation that exposes the structure of the model.
struct KalmanDecomposition {
    Matrix controllable_observable;
    Matrix controllable_unobservable;
    Matrix uncontrollable_observable;
    Matrix uncontrollable_unobservable;

    // Rank of each subspace, i.e. the column counts above.
    int controllable_observable_dimension = 0;
    int controllable_unobservable_dimension = 0;
    int uncontrollable_observable_dimension = 0;
    int uncontrollable_unobservable_dimension = 0;

    // True when the four bases together span R^n (which they must).
    bool spans_state_space = false;
    // The minimal realisation order: states that are both reachable and
    // observable. Everything else can be removed without changing the
    // input-output behaviour.
    int minimal_order() const { return controllable_observable_dimension; }
};

KalmanDecomposition DecomposeKalman(const LtiSystem& system,
                                    const AnalysisOptions& options = {});

// A minimal realisation: the model restricted to its controllable and
// observable subspace, in balanced coordinates. Input-output behaviour is
// preserved exactly (up to round-off) while the state dimension drops to
// `minimal_order()`. Useful when a model was assembled from subsystem models
// and carries redundant states.
LtiSystem MinimalRealisation(const LtiSystem& system,
                             const AnalysisOptions& options = {});

// Human-readable summary of a controllability/observability verdict, written
// for a report rather than for a log line: it says what is wrong and what it
// implies for a design.
std::string DescribeControllability(const ControllabilityResult& result);
std::string DescribeObservability(const ObservabilityResult& result);

}  // namespace control

#endif  // CONTROL_ANALYSIS_H_
