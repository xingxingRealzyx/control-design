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

#include "control/analysis.h"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/lti.h"
#include "tests/testing.h"

namespace {

// Diagonal A with the given poles and an input that enters only the first
// state.
control::LtiSystem Diagonal(const std::vector<double>& poles,
                            double input_scale_c,
                            const std::vector<double>& output_weights) {
    const int n = static_cast<int>(poles.size());
    control::Matrix a = control::Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        a(i, i) = poles[static_cast<std::size_t>(i)];
    }
    control::Matrix b = control::Matrix::Zero(n, 1);
    b(0, 0) = 1.0;
    control::Matrix c = control::Matrix::Zero(1, n);
    for (int i = 0; i < n; ++i) {
        c(0, i) = output_weights[static_cast<std::size_t>(i)];
    }
    static_cast<void>(input_scale_c);
    return control::LtiSystem::Continuous(a, b, c);
}

}  // namespace

TEST(Analysis, ControllabilityMatrixAndRankMatchTheDefinition) {
    // B = [0, 1]^T and A = [[0,1],[-2,-3]]: contr = [B AB] = [[0,1],[1,-3]].
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);

    const control::Matrix matrix = control::ControllabilityMatrix(system);
    CHECK_EQ(matrix.rows(), 2);
    CHECK_EQ(matrix.cols(), 2);
    CHECK_NEAR(matrix(0, 0), 0.0, 0.0);
    CHECK_NEAR(matrix(1, 0), 1.0, 0.0);
    CHECK_NEAR(matrix(0, 1), 1.0, 0.0);
    CHECK_NEAR(matrix(1, 1), -3.0, 0.0);

    const control::ControllabilityResult result =
        control::AnalyseControllability(system);
    CHECK_TRUE(result.fully_controllable);
    CHECK_TRUE(result.stabilizable);
    CHECK_FALSE(result.marginal);
    CHECK_EQ(result.rank.rank, 2);
    CHECK_TRUE(result.uncontrollable_modes.size() == 0);
}

TEST(Analysis, UnreachableStableModeIsStabilizableButNotControllable) {
    // Input enters x1 only; the mode at -3 in x2 can never be moved.
    const control::LtiSystem system = Diagonal({-1.0, -3.0}, 1.0, {1.0, 0.0});
    const control::ControllabilityResult result =
        control::AnalyseControllability(system);
    CHECK_FALSE(result.fully_controllable);
    CHECK_TRUE(result.stabilizable);  // the unreachable mode is already stable
    CHECK_EQ(result.rank.rank, 1);
    CHECK_EQ(result.uncontrollable_modes.size(), 1);
    // The mode is *named*: "rank 1 < 2" is not actionable, "-3.000" is.
    CHECK_NEAR(result.uncontrollable_modes(0).real(), -3.0, 1e-9);
    CHECK_TRUE(result.notes.find("stable") != std::string::npos);
}

TEST(Analysis, UnreachableUnstableModeIsRefusedWithAnExplanation) {
    // The case that no controller can fix, and the one a design tool must not
    // grind past: an unstable mode the actuator cannot influence.
    const control::LtiSystem system = Diagonal({-1.0, 1.0}, 1.0, {1.0, 0.0});
    const control::ControllabilityResult result =
        control::AnalyseControllability(system);
    CHECK_FALSE(result.fully_controllable);
    CHECK_FALSE(result.stabilizable);
    CHECK_EQ(result.uncontrollable_modes.size(), 1);
    CHECK_NEAR(result.uncontrollable_modes(0).real(), 1.0, 1e-9);
    // The message has to say what it implies, not just report a rank.
    CHECK_TRUE(result.notes.find("no controller fixes") != std::string::npos);
    CHECK_TRUE(control::DescribeControllability(result).find(
                   "unreachable modes") != std::string::npos);
}

TEST(Analysis, ObservabilityMirrorsControllability) {
    const control::LtiSystem system = Diagonal({-1.0, -4.0}, 1.0, {1.0, 0.0});
    const control::ObservabilityResult result =
        control::AnalyseObservability(system);
    CHECK_FALSE(result.fully_observable);
    CHECK_TRUE(result.detectable);  // the invisible mode is stable
    CHECK_EQ(result.unobservable_modes.size(), 1);
    CHECK_NEAR(result.unobservable_modes(0).real(), -4.0, 1e-9);

    // An unstable invisible mode makes the system undetectable, which is the
    // dual of the unstabilizable case.
    const control::LtiSystem undetectable =
        Diagonal({-1.0, 2.0}, 1.0, {1.0, 0.0});
    const control::ObservabilityResult bad =
        control::AnalyseObservability(undetectable);
    CHECK_FALSE(bad.detectable);
    CHECK_TRUE(bad.notes.find("no observer") != std::string::npos);
}

TEST(Analysis, ObservabilityMatrixMatchesTheDefinition) {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 3.0, 1.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);
    const control::Matrix matrix = control::ObservabilityMatrix(system);
    // Row 1 is C, row 2 is C A = [3, 1] * A = [-2, 0].
    CHECK_EQ(matrix.rows(), 2);
    CHECK_EQ(matrix.cols(), 2);
    CHECK_NEAR(matrix(0, 0), 3.0, 0.0);
    CHECK_NEAR(matrix(0, 1), 1.0, 0.0);
    CHECK_NEAR(matrix(1, 0), -2.0, 0.0);
    CHECK_NEAR(matrix(1, 1), 0.0, 0.0);
    CHECK_TRUE(control::AnalyseObservability(system).fully_observable);
}

TEST(Analysis, MarginalControllabilityIsReportedNotJustTheBoolean) {
    // Almost uncontrollable: the input reaches the second state only through
    // 1e-8. The rank is full in exact arithmetic, and pretending that means
    // "arbitrary pole placement is available" produces gains of order 1e8.
    control::Matrix a(2, 2);
    a << -1.0, 0.0, 0.0, -2.0;
    control::Matrix b(2, 1);
    b << 1.0, 1e-8;
    control::Matrix c(1, 2);
    c << 1.0, 1.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);
    const control::ControllabilityResult result =
        control::AnalyseControllability(system);
    CHECK_TRUE(result.fully_controllable);
    CHECK_TRUE(result.marginal);
    CHECK_TRUE(result.rank.condition_number() > 1e6);
    CHECK_TRUE(result.notes.find("large gains") != std::string::npos);
}

TEST(Analysis, PbhMeasureIsZeroExactlyForUnreachableModes) {
    const control::LtiSystem system = Diagonal({-1.0, -3.0}, 1.0, {1.0, 0.0});
    // The reachable mode at -1: the augmented pencil is not rank deficient.
    CHECK_TRUE(control::PbhControllabilityMeasure(system, {-1.0, 0.0}) > 1e-3);
    // The unreachable mode at -3: it is, and the measure collapses.
    CHECK_TRUE(control::PbhControllabilityMeasure(system, {-3.0, 0.0}) < 1e-12);
    // The dual statement for observability.
    CHECK_TRUE(control::PbhObservabilityMeasure(system, {-1.0, 0.0}) > 1e-3);
    CHECK_TRUE(control::PbhObservabilityMeasure(system, {-3.0, 0.0}) < 1e-12);
}

TEST(Analysis, KalmanDecompositionCountsAndSpansTheStateSpace) {
    // One state is controllable but not observable, one is neither: the four
    // dimensions must add up to n, which is the invariant that makes the
    // decomposition usable as a coordinate change.
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -7.0;
    control::Matrix b(3, 1);
    b << 1.0, 1.0, 0.0;
    control::Matrix c(1, 3);
    c << 1.0, 2.0, 0.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);

    const control::KalmanDecomposition decomposition =
        control::DecomposeKalman(system);
    CHECK_TRUE(decomposition.spans_state_space);
    CHECK_EQ(decomposition.controllable_observable_dimension, 2);
    CHECK_EQ(decomposition.controllable_unobservable_dimension, 0);
    CHECK_EQ(decomposition.uncontrollable_observable_dimension, 0);
    CHECK_EQ(decomposition.uncontrollable_unobservable_dimension, 1);
    CHECK_EQ(decomposition.minimal_order(), 2);

    // Every basis is orthonormal, and the four together are orthonormal.
    control::Matrix all(3, 3);
    int offset = 0;
    for (const control::Matrix* block :
         {&decomposition.controllable_observable,
          &decomposition.controllable_unobservable,
          &decomposition.uncontrollable_observable,
          &decomposition.uncontrollable_unobservable}) {
        all.block(0, offset, 3, block->cols()) = *block;
        offset += static_cast<int>(block->cols());
    }
    CHECK_NEAR((all.transpose() * all - control::Matrix::Identity(3, 3))
                   .cwiseAbs()
                   .maxCoeff(),
               0.0, 1e-10);
}

TEST(Analysis, MinimalRealisationPreservesTheInputOutputBehaviour) {
    // The state at -7 is neither reachable nor visible, so it can be removed
    // without changing anything an external observer could measure.
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -7.0;
    control::Matrix b(3, 1);
    b << 1.0, 1.0, 0.0;
    control::Matrix c(1, 3);
    c << 1.0, 2.0, 0.0;
    const control::LtiSystem full = control::LtiSystem::Continuous(a, b, c);

    const control::LtiSystem reduced = control::MinimalRealisation(full);
    CHECK_EQ(reduced.state_dimension(), 2);
    CHECK_NEAR(control::DcGain(reduced)(0, 0), control::DcGain(full)(0, 0),
               1e-12);

    // "Input-output preserved" is a statement about the frequency response, so
    // that is what is checked, across three decades.
    double worst = 0.0;
    double scale = 0.0;
    for (double omega = 0.01; omega < 100.0; omega *= 1.7) {
        const Eigen::MatrixXcd reference =
            control::FrequencyResponse(full, omega);
        const Eigen::MatrixXcd candidate =
            control::FrequencyResponse(reduced, omega);
        worst = std::max(worst, std::abs(reference(0, 0) - candidate(0, 0)));
        scale = std::max(scale, std::abs(reference(0, 0)));
    }
    CHECK_TRUE(worst < 1e-10 * scale);
}

TEST(Analysis, DisposeOfTheDegenerateCase) {
    // The input reaches x1 and the sensor sees x2, so no state is both
    // reachable and visible: the model's input-output behaviour is identically
    // zero. Refusing is better than returning an empty model, and the
    // decomposition still has to account for every state.
    control::Matrix a = control::Matrix::Zero(2, 2);
    a << -1.0, 0.0, 0.0, -2.0;
    control::Matrix b(2, 1);
    b << 1.0, 0.0;
    control::Matrix c(1, 2);
    c << 0.0, 1.0;  // the input reaches x1, the sensor sees x2
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);
    const control::KalmanDecomposition decomposition =
        control::DecomposeKalman(system);
    CHECK_EQ(decomposition.controllable_observable_dimension, 0);
    CHECK_EQ(decomposition.controllable_unobservable_dimension, 1);
    CHECK_EQ(decomposition.uncontrollable_observable_dimension, 1);
    CHECK_EQ(decomposition.uncontrollable_unobservable_dimension, 0);
    CHECK_EQ(decomposition.minimal_order(), 0);
    CHECK_TRUE(decomposition.spans_state_space);
    CHECK_THROWS(control::MinimalRealisation(system));
}
