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

#ifndef CONTROL_DISCRETIZE_H_
#define CONTROL_DISCRETIZE_H_

// Continuous -> discrete conversion, and the checks that decide whether the
// answer is usable.
//
// Two things this header refuses to do quietly:
//
//   * It will not pick a sample time for you. A sample rate is a design
//     decision with a rule attached (see `RequiredSampleTime`), and hiding it
//     inside a conversion call is how people end up with a 50 Hz loop on a
//     system with a 60 Hz bandwidth.
//   * It will not pretend the conversion is exact when it is not. `Convert`
//     returns a report that says which method was used and how much the
//     discrete model deviates from the continuous one, so "Tustin at 50 Hz"
//     stops being an act of faith.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/types.h"

namespace control {

struct DiscretizationOptions {
    DiscretizationMethod method = DiscretizationMethod::kZeroOrderHold;
    // For kMatchedPoleZero: match the gain at DC (default) or at the Nyquist
    // frequency. DC matching preserves the steady-state behaviour, which is
    // almost always what a controller design wants.
    bool match_gain_at_dc = true;
    // Poles closer to the origin than this are treated as being at the origin
    // when matching, which avoids dividing by a numerically zero pole.
    double pole_flush_tolerance = 1e-9;
};

struct DiscretizationReport {
    DiscretizationMethod method = DiscretizationMethod::kZeroOrderHold;
    double sample_time = 0.0;
    // Largest relative deviation between the discrete model's step response and
    // the continuous one, sampled at the sample instants, over the settling
    // window. This is a direct, physical measure of "did the conversion keep
    // the behaviour", unlike comparing matrix entries.
    double step_response_deviation = 0.0;
    // True when every continuous pole landed inside the unit circle.
    bool all_poles_mapped_inside_unit_circle = true;
    // Poles of the continuous model that the chosen method moved outside the
    // unit circle (forward Euler can do this). Empty in the normal case.
    std::vector<std::complex<double>> destabilized_poles;
    bool exact_for_constant_input = false;  // true only for zero-order hold
    std::string notes;
};

struct DiscretizationResult {
    LtiSystem system;  // discrete-time model (sample_time set)
    DiscretizationReport report;
    // True when the conversion is exact at the sample instants for the input
    // class the method assumes. Zero-order hold is exact; Euler methods are
    // not.
    bool ok = false;
};

// Converts a continuous model. Throws std::invalid_argument when the model is
// already discrete, has a non-positive sample time, or the method is not
// applicable (matched pole-zero needs a strictly proper SISO model).
DiscretizationResult Convert(const LtiSystem& continuous, double sample_time,
                             const DiscretizationOptions& options = {});

// Convenience wrapper keeping only the model.
LtiSystem Discretize(
    const LtiSystem& continuous, double sample_time,
    DiscretizationMethod method = DiscretizationMethod::kZeroOrderHold);

// Sample time that keeps a discretized design honest: the rule of thumb is
// 20 samples per closed-loop rise time, or equivalently ws >= 20 * bandwidth.
// `bandwidth` is in rad/s; the return value is in seconds.
//
// Prefer this over picking a number: a sample time that is too slow is the most
// common reason a design that worked in continuous time fails on hardware, and
// it fails in a way that looks like a tuning problem.
double RequiredSampleTime(double bandwidth,
                          double samples_per_bandwidth = 20.0);

// Continuous step response and the discrete one, sampled at the same instants,
// for the deviation measure above and for the tests. `points` samples spaced by
// `sample_time`, starting at t = 0.
struct StepResponseComparison {
    Vector time;
    Matrix continuous;  // outputs x points
    Matrix discrete;
    double max_relative_deviation = 0.0;
};
StepResponseComparison CompareStepResponses(const LtiSystem& continuous,
                                            const LtiSystem& discrete,
                                            int points);

// Padé approximation of a pure input delay exp(-s T), as a state-space factor
// for use with the design functions. `order` is 1..3; order 1 is the classic
// (1 - sT/2) / (1 + sT/2) and is usually enough for a design model.
//
// A delay is not a rational function, so this is an approximation by
// construction: the returned system reports the delay it approximates.
LtiSystem PadeDelay(double delay, int order = 1);

// Series connection: the input of `second` is the output of `first`. Used to
// put a delay in front of a plant.
LtiSystem Series(const LtiSystem& first, const LtiSystem& second);

}  // namespace control

#endif  // CONTROL_DISCRETIZE_H_
