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

#ifndef CONTROL_IDENT_EXCITATION_H_
#define CONTROL_IDENT_EXCITATION_H_

// Designing the experiment, which decides the outcome more than the estimator
// does.
//
// No algorithm can identify what the data does not contain. A step test excites
// one frequency and gives you a static gain; a relay test gives you the
// critical point; a pseudo-random binary sequence gives you everything up to
// its clock rate. The most common reason an identification fails is not a wrong
// method but an input that never moved the modes being identified - and that
// failure is diagnosable *before* the experiment, from the regressor's
// condition number.
//
// This header provides the standard signals and that diagnosis.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/ident/dataset.h"

namespace control {
namespace ident {

struct PrbsOptions {
    // Register length: the sequence repeats after 2^order - 1 bits.
    int order = 10;
    double amplitude = 1.0;
    double offset = 0.0;
    // Samples held per bit. The clock rate sets the bandwidth: the sequence has
    // power up to roughly half the bit rate, so a bit period of T excites the
    // plant usefully up to about 1/(2T) Hz.
    int samples_per_bit = 1;
    unsigned int seed = 0x5eedu;
    // Number of bits to generate; 0 means one full period.
    int bits = 0;
};

// Pseudo-random binary sequence from a maximal-length linear feedback shift
// register. The periodicity is a feature - repeating it averages out noise -
// and a hazard, because a plant that drifts on the same timescale is identified
// as part of the periodic response.
Eigen::VectorXd GeneratePrbs(const PrbsOptions& options);

struct ChirpOptions {
    double start_frequency = 0.01;  // Hz
    double end_frequency = 1.0;     // Hz
    double amplitude = 1.0;
    double duration = 0.0;  // seconds
    double sample_time = 0.0;
    // Logarithmic sweep spends equal time per decade, which is what a frequency
    // response measurement wants. A linear sweep spends most of its time at
    // high frequency and is the wrong choice for a plant with a low-frequency
    // pole.
    bool logarithmic = true;
};

Eigen::VectorXd GenerateChirp(const ChirpOptions& options);

struct MultisineOptions {
    int components = 10;
    double start_frequency = 0.01;
    double end_frequency = 1.0;
    double amplitude = 1.0;  // total, split across components
    double duration = 0.0;
    double sample_time = 0.0;
    // Schroeder phasing spreads the crest factor: a multisine with in-phase
    // components has a peak many times its RMS, which drives the plant into
    // saturation and identifies a nonlinearity instead of a plant.
    bool schroeder_phasing = true;
};

Eigen::VectorXd GenerateMultisine(const MultisineOptions& options);

// Is this input rich enough to identify a model of this size?
//
// The measure is the condition number of the regressor matrix that the
// estimator will build: it says how much the parameters have to be trusted
// individually, independently of any fitting. A condition number of 1e3 is
// comfortable, 1e6 is marginal, 1e10 means the answer is decided by round-off.
struct PersistencyResult {
    double condition_number = 0.0;
    double smallest_singular_value = 0.0;
    double largest_singular_value = 0.0;
    int parameters = 0;
    int samples = 0;
    bool persistently_exciting = false;
    // The largest order this input *can* identify, found by raising the order
    // until the condition number crosses the threshold. This is the number to
    // act on: it converts "excite it more" into "you have enough for a
    // 3rd-order model, not a 6th".
    int identifiable_order = 0;
    std::string notes;
};

PersistencyResult CheckPersistency(const Eigen::VectorXd& input,
                                   int denominator_order, int numerator_order,
                                   int delay = 1,
                                   double condition_threshold = 1e6);

// The smallest input amplitude that produces a distinguishable output, given
// the noise level and the output range available. A signal below the noise
// identifies noise; one that saturates the actuator identifies the saturation.
struct ExcitationAdvice {
    double recommended_amplitude = 0.0;
    double expected_output_swing = 0.0;
    double suggested_samples = 0;
    std::vector<std::string> warnings;
    std::string notes;
};
ExcitationAdvice AdviseExcitation(double dc_gain,
                                  double noise_standard_deviation,
                                  double output_limit, double actuator_limit,
                                  double settling_time, double sample_time);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_EXCITATION_H_
