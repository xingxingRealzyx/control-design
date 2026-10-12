// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Dense>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "control/ident/excitation.h"

namespace control {
namespace ident {
namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

Eigen::VectorXd GeneratePrbs(const PrbsOptions& options) {
    if (options.order < 2 || options.order > 30) {
        throw std::invalid_argument(
            "GeneratePrbs: the register length must be between 2 and 30");
    }
    if (options.samples_per_bit < 1) {
        throw std::invalid_argument(
            "GeneratePrbs: samples_per_bit must be at least one");
    }
    const long long period = (1LL << options.order) - 1;
    const long long bits = options.bits > 0 ? options.bits : period;
    if (bits < 2 * options.order) {
        throw std::invalid_argument(
            "GeneratePrbs: fewer than two register lengths of bits identifies "
            "nothing; "
            "each parameter needs several independent transitions");
    }

    // Maximal-length sequence from a Fibonacci linear feedback shift register.
    // The tap set is the standard one; the sequence visits every non-zero
    // state, which is what makes it "maximal length" and gives it a flat
    // spectrum.
    const unsigned int taps[][4] = {
        {0, 0, 0, 0},     {0, 0, 0, 0},   {2, 1, 0, 0},   {3, 2, 0, 0},
        {4, 3, 0, 0},     {5, 3, 0, 0},   {6, 5, 0, 0},   {7, 6, 0, 0},
        {8, 6, 5, 4},     {9, 5, 0, 0},   {10, 7, 0, 0},  {11, 9, 0, 0},
        {12, 6, 4, 1},    {13, 4, 3, 1},  {14, 5, 3, 1},  {15, 14, 0, 0},
        {16, 15, 13, 4},  {17, 14, 0, 0}, {18, 11, 0, 0}, {19, 6, 2, 1},
        {20, 17, 0, 0},   {21, 19, 0, 0}, {22, 21, 0, 0}, {23, 18, 0, 0},
        {24, 23, 22, 17}, {25, 22, 0, 0}, {26, 6, 2, 1},  {27, 5, 2, 1},
        {28, 25, 0, 0},   {29, 27, 0, 0}, {30, 6, 4, 1}};
    const int order = options.order;
    unsigned int state = options.seed == 0 ? 0x5eedu : options.seed;
    if (state == 0) {
        state = 1;
    }
    state &= (1u << order) - 1u;
    if (state == 0) {
        state = 1;
    }

    Eigen::VectorXd signal(bits * options.samples_per_bit);
    for (long long bit = 0; bit < bits; ++bit) {
        const unsigned int feedback =
            ((state >> (taps[order][0] - 1)) ^ (state >> (taps[order][1] - 1)) ^
             (taps[order][2] ? (state >> (taps[order][2] - 1)) : 0u) ^
             (taps[order][3] ? (state >> (taps[order][3] - 1)) : 0u)) &
            1u;
        state = ((state << 1) | feedback) & ((1u << order) - 1u);
        const double level = feedback ? options.amplitude : -options.amplitude;
        for (int repeat = 0; repeat < options.samples_per_bit; ++repeat) {
            signal(bit * options.samples_per_bit + repeat) =
                level + options.offset;
        }
    }
    return signal;
}

Eigen::VectorXd GenerateChirp(const ChirpOptions& options) {
    if (!(options.sample_time > 0.0) || !(options.duration > 0.0)) {
        throw std::invalid_argument(
            "GenerateChirp: the sample time and duration must be positive");
    }
    if (!(options.start_frequency > 0.0) ||
        !(options.end_frequency > options.start_frequency)) {
        throw std::invalid_argument(
            "GenerateChirp: the end frequency must exceed the start frequency, "
            "and "
            "both must be positive (a sweep through zero has no meaningful "
            "phase)");
    }
    const int samples =
        static_cast<int>(options.duration / options.sample_time);
    const double f0 = options.start_frequency;
    const double f1 = options.end_frequency;
    const double sweep = f1 / f0;
    Eigen::VectorXd signal(samples);
    for (int k = 0; k < samples; ++k) {
        const double t = k * options.sample_time;
        double phase = 0.0;
        if (options.logarithmic) {
            // Phase of a logarithmic sweep: the instantaneous frequency is
            // f0 * sweep^(t/T), whose integral is the expression below.
            const double duration = options.duration;
            phase = 2.0 * kPi * f0 * duration / std::log(sweep) *
                    (std::pow(sweep, t / duration) - 1.0);
        } else {
            phase = 2.0 * kPi *
                    (f0 * t + 0.5 * (f1 - f0) / options.duration * t * t);
        }
        signal(k) = options.amplitude * std::sin(phase);
    }
    return signal;
}

Eigen::VectorXd GenerateMultisine(const MultisineOptions& options) {
    if (options.components < 1) {
        throw std::invalid_argument(
            "GenerateMultisine: at least one component is needed");
    }
    if (!(options.sample_time > 0.0) || !(options.duration > 0.0)) {
        throw std::invalid_argument(
            "GenerateMultisine: the sample time and duration must be positive");
    }
    if (!(options.start_frequency > 0.0) ||
        !(options.end_frequency > options.start_frequency)) {
        throw std::invalid_argument(
            "GenerateMultisine: the end frequency must exceed the start "
            "frequency");
    }
    const int samples =
        static_cast<int>(options.duration / options.sample_time);
    const double per_component =
        options.amplitude / std::sqrt(static_cast<double>(options.components));
    Eigen::VectorXd signal = Eigen::VectorXd::Zero(samples);
    for (int component = 0; component < options.components; ++component) {
        const double fraction =
            options.components > 1
                ? static_cast<double>(component) / (options.components - 1)
                : 0.0;
        // Frequencies spaced logarithmically: linear spacing crowds the high
        // end and starves the low end, which is where the interesting dynamics
        // are.
        const double frequency =
            options.start_frequency *
            std::pow(options.end_frequency / options.start_frequency, fraction);
        // Schroeder phasing: the phases are chosen so the components do not
        // line up, which cuts the crest factor. In phase, a ten-component
        // multisine peaks at ten times its per-component amplitude and pushes
        // the actuator into saturation - identifying the saturation rather than
        // the plant.
        const double phase =
            options.schroeder_phasing ? -kPi * fraction * (fraction) : 0.0;
        for (int k = 0; k < samples; ++k) {
            signal(k) += per_component * std::sin(2.0 * kPi * frequency * k *
                                                      options.sample_time +
                                                  phase);
        }
    }
    return signal;
}

PersistencyResult CheckPersistency(const Eigen::VectorXd& input,
                                   int denominator_order, int numerator_order,
                                   int delay, double condition_threshold) {
    if (denominator_order < 1 || numerator_order < 1) {
        throw std::invalid_argument(
            "CheckPersistency: the orders must be at least one");
    }
    const int parameters = denominator_order + numerator_order;
    // The regressor is built from `parameters` lagged copies of the input, so the
    // first usable index has to leave that many past samples. The usual ARX bound,
    // max(na, delay + nb - 1), is not enough: with na = nb = 3 it gives 3, and the
    // regressor then asks for input[index - 5] on its first row - a negative index.
    // In a release build that reads whatever is in front of the array, silently, and
    // the condition number reported here was different on different platforms
    // because it was measuring that garbage. A debug build and AddressSanitizer both
    // stop at it; the release build does not.
    const int first = std::max(parameters, std::max(denominator_order, delay + numerator_order - 1));
    PersistencyResult result;
    result.parameters = parameters;
    result.samples = static_cast<int>(input.size()) - first;
    if (result.samples < result.parameters) {
        result.notes = "not enough samples for the requested order: " +
                       std::to_string(static_cast<int>(input.size())) +
                       " samples for " + std::to_string(result.parameters) +
                       " parameters";
        return result;
    }

    // The diagnosis is about the *input*: an input is persistently exciting of
    // order n when its n lagged copies are linearly independent, which is the
    // positive-definiteness of its autocorrelation matrix. So the regressor
    // here is built from the input alone, as n lags of it.
    //
    // An earlier version filled the "output" block with the input as well, on
    // the reasoning that no output was available. That makes the two blocks
    // nearly identical and the condition number enormous for *every* input -
    // 1e32 for a pseudo-random sequence as well as for a step - so the check
    // could not distinguish anything, which the test caught by asserting that a
    // PRBS passes where a step fails.
    Eigen::MatrixXd regressors(result.samples, result.parameters);
    for (int row = 0; row < result.samples; ++row) {
        const int index = first + row;
        for (int lag = 0; lag < result.parameters; ++lag) {
            regressors(row, lag) = input(index - lag);
        }
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(regressors);
    const Eigen::VectorXd& singular = svd.singularValues();
    result.largest_singular_value = singular.size() > 0 ? singular(0) : 0.0;
    result.smallest_singular_value =
        singular.size() > 0 ? singular(singular.size() - 1) : 0.0;
    result.condition_number =
        result.smallest_singular_value > 0.0
            ? result.largest_singular_value / result.smallest_singular_value
            : std::numeric_limits<double>::infinity();
    result.persistently_exciting =
        result.condition_number < condition_threshold;

    // The order this input can actually support: lower the order until the
    // condition number is acceptable. This is the actionable form of the answer
    // - "excite it more" becomes "you have enough for a 3rd-order model, not a
    // 6th".
    for (int candidate = result.parameters; candidate >= 1; --candidate) {
        Eigen::MatrixXd reduced(result.samples, candidate);
        reduced = regressors.leftCols(candidate);
        const Eigen::JacobiSVD<Eigen::MatrixXd> reduced_svd(reduced);
        const Eigen::VectorXd& reduced_singular = reduced_svd.singularValues();
        const double smallest = reduced_singular(reduced_singular.size() - 1);
        if (smallest > 0.0 &&
            reduced_singular(0) / smallest < condition_threshold) {
            result.identifiable_order = candidate;
            break;
        }
    }

    std::ostringstream notes;
    notes << "regressor condition number " << result.condition_number << " for "
          << result.parameters << " parameters over " << result.samples
          << " samples";
    if (result.persistently_exciting) {
        notes << ": the input supports this order";
    } else {
        notes
            << ": the input does NOT support this order. It can identify up to "
            << result.identifiable_order
            << " parameter(s). Widen the frequency content (a pseudo-random "
               "sequence or a chirp rather than a step) or reduce the order";
    }
    result.notes = notes.str();
    return result;
}

ExcitationAdvice AdviseExcitation(double dc_gain,
                                  double noise_standard_deviation,
                                  double output_limit, double actuator_limit,
                                  double settling_time, double sample_time) {
    if (!(sample_time > 0.0) || !(settling_time > 0.0)) {
        throw std::invalid_argument(
            "AdviseExcitation: the sample time and settling time must be "
            "positive");
    }
    ExcitationAdvice advice;
    // The input has to be large enough that the output moves well clear of the
    // noise, and small enough not to saturate. Ten times the noise is the usual
    // working figure: below that the signal-to-noise ratio makes the fit
    // meaningless, above it there is nothing to gain.
    const double minimum_output_swing =
        10.0 * std::max(noise_standard_deviation, 1e-12);
    const double gain = std::abs(dc_gain) > 1e-12 ? std::abs(dc_gain) : 1e-12;
    advice.recommended_amplitude = minimum_output_swing / gain;
    advice.expected_output_swing = advice.recommended_amplitude * gain;

    if (advice.recommended_amplitude > actuator_limit) {
        advice.recommended_amplitude = actuator_limit;
        advice.expected_output_swing = actuator_limit * gain;
        advice.warnings.push_back(
            "the actuator limit is reached before the output clears the noise "
            "by a "
            "factor of ten: at this signal-to-noise ratio the identified "
            "parameters "
            "will be noisy. Average several periods of a periodic input, or "
            "reduce "
            "the noise");
    }
    if (advice.expected_output_swing > output_limit) {
        advice.warnings.push_back(
            "the recommended amplitude drives the output to " +
            std::to_string(advice.expected_output_swing) +
            ", beyond the output limit of " + std::to_string(output_limit) +
            ": the plant will saturate and what is identified will be the "
            "saturation");
    }
    // A record long enough to contain many settling times: the low-frequency
    // content that identifies the dominant pole only appears over that scale.
    const double suggested_duration = 20.0 * settling_time;
    advice.suggested_samples =
        static_cast<int>(suggested_duration / sample_time);

    std::ostringstream notes;
    notes
        << "aim for an input amplitude of " << advice.recommended_amplitude
        << " (an output swing of " << advice.expected_output_swing
        << ") over about " << advice.suggested_samples << " samples ("
        << suggested_duration
        << " s, twenty settling times). The low-frequency "
           "content that identifies a slow pole only shows up over that scale, "
           "which is why a short record produces a model with the right high-"
           "frequency behaviour and the wrong DC gain";
    advice.notes = notes.str();
    return advice;
}

}  // namespace ident
}  // namespace control
