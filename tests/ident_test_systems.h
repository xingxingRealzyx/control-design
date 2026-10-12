// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#ifndef CONTROL_TESTS_IDENT_TEST_SYSTEMS_H_
#define CONTROL_TESTS_IDENT_TEST_SYSTEMS_H_

// Shared fixtures: a known system, and data generated from it.
//
// Every identification test follows the same shape - generate data from a
// system whose parameters are known exactly, identify, and compare against
// those parameters. That is the only way to test an identification method:
// comparing against a previously recorded output would lock in whatever the
// method happened to do on the day, bugs included.

#include <Eigen/Core>
#include <cmath>

#include "control/ident/dataset.h"
#include "control/ident/excitation.h"
#include "control/ident/ls.h"

namespace ident_test {

// A deterministic normal generator, written out rather than taken from the
// standard library.
//
// `std::mt19937` is specified exactly, but `std::normal_distribution` is not:
// its algorithm is implementation-defined, so the "same" test data is a
// different sequence on libstdc++ (Linux CI) than on libc++ (macOS). The
// statistical tests below then land on different sides of their thresholds on
// the two platforms, which is what happened - two validation tests passed
// locally and failed on the Linux runners with no code difference between them.
// Test data that depends on implementation-defined behaviour is not
// reproducible, and a statistical test whose data is not reproducible is not a
// test.
class DeterministicNormal {
 public:
    explicit DeterministicNormal(unsigned int seed) : state_(seed | 1u) {}

    double Next() {
        const double u1 = NextUniform();
        const double u2 = NextUniform();
        return std::sqrt(-2.0 * std::log(u1)) *
               std::cos(2.0 * 3.14159265358979323846 * u2);
    }

 private:
    // xorshift64, not a linear congruential generator.
    //
    // An LCG was tried first and its lattice structure left enough low-order
    // correlation for a 20000-sample portmanteau test to detect it: the
    // whiteness test then failed on the "white" data, which is a true statement
    // about the generator and a useless statement about the test. A generator
    // used to produce reference noise has to be good enough that a whiteness
    // test on its output passes, otherwise it is testing the generator.
    double NextUniform() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return (static_cast<double>(state_ >> 11) + 0.5) / 9007199254740992.0;
    }
    unsigned long long state_;
};

// y[k] = 1.5 y[k-1] - 0.7 y[k-2] + 0.5 u[k-1] + 0.2 u[k-2]
// Poles: roots of z^2 - 1.5z + 0.7, at 0.75 +/- 0.37j, |z| = 0.838.
struct KnownSystem {
    double a1 = -1.5;
    double a2 = 0.7;
    double b1 = 0.5;
    double b2 = 0.2;
    double sample_time = 0.1;
};

inline control::ident::ArxModel Truth(const KnownSystem& known) {
    control::ident::ArxModel model;
    model.a.resize(2);
    model.a << known.a1, known.a2;
    model.b.resize(2);
    model.b << known.b1, known.b2;
    model.delay = 1;
    model.sample_time = known.sample_time;
    return model;
}

// Simulates the truth exactly (no state-space conversion, so the test data does
// not depend on the code being tested) with optional white noise on the output.
inline control::ident::Dataset GenerateData(
    const KnownSystem& known, const Eigen::VectorXd& input,
    double noise_standard_deviation = 0.0, unsigned int seed = 12345u) {
    control::ident::Dataset data;
    data.sample_time = known.sample_time;
    data.input = input;
    data.input_name = "u";
    data.output_name = "y";
    const int n = static_cast<int>(input.size());
    data.output = Eigen::VectorXd::Zero(n);
    DeterministicNormal generator(seed);
    for (int k = 0; k < n; ++k) {
        double value = 0.0;
        if (k >= 1) {
            value += known.b1 * input(k - 1);
        }
        if (k >= 2) {
            value += known.b2 * input(k - 2) - known.a1 * data.output(k - 1) -
                     known.a2 * data.output(k - 2);
        } else if (k == 1) {
            value += -known.a1 * data.output(0);
        }
        data.output(k) = value + noise_standard_deviation * generator.Next();
    }
    return data;
}

// A record long enough for two poles and rich enough to identify them.
inline control::ident::Dataset MakeTrainingData(const KnownSystem& known,
                                                int samples = 2000,
                                                double noise = 0.0,
                                                unsigned int seed = 12345u) {
    control::ident::PrbsOptions prbs;
    prbs.order = 11;
    prbs.amplitude = 1.0;
    prbs.samples_per_bit = 3;
    prbs.bits = samples / 3 + 1;
    prbs.seed = 0x1234u;
    Eigen::VectorXd input = control::ident::GeneratePrbs(prbs);
    if (input.size() > samples) {
        // Materialise the head before assigning: `input = input.head(n)` is
        // self-aliasing, and Eigen resizes the destination - invalidating the
        // source view - before copying, so it reads freed memory. The symptom
        // is not a crash but a test suite whose pass count changes between runs
        // of the same binary (measured: 19, 15 and 22 of 29 across three runs),
        // which is worse than a crash because it looks like flakiness in the
        // code under test. AddressSanitizer pointed straight at it.
        input = Eigen::VectorXd(input.head(samples));
    }
    return GenerateData(known, input, noise, seed);
}

}  // namespace ident_test

#endif  // CONTROL_TESTS_IDENT_TEST_SYSTEMS_H_
