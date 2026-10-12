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

// Emits the deployable controller header for the positioning example.
//
//   generate_controller [output-path]
//
// With no argument it writes to stdout, so the output can be piped or diffed.
// This is the same generator the test suite runs before compiling the emitted
// code and comparing it against an independent implementation.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "control/analysis.h"
#include "control/codegen.h"
#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"
#include "positioning_design.h"

int main(int argc, char** argv) {
    try {
        const control::LtiSystem discrete = positioning::DiscretePlant();
        const control::AugmentedDesign design = positioning::Design();

        const control::ControllabilityResult controllability =
            control::AnalyseControllability(discrete);
        const control::ObservabilityResult observability =
            control::AnalyseObservability(discrete);
        std::cerr << "controllable: "
                  << (controllability.fully_controllable ? "yes" : "no")
                  << "  observable: "
                  << (observability.fully_observable ? "yes" : "no") << "\n";
        std::cerr << control::FormatDesignSummary(
            design.design,
            "augmented design (gain acts on [position; velocity; z]):");

        const control::CodegenResult generated =
            control::GenerateControllerWithIntegral(discrete, design,
                                                    positioning::Options());

        if (argc > 1) {
            std::ofstream output(argv[1]);
            if (!output) {
                std::cerr << "cannot write " << argv[1] << "\n";
                return 1;
            }
            output << generated.header;
            std::cerr << "wrote " << argv[1] << " (" << generated.header.size()
                      << " bytes)\n";
        } else {
            std::cout << generated.header;
        }
        return 0;
    } catch (const std::exception& problem) {
        std::cerr << "generation failed: " << problem.what() << "\n";
        return 1;
    }
}
