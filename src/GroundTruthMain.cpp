#include "GroundTruthAnnotator.hpp"
#include "HaifaSamples.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./GroundTruth <sample_number: 1|2|3>" << std::endl;
        return 1;
    }

    int sample_num = std::atoi(argv[1]);
    auto samples = getHaifaSamples();
    if (sample_num < 1 || sample_num > static_cast<int>(samples.size())) {
        std::cerr << "Invalid sample number. Choose 1-" << samples.size() << "." << std::endl;
        return 1;
    }

    runGroundTruthAnnotator(samples[sample_num - 1]);
    return 0;
}
