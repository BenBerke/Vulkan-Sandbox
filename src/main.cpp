#include "Application.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path executableDirectory =
            std::filesystem::absolute(argv[0]).parent_path();

        const std::filesystem::path modelPath = argc >= 2 ? argv[1] : "";
        const std::filesystem::path texturePath = argc >= 3 ? argv[2] : "";

        Application application(executableDirectory, modelPath, texturePath);
        return application.Run();
    } catch (const std::exception& exception) {
        std::cerr << "Fatal error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}

