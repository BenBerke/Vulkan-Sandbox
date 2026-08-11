#pragma once

#include "Camera.hpp"
#include "renderer/VulkanRenderer.hpp"

#include <filesystem>

struct GLFWwindow;

class Application {
public:
    Application(
        std::filesystem::path executableDirectory,
        std::filesystem::path modelPath = {},
        std::filesystem::path texturePath = {}
    );

    int Run();

private:
    static constexpr int InitialWidth = 1280;
    static constexpr int InitialHeight = 720;

    std::filesystem::path executableDirectory_;
    std::filesystem::path modelPath_;
    std::filesystem::path texturePath_;

    GLFWwindow* window_{nullptr};
    VulkanRenderer renderer_;
    Camera camera_;

    Mesh mesh_;
    Texture texture_;
    Material orangeMaterial_;
    Material whiteMaterial_;
    Material blueMaterial_;

    bool mouseCaptured_{false};
    bool firstMouseSample_{true};
    double lastMouseX_{0.0};
    double lastMouseY_{0.0};

    void InitializeWindow();
    void InitializeScene();
    void MainLoop();
    void Shutdown();
    void ProcessInput(float deltaSeconds);
    void DrawScene(float elapsedSeconds);

    [[nodiscard]] std::filesystem::path FindDataDirectory(const std::string& name) const;

    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
};

