#include "Application.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

Application::Application(
    std::filesystem::path executableDirectory,
    std::filesystem::path modelPath,
    std::filesystem::path texturePath
)
    : executableDirectory_(std::move(executableDirectory)),
      modelPath_(std::move(modelPath)),
      texturePath_(std::move(texturePath)) {}

int Application::Run() {
    try {
        InitializeWindow();
        InitializeScene();
        MainLoop();
        Shutdown();
        return 0;
    } catch (...) {
        Shutdown();
        throw;
    }
}

void Application::InitializeWindow() {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("GLFW could not be initialized");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(
        InitialWidth,
        InitialHeight,
        "Usable Vulkan Renderer",
        nullptr,
        nullptr
    );

    if (window_ == nullptr) {
        glfwTerminate();
        throw std::runtime_error("GLFW could not create the window");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, FramebufferResizeCallback);
}

void Application::InitializeScene() {
    const std::filesystem::path shaderDirectory = FindDataDirectory("shaders");
    const std::filesystem::path assetDirectory = FindDataDirectory("assets");

    if (modelPath_.empty()) {
        modelPath_ = assetDirectory / "models" / "cube.obj";
    }
    if (texturePath_.empty()) {
        texturePath_ = assetDirectory / "textures" / "checker.ppm";
    }

    renderer_.Initialize(window_, shaderDirectory);
    mesh_ = renderer_.LoadMesh(modelPath_);
    texture_ = renderer_.LoadTexture(texturePath_);

    orangeMaterial_ = renderer_.CreateMaterial(texture_, {1.0f, 0.45f, 0.15f, 1.0f});
    whiteMaterial_ = renderer_.CreateMaterial(texture_, {1.0f, 1.0f, 1.0f, 1.0f});
    blueMaterial_ = renderer_.CreateMaterial(texture_, {0.2f, 0.55f, 1.0f, 1.0f});
}

void Application::MainLoop() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto previous = start;

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        glfwPollEvents();

        const auto now = Clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previous).count();
        const float elapsedSeconds = std::chrono::duration<float>(now - start).count();
        previous = now;

        ProcessInput(std::min(deltaSeconds, 0.1f));
        DrawScene(elapsedSeconds);
    }
}

void Application::ProcessInput(const float deltaSeconds) {
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }

    constexpr float movementSpeed = 3.5f;
    const float distance = movementSpeed * deltaSeconds;

    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) camera_.position += camera_.Forward() * distance;
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) camera_.position -= camera_.Forward() * distance;
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) camera_.position += camera_.Right() * distance;
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) camera_.position -= camera_.Right() * distance;
    if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) camera_.position.y += distance;
    if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) camera_.position.y -= distance;

    const bool shouldCaptureMouse = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (shouldCaptureMouse != mouseCaptured_) {
        mouseCaptured_ = shouldCaptureMouse;
        firstMouseSample_ = true;
        glfwSetInputMode(window_, GLFW_CURSOR, mouseCaptured_ ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    if (!mouseCaptured_) return;

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(window_, &mouseX, &mouseY);

    if (firstMouseSample_) {
        lastMouseX_ = mouseX;
        lastMouseY_ = mouseY;
        firstMouseSample_ = false;
        return;
    }

    constexpr float sensitivity = 0.1f;
    camera_.yawDegrees += static_cast<float>(mouseX - lastMouseX_) * sensitivity;
    camera_.pitchDegrees -= static_cast<float>(mouseY - lastMouseY_) * sensitivity;
    camera_.pitchDegrees = std::clamp(camera_.pitchDegrees, -89.0f, 89.0f);

    lastMouseX_ = mouseX;
    lastMouseY_ = mouseY;
}

void Application::DrawScene(const float elapsedSeconds) {
    if (!renderer_.BeginFrame(camera_)) return;

    const glm::mat4 left =
        glm::translate(glm::mat4{1.0f}, {-1.4f, 0.0f, 0.0f}) *
        glm::rotate(glm::mat4{1.0f}, elapsedSeconds, {0.0f, 1.0f, 0.0f});

    const glm::mat4 middle =
        glm::rotate(glm::mat4{1.0f}, -elapsedSeconds * 0.7f, {1.0f, 1.0f, 0.0f});

    const glm::mat4 right =
        glm::translate(glm::mat4{1.0f}, {1.4f, 0.0f, 0.0f}) *
        glm::rotate(glm::mat4{1.0f}, elapsedSeconds * 1.3f, {0.0f, 1.0f, 1.0f});

    renderer_.DrawMesh(mesh_, orangeMaterial_, left);
    renderer_.DrawMesh(mesh_, whiteMaterial_, middle);
    renderer_.DrawMesh(mesh_, blueMaterial_, right);
    renderer_.EndFrame();
}

std::filesystem::path Application::FindDataDirectory(const std::string& name) const {
    const std::filesystem::path besideExecutable = executableDirectory_ / name;
    if (std::filesystem::is_directory(besideExecutable)) return besideExecutable;

    const std::filesystem::path inWorkingDirectory = std::filesystem::current_path() / name;
    if (std::filesystem::is_directory(inWorkingDirectory)) return inWorkingDirectory;

    throw std::runtime_error(
        "Could not find the '" + name + "' directory beside the executable or in the working directory"
    );
}

void Application::Shutdown() {
    renderer_.Shutdown();

    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
}

void Application::FramebufferResizeCallback(GLFWwindow* window, int, int) {
    auto* application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (application != nullptr) application->renderer_.NotifyFramebufferResized();
}

