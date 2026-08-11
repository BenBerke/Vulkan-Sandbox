#pragma once

#include "Camera.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>

struct GLFWwindow;

namespace RendererHandle {
inline constexpr std::uint32_t Invalid = UINT32_MAX;
}

struct Mesh {
    std::uint32_t id{RendererHandle::Invalid};
    [[nodiscard]] explicit operator bool() const { return id != RendererHandle::Invalid; }
};

struct Texture {
    std::uint32_t id{RendererHandle::Invalid};
    [[nodiscard]] explicit operator bool() const { return id != RendererHandle::Invalid; }
};

struct Material {
    std::uint32_t id{RendererHandle::Invalid};
    [[nodiscard]] explicit operator bool() const { return id != RendererHandle::Invalid; }
};

class VulkanRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;
    VulkanRenderer(VulkanRenderer&&) = delete;
    VulkanRenderer& operator=(VulkanRenderer&&) = delete;

    void Initialize(GLFWwindow* window, const std::filesystem::path& shaderDirectory);
    void Shutdown() noexcept;

    [[nodiscard]] Mesh LoadMesh(const std::filesystem::path& path);
    [[nodiscard]] Texture LoadTexture(const std::filesystem::path& path);
    [[nodiscard]] Material CreateMaterial(
        const Texture& texture,
        const glm::vec4& color = glm::vec4(1.0f)
    );

    void Destroy(Mesh& mesh);
    void Destroy(Texture& texture);
    void Destroy(Material& material);

    [[nodiscard]] bool BeginFrame(const Camera& camera);
    void DrawMesh(
        const Mesh& mesh,
        const Material& material,
        const glm::mat4& transform
    );
    void EndFrame();

    void NotifyFramebufferResized() noexcept;
    [[nodiscard]] bool IsInitialized() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

