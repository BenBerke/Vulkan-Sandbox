#include "renderer/VulkanRenderer.hpp"

#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <tiny_obj_loader.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t MaxFramesInFlight = 2;
constexpr std::uint32_t MaxMaterials = 256;

#ifdef NDEBUG
constexpr bool EnableValidationLayers = false;
#else
constexpr bool EnableValidationLayers = true;
#endif

const std::vector<const char*> ValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

struct Vertex {
    glm::vec3 position{};
    glm::vec3 color{1.0f};
    glm::vec2 textureCoordinate{};

    static vk::VertexInputBindingDescription BindingDescription() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> AttributeDescriptions() {
        return {{
            {
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, position)
            },
            {
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, color)
            },
            {
                .location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, textureCoordinate)
            }
        }};
    }

    bool operator==(const Vertex&) const = default;
};

inline void HashCombine(std::size_t& seed, const float value) {
    seed ^= std::hash<float>{}(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

struct VertexHash {
    std::size_t operator()(const Vertex& vertex) const noexcept {
        std::size_t seed = 0;
        HashCombine(seed, vertex.position.x);
        HashCombine(seed, vertex.position.y);
        HashCombine(seed, vertex.position.z);
        HashCombine(seed, vertex.color.r);
        HashCombine(seed, vertex.color.g);
        HashCombine(seed, vertex.color.b);
        HashCombine(seed, vertex.textureCoordinate.x);
        HashCombine(seed, vertex.textureCoordinate.y);
        return seed;
    }
};

struct CameraUniform {
    alignas(16) glm::mat4 view{1.0f};
    alignas(16) glm::mat4 projection{1.0f};
};

struct PushConstants {
    alignas(16) glm::mat4 model{1.0f};
    alignas(16) glm::vec4 color{1.0f};
};

static_assert(sizeof(PushConstants) <= 128, "Push constants exceed Vulkan's guaranteed minimum");

struct BufferAllocation {
    // Memory is declared first so the buffer is destroyed before its bound memory.
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
};

struct ImageAllocation {
    // Memory is declared first so the image is destroyed before its bound memory.
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
};

struct MeshResource {
    BufferAllocation vertexBuffer;
    BufferAllocation indexBuffer;
    std::uint32_t indexCount{0};
};

struct TextureResource {
    ImageAllocation image;
    vk::raii::ImageView view = nullptr;
};

struct MaterialResource {
    Texture texture;
    glm::vec4 color{1.0f};
    std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct DrawCommand {
    Mesh mesh;
    Material material;
    glm::mat4 transform{1.0f};
};

std::vector<char> ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open shader: " + path.string());
    }

    const auto fileSize = file.tellg();
    if (fileSize <= 0 || (fileSize % 4) != 0) {
        throw std::runtime_error("Shader is empty or is not valid SPIR-V: " + path.string());
    }

    std::vector<char> bytes(static_cast<std::size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    if (!file) {
        throw std::runtime_error("Could not read shader: " + path.string());
    }
    return bytes;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*
) {
    std::cerr << "Vulkan validation: " << callbackData->pMessage << '\n';
    return VK_FALSE;
}

} // namespace

class VulkanRenderer::Impl {
public:
    GLFWwindow* window{nullptr};
    std::filesystem::path shaderDirectory;
    bool initialized{false};
    bool framebufferResized{false};
    bool frameInProgress{false};
    bool acquiredSuboptimalSwapchain{false};

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue queue = nullptr;
    std::uint32_t queueFamilyIndex = std::numeric_limits<std::uint32_t>::max();

    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> swapchainImages;
    vk::SurfaceFormatKHR swapchainSurfaceFormat{};
    vk::Extent2D swapchainExtent{};
    std::vector<vk::raii::ImageView> swapchainImageViews;

    ImageAllocation depthImage;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    // Memory precedes buffers so buffers are destroyed first during normal unwinding.
    std::vector<vk::raii::DeviceMemory> cameraUniformMemory;
    std::vector<vk::raii::Buffer> cameraUniformBuffers;
    std::vector<void*> cameraUniformMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::Sampler textureSampler = nullptr;

    std::vector<std::optional<MeshResource>> meshes;
    std::vector<std::optional<TextureResource>> textures;
    std::vector<std::optional<MaterialResource>> materials;
    std::vector<DrawCommand> drawCommands;

    std::uint32_t frameIndex{0};
    std::uint32_t acquiredImageIndex{0};

    void Initialize(GLFWwindow* targetWindow, const std::filesystem::path& targetShaderDirectory) {
        if (initialized) throw std::logic_error("VulkanRenderer is already initialized");
        if (targetWindow == nullptr) throw std::invalid_argument("VulkanRenderer needs a valid GLFW window");

        window = targetWindow;
        shaderDirectory = targetShaderDirectory;

        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapchain();
        CreateSwapchainImageViews();
        CreateDescriptorSetLayout();
        CreatePipelineLayout();
        CreateGraphicsPipeline();
        CreateCommandPool();
        CreateDepthResources();
        CreateTextureSampler();
        CreateCameraUniformBuffers();
        CreateDescriptorPool();
        CreateCommandBuffers();
        CreateSyncObjects();

        initialized = true;
    }

    void Shutdown() noexcept {
        if (device) {
            try {
                device.waitIdle();
            } catch (...) {
                // Destructors and explicit shutdown must not throw.
            }
        }

        frameInProgress = false;
        drawCommands.clear();

        materials.clear();
        meshes.clear();
        textures.clear();
        textureSampler = nullptr;
        descriptorPool = nullptr;

        for (std::size_t i = 0; i < cameraUniformMemory.size() && i < cameraUniformMapped.size(); ++i) {
            if (cameraUniformMapped[i] != nullptr) {
                cameraUniformMemory[i].unmapMemory();
                cameraUniformMapped[i] = nullptr;
            }
        }
        cameraUniformBuffers.clear();
        cameraUniformMemory.clear();
        cameraUniformMapped.clear();

        commandBuffers.clear();
        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();

        CleanupSwapchain();
        graphicsPipeline = nullptr;
        pipelineLayout = nullptr;
        descriptorSetLayout = nullptr;
        commandPool = nullptr;

        queue = nullptr;
        device = nullptr;
        physicalDevice = nullptr;
        debugMessenger = nullptr;
        surface = nullptr;
        instance = nullptr;

        queueFamilyIndex = std::numeric_limits<std::uint32_t>::max();
        frameIndex = 0;
        acquiredImageIndex = 0;
        framebufferResized = false;
        acquiredSuboptimalSwapchain = false;
        shaderDirectory.clear();
        window = nullptr;
        initialized = false;
    }

    Mesh LoadMesh(const std::filesystem::path& path) {
        RequireInitialized();

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> objMaterials;
        std::string warning;
        std::string error;
        const std::string pathString = path.string();

        if (!tinyobj::LoadObj(
                &attributes,
                &shapes,
                &objMaterials,
                &warning,
                &error,
                pathString.c_str(),
                nullptr,
                true
            )) {
            throw std::runtime_error("Could not load OBJ '" + pathString + "': " + warning + error);
        }

        if (!warning.empty()) {
            std::cerr << "OBJ warning for '" << pathString << "': " << warning << '\n';
        }

        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
        std::unordered_map<Vertex, std::uint32_t, VertexHash> uniqueVertices;

        for (const tinyobj::shape_t& shape : shapes) {
            for (const tinyobj::index_t& index : shape.mesh.indices) {
                if (index.vertex_index < 0) continue;

                const std::size_t positionOffset = static_cast<std::size_t>(index.vertex_index) * 3U;
                if (positionOffset + 2U >= attributes.vertices.size()) {
                    throw std::runtime_error("OBJ contains an invalid position index: " + pathString);
                }

                Vertex vertex{};
                vertex.position = {
                    attributes.vertices[positionOffset],
                    attributes.vertices[positionOffset + 1U],
                    attributes.vertices[positionOffset + 2U]
                };

                if (index.texcoord_index >= 0) {
                    const std::size_t textureOffset = static_cast<std::size_t>(index.texcoord_index) * 2U;
                    if (textureOffset + 1U >= attributes.texcoords.size()) {
                        throw std::runtime_error("OBJ contains an invalid texture-coordinate index: " + pathString);
                    }
                    vertex.textureCoordinate = {
                        attributes.texcoords[textureOffset],
                        1.0f - attributes.texcoords[textureOffset + 1U]
                    };
                }

                const auto [iterator, inserted] = uniqueVertices.emplace(
                    vertex,
                    static_cast<std::uint32_t>(vertices.size())
                );
                if (inserted) vertices.push_back(vertex);
                indices.push_back(iterator->second);
            }
        }

        if (vertices.empty() || indices.empty()) {
            throw std::runtime_error("OBJ has no renderable triangles: " + pathString);
        }

        MeshResource resource{
            .vertexBuffer = UploadBuffer(
                vertices.data(),
                sizeof(Vertex) * vertices.size(),
                vk::BufferUsageFlagBits::eVertexBuffer
            ),
            .indexBuffer = UploadBuffer(
                indices.data(),
                sizeof(std::uint32_t) * indices.size(),
                vk::BufferUsageFlagBits::eIndexBuffer
            ),
            .indexCount = static_cast<std::uint32_t>(indices.size())
        };

        const std::uint32_t id = static_cast<std::uint32_t>(meshes.size());
        meshes.emplace_back(std::move(resource));
        return Mesh{id};
    }

    Texture LoadTexture(const std::filesystem::path& path) {
        RequireInitialized();

        int width = 0;
        int height = 0;
        int channelCount = 0;
        const std::string pathString = path.string();
        stbi_uc* pixels = stbi_load(pathString.c_str(), &width, &height, &channelCount, STBI_rgb_alpha);
        (void)channelCount;
        if (pixels == nullptr) {
            const char* reason = stbi_failure_reason();
            throw std::runtime_error(
                "Could not load texture '" + pathString + "': " +
                (reason != nullptr ? reason : "unknown stb_image error")
            );
        }

        struct PixelGuard {
            stbi_uc* pixels;
            ~PixelGuard() { stbi_image_free(pixels); }
        } pixelGuard{pixels};

        if (width <= 0 || height <= 0) {
            throw std::runtime_error("Texture has invalid dimensions: " + pathString);
        }

        const vk::DeviceSize imageSize =
            static_cast<vk::DeviceSize>(width) *
            static_cast<vk::DeviceSize>(height) * 4U;

        BufferAllocation staging = CreateBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void* mapped = staging.memory.mapMemory(0, imageSize);
        std::memcpy(mapped, pixels, static_cast<std::size_t>(imageSize));
        staging.memory.unmapMemory();

        ImageAllocation image = CreateImage(
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        vk::raii::CommandBuffer commandBuffer = BeginSingleTimeCommands();
        TransitionTextureImage(
            commandBuffer,
            *image.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );
        CopyBufferToImage(
            commandBuffer,
            *staging.buffer,
            *image.image,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height)
        );
        TransitionTextureImage(
            commandBuffer,
            *image.image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );
        EndSingleTimeCommands(std::move(commandBuffer));

        vk::raii::ImageView imageView = CreateImageView(
            *image.image,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );
        TextureResource resource{
            .image = std::move(image),
            .view = std::move(imageView)
        };

        const std::uint32_t id = static_cast<std::uint32_t>(textures.size());
        textures.emplace_back(std::move(resource));
        return Texture{id};
    }

    Material CreateMaterial(const Texture& texture, const glm::vec4& color) {
        RequireInitialized();
        const TextureResource& textureResource = TextureAt(texture);

        const std::size_t activeMaterialCount = static_cast<std::size_t>(std::ranges::count_if(
            materials,
            [](const auto& material) { return material.has_value(); }
        ));
        if (activeMaterialCount >= MaxMaterials) {
            throw std::runtime_error("The starter renderer's material limit has been reached");
        }

        std::vector<vk::DescriptorSetLayout> layouts(MaxFramesInFlight, *descriptorSetLayout);
        const vk::DescriptorSetAllocateInfo allocateInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<std::uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        std::vector<vk::raii::DescriptorSet> descriptorSets =
            device.allocateDescriptorSets(allocateInfo);

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i) {
            const vk::DescriptorBufferInfo bufferInfo{
                .buffer = *cameraUniformBuffers[i],
                .offset = 0,
                .range = sizeof(CameraUniform)
            };
            const vk::DescriptorImageInfo imageInfo{
                .sampler = *textureSampler,
                .imageView = *textureResource.view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };

            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                },
                {
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &imageInfo
                }
            }};
            device.updateDescriptorSets(writes, {});
        }

        MaterialResource resource{
            .texture = texture,
            .color = color,
            .descriptorSets = std::move(descriptorSets)
        };
        const std::uint32_t id = static_cast<std::uint32_t>(materials.size());
        materials.emplace_back(std::move(resource));
        return Material{id};
    }

    void Destroy(Mesh& mesh) {
        RequireInitialized();
        RequireNoFrameInProgress();
        (void)MeshAt(mesh);
        device.waitIdle();
        meshes[mesh.id].reset();
        mesh.id = RendererHandle::Invalid;
    }

    void Destroy(Texture& texture) {
        RequireInitialized();
        RequireNoFrameInProgress();
        (void)TextureAt(texture);

        for (const auto& material : materials) {
            if (material && material->texture.id == texture.id) {
                throw std::logic_error("Destroy materials that use this texture before destroying the texture");
            }
        }

        device.waitIdle();
        textures[texture.id].reset();
        texture.id = RendererHandle::Invalid;
    }

    void Destroy(Material& material) {
        RequireInitialized();
        RequireNoFrameInProgress();
        (void)MaterialAt(material);
        device.waitIdle();
        materials[material.id].reset();
        material.id = RendererHandle::Invalid;
    }

    bool BeginFrame(const Camera& camera) {
        RequireInitialized();
        if (frameInProgress) throw std::logic_error("BeginFrame was called twice without EndFrame");

        if (framebufferResized) {
            RecreateSwapchain();
            return false;
        }

        const vk::Result fenceResult = device.waitForFences(
            *inFlightFences[frameIndex],
            vk::True,
            UINT64_MAX
        );
        if (fenceResult != vk::Result::eSuccess) {
            throw std::runtime_error("Could not wait for the current frame fence");
        }

        const auto [acquireResult, imageIndex] = swapchain.acquireNextImage(
            UINT64_MAX,
            *imageAvailableSemaphores[frameIndex],
            nullptr
        );

        if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
            RecreateSwapchain();
            return false;
        }
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Could not acquire the next swapchain image");
        }

        acquiredSuboptimalSwapchain = acquireResult == vk::Result::eSuboptimalKHR;
        acquiredImageIndex = imageIndex;
        UpdateCameraUniform(camera);
        drawCommands.clear();
        frameInProgress = true;
        return true;
    }

    void DrawMesh(const Mesh& mesh, const Material& material, const glm::mat4& transform) {
        RequireInitialized();
        if (!frameInProgress) throw std::logic_error("DrawMesh must be called between BeginFrame and EndFrame");
        (void)MeshAt(mesh);
        (void)MaterialAt(material);
        drawCommands.push_back({mesh, material, transform});
    }

    void EndFrame() {
        RequireInitialized();
        if (!frameInProgress) throw std::logic_error("EndFrame was called without a successful BeginFrame");

        vk::raii::CommandBuffer& commandBuffer = commandBuffers[frameIndex];
        commandBuffer.reset();
        RecordCommandBuffer(commandBuffer, acquiredImageIndex);

        // Reset exactly once, and only after all operations that may skip this submission.
        device.resetFences(*inFlightFences[frameIndex]);

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*imageAvailableSemaphores[frameIndex],
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[acquiredImageIndex]
        };
        queue.submit(submitInfo, *inFlightFences[frameIndex]);

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[acquiredImageIndex],
            .swapchainCount = 1,
            .pSwapchains = &*swapchain,
            .pImageIndices = &acquiredImageIndex
        };

        const vk::Result presentResult = queue.presentKHR(presentInfo);
        if (presentResult != vk::Result::eSuccess &&
            presentResult != vk::Result::eSuboptimalKHR &&
            presentResult != vk::Result::eErrorOutOfDateKHR) {
            throw std::runtime_error("Could not present the rendered frame");
        }

        frameInProgress = false;
        drawCommands.clear();
        frameIndex = (frameIndex + 1U) % MaxFramesInFlight;

        if (framebufferResized || acquiredSuboptimalSwapchain ||
            presentResult == vk::Result::eSuboptimalKHR ||
            presentResult == vk::Result::eErrorOutOfDateKHR) {
            RecreateSwapchain();
        }
    }

private:
    void RequireInitialized() const {
        if (!initialized) throw std::logic_error("VulkanRenderer is not initialized");
    }

    void RequireNoFrameInProgress() const {
        if (frameInProgress) throw std::logic_error("Resources cannot be destroyed during a frame");
    }

    MeshResource& MeshAt(const Mesh& mesh) {
        if (!mesh || mesh.id >= meshes.size() || !meshes[mesh.id]) {
            throw std::invalid_argument("Invalid or destroyed mesh handle");
        }
        return *meshes[mesh.id];
    }

    TextureResource& TextureAt(const Texture& texture) {
        if (!texture || texture.id >= textures.size() || !textures[texture.id]) {
            throw std::invalid_argument("Invalid or destroyed texture handle");
        }
        return *textures[texture.id];
    }

    const TextureResource& TextureAt(const Texture& texture) const {
        if (!texture || texture.id >= textures.size() || !textures[texture.id]) {
            throw std::invalid_argument("Invalid or destroyed texture handle");
        }
        return *textures[texture.id];
    }

    MaterialResource& MaterialAt(const Material& material) {
        if (!material || material.id >= materials.size() || !materials[material.id]) {
            throw std::invalid_argument("Invalid or destroyed material handle");
        }
        return *materials[material.id];
    }

    void UpdateCameraUniform(const Camera& camera) {
        const float aspectRatio = static_cast<float>(swapchainExtent.width) /
                                  static_cast<float>(swapchainExtent.height);
        const CameraUniform uniform{
            .view = camera.ViewMatrix(),
            .projection = camera.ProjectionMatrix(aspectRatio)
        };
        std::memcpy(cameraUniformMapped[frameIndex], &uniform, sizeof(uniform));
    }

    BufferAllocation CreateBuffer(
        const vk::DeviceSize size,
        const vk::BufferUsageFlags usage,
        const vk::MemoryPropertyFlags memoryProperties
    ) const {
        if (size == 0) throw std::invalid_argument("Cannot create an empty Vulkan buffer");

        vk::raii::DeviceMemory memory = nullptr;
        const vk::BufferCreateInfo createInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };
        vk::raii::Buffer buffer(device, createInfo);

        const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
        const vk::MemoryAllocateInfo allocationInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, memoryProperties)
        };
        memory = vk::raii::DeviceMemory(device, allocationInfo);
        buffer.bindMemory(*memory, 0);

        return {std::move(memory), std::move(buffer)};
    }

    BufferAllocation UploadBuffer(
        const void* source,
        const vk::DeviceSize size,
        const vk::BufferUsageFlags finalUsage
    ) {
        BufferAllocation staging = CreateBuffer(
            size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void* mapped = staging.memory.mapMemory(0, size);
        std::memcpy(mapped, source, static_cast<std::size_t>(size));
        staging.memory.unmapMemory();

        BufferAllocation result = CreateBuffer(
            size,
            finalUsage | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        CopyBuffer(*staging.buffer, *result.buffer, size);
        return result;
    }

    ImageAllocation CreateImage(
        const std::uint32_t width,
        const std::uint32_t height,
        const vk::Format format,
        const vk::ImageTiling tiling,
        const vk::ImageUsageFlags usage,
        const vk::MemoryPropertyFlags memoryProperties
    ) const {
        vk::raii::DeviceMemory memory = nullptr;
        const vk::ImageCreateInfo createInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };
        vk::raii::Image image(device, createInfo);

        const vk::MemoryRequirements requirements = image.getMemoryRequirements();
        const vk::MemoryAllocateInfo allocationInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits, memoryProperties)
        };
        memory = vk::raii::DeviceMemory(device, allocationInfo);
        image.bindMemory(*memory, 0);

        return {std::move(memory), std::move(image)};
    }

    vk::raii::ImageView CreateImageView(
        const vk::Image image,
        const vk::Format format,
        const vk::ImageAspectFlags aspect
    ) const {
        const vk::ImageViewCreateInfo createInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        return vk::raii::ImageView(device, createInfo);
    }

    std::uint32_t FindMemoryType(
        const std::uint32_t allowedTypes,
        const vk::MemoryPropertyFlags properties
    ) const {
        const vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            const bool allowed = (allowedTypes & (1U << index)) != 0;
            const bool hasProperties =
                (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties;
            if (allowed && hasProperties) return index;
        }
        throw std::runtime_error("Could not find suitable Vulkan memory");
    }

    vk::raii::CommandBuffer BeginSingleTimeCommands() const {
        const vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffer commandBuffer =
            std::move(vk::raii::CommandBuffers(device, allocateInfo).front());
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return commandBuffer;
    }

    void EndSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer) const {
        commandBuffer.end();
        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    void CopyBuffer(
        const vk::Buffer source,
        const vk::Buffer destination,
        const vk::DeviceSize size
    ) const {
        vk::raii::CommandBuffer commandBuffer = BeginSingleTimeCommands();
        const vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };
        commandBuffer.copyBuffer(source, destination, copyRegion);
        EndSingleTimeCommands(std::move(commandBuffer));
    }

    static void CopyBufferToImage(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::Buffer source,
        const vk::Image destination,
        const std::uint32_t width,
        const std::uint32_t height
    ) {
        const vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1}
        };
        commandBuffer.copyBufferToImage(
            source,
            destination,
            vk::ImageLayout::eTransferDstOptimal,
            region
        );
    }

    static void TransitionTextureImage(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::Image image,
        const vk::ImageLayout oldLayout,
        const vk::ImageLayout newLayout
    ) {
        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
        if (oldLayout == vk::ImageLayout::eUndefined &&
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("Unsupported texture image layout transition");
        }

        commandBuffer.pipelineBarrier(
            sourceStage,
            destinationStage,
            {},
            {},
            {},
            barrier
        );
    }

    static void TransitionRenderingImage(
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::Image image,
        const vk::ImageLayout oldLayout,
        const vk::ImageLayout newLayout,
        const vk::AccessFlags2 sourceAccess,
        const vk::AccessFlags2 destinationAccess,
        const vk::PipelineStageFlags2 sourceStage,
        const vk::PipelineStageFlags2 destinationStage,
        const vk::ImageAspectFlags aspect
    ) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = sourceStage,
            .srcAccessMask = sourceAccess,
            .dstStageMask = destinationStage,
            .dstAccessMask = destinationAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        const vk::DependencyInfo dependencyInfo{
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
    }

    void RecordCommandBuffer(
        const vk::raii::CommandBuffer& commandBuffer,
        const std::uint32_t imageIndex
    ) {
        commandBuffer.begin({});

        TransitionRenderingImage(
            commandBuffer,
            swapchainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor
        );
        TransitionRenderingImage(
            commandBuffer,
            *depthImage.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::ImageAspectFlagBits::eDepth
        );

        const vk::ClearValue clearColor{
            .color = {.float32 = {0.025f, 0.03f, 0.045f, 1.0f}}
        };
        const vk::ClearValue clearDepth{
            .depthStencil = {.depth = 1.0f, .stencil = 0}
        };
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView = *swapchainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor
        };
        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView = *depthImageView,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clearDepth
        };
        const vk::RenderingInfo renderingInfo{
            .renderArea = {
                .offset = {0, 0},
                .extent = swapchainExtent
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment,
            .pDepthAttachment = &depthAttachment
        };

        commandBuffer.beginRendering(renderingInfo);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer.setViewport(0, vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapchainExtent.width),
            .height = static_cast<float>(swapchainExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        });
        commandBuffer.setScissor(0, vk::Rect2D{
            .offset = {0, 0},
            .extent = swapchainExtent
        });

        for (const DrawCommand& draw : drawCommands) {
            MeshResource& mesh = MeshAt(draw.mesh);
            MaterialResource& material = MaterialAt(draw.material);
            const std::array<vk::Buffer, 1> vertexBuffers{*mesh.vertexBuffer.buffer};
            const std::array<vk::DeviceSize, 1> vertexOffsets{0};
            const std::array<vk::DescriptorSet, 1> materialSets{
                *material.descriptorSets[frameIndex]
            };

            commandBuffer.bindVertexBuffers(0, vertexBuffers, vertexOffsets);
            commandBuffer.bindIndexBuffer(*mesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *pipelineLayout,
                0,
                materialSets,
                {}
            );

            const PushConstants pushConstants{
                .model = draw.transform,
                .color = material.color
            };
            commandBuffer.pushConstants(
                *pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0,
                sizeof(PushConstants),
                &pushConstants
            );
            commandBuffer.drawIndexed(mesh.indexCount, 1, 0, 0, 0);
        }

        commandBuffer.endRendering();
        TransitionRenderingImage(
            commandBuffer,
            swapchainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor
        );
        commandBuffer.end();
    }

    void CreateInstance() {
        constexpr vk::ApplicationInfo applicationInfo{
            .pApplicationName = "Vulkan Renderer Starter",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "Tilky Vulkan Prototype",
            .engineVersion = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = vk::ApiVersion13
        };

        std::vector<const char*> requiredExtensions = RequiredInstanceExtensions();
        const std::vector<vk::ExtensionProperties> availableExtensions =
            context.enumerateInstanceExtensionProperties();
        for (const char* required : requiredExtensions) {
            const bool found = std::ranges::any_of(availableExtensions, [required](const auto& available) {
                return std::strcmp(available.extensionName, required) == 0;
            });
            if (!found) throw std::runtime_error("Required Vulkan instance extension is unavailable: " + std::string(required));
        }

        std::vector<const char*> enabledLayers;
        if constexpr (EnableValidationLayers) {
            const std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();
            for (const char* required : ValidationLayers) {
                const bool found = std::ranges::any_of(availableLayers, [required](const auto& available) {
                    return std::strcmp(available.layerName, required) == 0;
                });
                if (!found) throw std::runtime_error("Required Vulkan validation layer is unavailable: " + std::string(required));
            }
            enabledLayers = ValidationLayers;
        }

        const vk::DebugUtilsMessengerCreateInfoEXT debugInfo = DebugMessengerCreateInfo();
        const vk::InstanceCreateInfo createInfo{
            .pNext = EnableValidationLayers ? &debugInfo : nullptr,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .enabledExtensionCount = static_cast<std::uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()
        };
        instance = vk::raii::Instance(context, createInfo);
    }

    std::vector<const char*> RequiredInstanceExtensions() const {
        std::uint32_t count = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&count);
        if (glfwExtensions == nullptr || count == 0) {
            throw std::runtime_error("GLFW did not provide required Vulkan extensions");
        }

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);
        if constexpr (EnableValidationLayers) extensions.push_back(vk::EXTDebugUtilsExtensionName);
        return extensions;
    }

    static vk::DebugUtilsMessengerCreateInfoEXT DebugMessengerCreateInfo() {
        return {
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
            .pfnUserCallback = &VulkanDebugCallback
        };
    }

    void SetupDebugMessenger() {
        if constexpr (EnableValidationLayers) {
            debugMessenger = instance.createDebugUtilsMessengerEXT(DebugMessengerCreateInfo());
        }
    }

    void CreateSurface() {
        VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
        const VkResult result = glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface);
        if (result != VK_SUCCESS) throw std::runtime_error("Could not create the Vulkan window surface");
        surface = vk::raii::SurfaceKHR(instance, rawSurface);
    }

    void PickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty()) throw std::runtime_error("No Vulkan-capable GPU was found");

        std::multimap<std::uint32_t, vk::raii::PhysicalDevice> candidates;
        for (vk::raii::PhysicalDevice& candidate : devices) {
            const vk::PhysicalDeviceProperties properties = candidate.getProperties();
            if (properties.apiVersion < vk::ApiVersion13) continue;

            const auto features = candidate.getFeatures2<
                vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan13Features
            >();
            const bool requiredFeatures =
                features.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2;
            if (!requiredFeatures) continue;

            const std::vector<vk::ExtensionProperties> extensions =
                candidate.enumerateDeviceExtensionProperties();
            const bool supportsSwapchain = std::ranges::any_of(extensions, [](const auto& extension) {
                return std::strcmp(extension.extensionName, vk::KHRSwapchainExtensionName) == 0;
            });
            if (!supportsSwapchain) continue;

            bool hasGraphicsAndPresentQueue = false;
            const std::vector<vk::QueueFamilyProperties> queueFamilies = candidate.getQueueFamilyProperties();
            for (std::uint32_t index = 0; index < queueFamilies.size(); ++index) {
                if ((queueFamilies[index].queueFlags & vk::QueueFlagBits::eGraphics) &&
                    candidate.getSurfaceSupportKHR(index, *surface)) {
                    hasGraphicsAndPresentQueue = true;
                    break;
                }
            }
            if (!hasGraphicsAndPresentQueue) continue;
            if (candidate.getSurfaceFormatsKHR(*surface).empty() ||
                candidate.getSurfacePresentModesKHR(*surface).empty()) continue;

            std::uint32_t score = properties.limits.maxImageDimension2D;
            if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score += 10000U;
            candidates.emplace(score, std::move(candidate));
        }

        if (candidates.empty()) {
            throw std::runtime_error("No GPU supports the required Vulkan 1.3 rendering features");
        }
        physicalDevice = std::move(std::prev(candidates.end())->second);
    }

    void CreateLogicalDevice() {
        const std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();
        for (std::uint32_t index = 0; index < queueFamilies.size(); ++index) {
            if ((queueFamilies[index].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(index, *surface)) {
                queueFamilyIndex = index;
                break;
            }
        }
        if (queueFamilyIndex == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("No queue supports both graphics and presentation");
        }

        const float priority = 1.0f;
        const vk::DeviceQueueCreateInfo queueInfo{
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &priority
        };
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features> features{
            {.features = {.samplerAnisotropy = vk::True}},
            {.synchronization2 = vk::True, .dynamicRendering = vk::True}
        };
        const std::array<const char*, 1> extensions{vk::KHRSwapchainExtensionName};
        const vk::DeviceCreateInfo createInfo{
            .pNext = &features.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        device = vk::raii::Device(physicalDevice, createInfo);
        queue = vk::raii::Queue(device, queueFamilyIndex, 0);
    }

    void CreateSwapchain() {
        const vk::SurfaceCapabilitiesKHR capabilities =
            physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const std::vector<vk::SurfaceFormatKHR> formats =
            physicalDevice.getSurfaceFormatsKHR(*surface);
        const std::vector<vk::PresentModeKHR> presentModes =
            physicalDevice.getSurfacePresentModesKHR(*surface);

        swapchainSurfaceFormat = ChooseSwapchainFormat(formats);
        swapchainExtent = ChooseSwapchainExtent(capabilities);
        const std::uint32_t imageCount = ChooseSwapchainImageCount(capabilities);

        const vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = imageCount,
            .imageFormat = swapchainSurfaceFormat.format,
            .imageColorSpace = swapchainSurfaceFormat.colorSpace,
            .imageExtent = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = ChoosePresentMode(presentModes),
            .clipped = vk::True
        };
        swapchain = vk::raii::SwapchainKHR(device, createInfo);
        swapchainImages = swapchain.getImages();
    }

    static vk::SurfaceFormatKHR ChooseSwapchainFormat(
        const std::vector<vk::SurfaceFormatKHR>& formats
    ) {
        if (formats.empty()) throw std::runtime_error("The GPU returned no swapchain formats");
        const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR& format) {
            return format.format == vk::Format::eB8G8R8A8Srgb &&
                   format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
        return preferred != formats.end() ? *preferred : formats.front();
    }

    static vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& modes) {
        const bool mailboxAvailable = std::ranges::any_of(modes, [](const vk::PresentModeKHR mode) {
            return mode == vk::PresentModeKHR::eMailbox;
        });
        return mailboxAvailable ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D ChooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        return {
            std::clamp(
                static_cast<std::uint32_t>(std::max(width, 1)),
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width
            ),
            std::clamp(
                static_cast<std::uint32_t>(std::max(height, 1)),
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height
            )
        };
    }

    static std::uint32_t ChooseSwapchainImageCount(const vk::SurfaceCapabilitiesKHR& capabilities) {
        std::uint32_t imageCount = std::max(3U, capabilities.minImageCount);
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }
        return imageCount;
    }

    void CreateSwapchainImageViews() {
        swapchainImageViews.clear();
        swapchainImageViews.reserve(swapchainImages.size());
        for (const vk::Image image : swapchainImages) {
            swapchainImageViews.emplace_back(CreateImageView(
                image,
                swapchainSurfaceFormat.format,
                vk::ImageAspectFlagBits::eColor
            ));
        }
    }

    vk::Format FindDepthFormat() const {
        constexpr std::array candidates{
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint
        };
        for (const vk::Format format : candidates) {
            const vk::FormatProperties properties = physicalDevice.getFormatProperties(format);
            if ((properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) ==
                vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth format was found");
    }

    void CreateDepthResources() {
        const vk::Format format = FindDepthFormat();
        depthImage = CreateImage(
            swapchainExtent.width,
            swapchainExtent.height,
            format,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        depthImageView = CreateImageView(*depthImage.image, format, vk::ImageAspectFlagBits::eDepth);
    }

    void CleanupSwapchain() noexcept {
        depthImageView = nullptr;
        depthImage.image = nullptr;
        depthImage.memory = nullptr;
        swapchainImageViews.clear();
        swapchainImages.clear();
        swapchain = nullptr;
    }

    void RecreateSwapchain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwWaitEvents();
            glfwGetFramebufferSize(window, &width, &height);
        }

        device.waitIdle();
        const vk::Format oldFormat = swapchainSurfaceFormat.format;
        renderFinishedSemaphores.clear();
        CleanupSwapchain();
        CreateSwapchain();
        CreateSwapchainImageViews();
        CreateDepthResources();
        CreateRenderFinishedSemaphores();

        if (swapchainSurfaceFormat.format != oldFormat) CreateGraphicsPipeline();

        framebufferResized = false;
        acquiredSuboptimalSwapchain = false;
    }

    void CreateDescriptorSetLayout() {
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex
            },
            {
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            }
        }};
        const vk::DescriptorSetLayoutCreateInfo createInfo{
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, createInfo);
    }

    void CreatePipelineLayout() {
        const vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        const vk::PipelineLayoutCreateInfo createInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };
        pipelineLayout = vk::raii::PipelineLayout(device, createInfo);
    }

    vk::raii::ShaderModule CreateShaderModule(const std::filesystem::path& path) const {
        const std::vector<char> code = ReadBinaryFile(path);
        const vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const std::uint32_t*>(code.data())
        };
        return vk::raii::ShaderModule(device, createInfo);
    }

    void CreateGraphicsPipeline() {
        vk::raii::ShaderModule vertexShader = CreateShaderModule(shaderDirectory / "mesh.vert.spv");
        vk::raii::ShaderModule fragmentShader = CreateShaderModule(shaderDirectory / "mesh.frag.spv");
        const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{{
            {
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertexShader,
                .pName = "main"
            },
            {
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragmentShader,
                .pName = "main"
            }
        }};

        const vk::VertexInputBindingDescription binding = Vertex::BindingDescription();
        const auto attributes = Vertex::AttributeDescriptions();
        const vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data()
        };
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False
        };
        const std::array dynamicStates{
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        const vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };
        const vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };
        const vk::PipelineRasterizationStateCreateInfo rasterization{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };
        const vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False,
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f
        };
        const vk::PipelineColorBlendAttachmentState blendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlend{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &blendAttachment
        };
        const vk::Format depthFormat = FindDepthFormat();
        const vk::PipelineRenderingCreateInfo renderingInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapchainSurfaceFormat.format,
            .depthAttachmentFormat = depthFormat
        };
        const vk::GraphicsPipelineCreateInfo createInfo{
            .pNext = &renderingInfo,
            .stageCount = static_cast<std::uint32_t>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlend,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .renderPass = nullptr,
            .subpass = 0
        };
        graphicsPipeline = vk::raii::Pipeline(device, nullptr, createInfo);
    }

    void CreateCommandPool() {
        const vk::CommandPoolCreateInfo createInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndex
        };
        commandPool = vk::raii::CommandPool(device, createInfo);
    }

    void CreateCommandBuffers() {
        const vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MaxFramesInFlight
        };
        commandBuffers = vk::raii::CommandBuffers(device, allocateInfo);
    }

    void CreateTextureSampler() {
        const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        const vk::SamplerCreateInfo createInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False
        };
        textureSampler = vk::raii::Sampler(device, createInfo);
    }

    void CreateCameraUniformBuffers() {
        cameraUniformMemory.reserve(MaxFramesInFlight);
        cameraUniformBuffers.reserve(MaxFramesInFlight);
        cameraUniformMapped.reserve(MaxFramesInFlight);

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i) {
            BufferAllocation allocation = CreateBuffer(
                sizeof(CameraUniform),
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            cameraUniformMapped.push_back(allocation.memory.mapMemory(0, sizeof(CameraUniform)));
            cameraUniformMemory.push_back(std::move(allocation.memory));
            cameraUniformBuffers.push_back(std::move(allocation.buffer));
        }
    }

    void CreateDescriptorPool() {
        const std::uint32_t descriptorCount = MaxMaterials * MaxFramesInFlight;
        const std::array<vk::DescriptorPoolSize, 2> poolSizes{{
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = descriptorCount},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = descriptorCount}
        }};
        const vk::DescriptorPoolCreateInfo createInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = descriptorCount,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        descriptorPool = vk::raii::DescriptorPool(device, createInfo);
    }

    void CreateSyncObjects() {
        imageAvailableSemaphores.reserve(MaxFramesInFlight);
        inFlightFences.reserve(MaxFramesInFlight);
        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i) {
            imageAvailableSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{
                .flags = vk::FenceCreateFlagBits::eSignaled
            });
        }
        CreateRenderFinishedSemaphores();
    }

    void CreateRenderFinishedSemaphores() {
        renderFinishedSemaphores.clear();
        renderFinishedSemaphores.reserve(swapchainImages.size());
        for (std::size_t i = 0; i < swapchainImages.size(); ++i) {
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        }
    }
};

VulkanRenderer::VulkanRenderer()
    : impl_(std::make_unique<Impl>()) {}

VulkanRenderer::~VulkanRenderer() {
    Shutdown();
}

void VulkanRenderer::Initialize(
    GLFWwindow* window,
    const std::filesystem::path& shaderDirectory
) {
    try {
        impl_->Initialize(window, shaderDirectory);
    } catch (...) {
        impl_->Shutdown();
        throw;
    }
}

void VulkanRenderer::Shutdown() noexcept {
    if (impl_) impl_->Shutdown();
}

Mesh VulkanRenderer::LoadMesh(const std::filesystem::path& path) {
    return impl_->LoadMesh(path);
}

Texture VulkanRenderer::LoadTexture(const std::filesystem::path& path) {
    return impl_->LoadTexture(path);
}

Material VulkanRenderer::CreateMaterial(const Texture& texture, const glm::vec4& color) {
    return impl_->CreateMaterial(texture, color);
}

void VulkanRenderer::Destroy(Mesh& mesh) {
    impl_->Destroy(mesh);
}

void VulkanRenderer::Destroy(Texture& texture) {
    impl_->Destroy(texture);
}

void VulkanRenderer::Destroy(Material& material) {
    impl_->Destroy(material);
}

bool VulkanRenderer::BeginFrame(const Camera& camera) {
    return impl_->BeginFrame(camera);
}

void VulkanRenderer::DrawMesh(
    const Mesh& mesh,
    const Material& material,
    const glm::mat4& transform
) {
    impl_->DrawMesh(mesh, material, transform);
}

void VulkanRenderer::EndFrame() {
    impl_->EndFrame();
}

void VulkanRenderer::NotifyFramebufferResized() noexcept {
    if (impl_) impl_->framebufferResized = true;
}

bool VulkanRenderer::IsInitialized() const noexcept {
    return impl_ && impl_->initialized;
}
