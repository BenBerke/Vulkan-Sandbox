#include "Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

glm::vec3 Camera::Forward() const {
    const float yaw = glm::radians(yawDegrees);
    const float pitch = glm::radians(pitchDegrees);

    return glm::normalize(glm::vec3{
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)
    });
}

glm::vec3 Camera::Right() const {
    return glm::normalize(glm::cross(Forward(), glm::vec3{0.0f, 1.0f, 0.0f}));
}

glm::mat4 Camera::ViewMatrix() const {
    return glm::lookAt(position, position + Forward(), glm::vec3{0.0f, 1.0f, 0.0f});
}

glm::mat4 Camera::ProjectionMatrix(const float aspectRatio) const {
    glm::mat4 projection = glm::perspective(
        glm::radians(verticalFovDegrees),
        std::max(aspectRatio, 0.001f),
        nearPlane,
        farPlane
    );

    // GLM's clip-space Y direction is opposite Vulkan's.
    projection[1][1] *= -1.0f;
    return projection;
}

