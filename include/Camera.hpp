#pragma once

#include <glm/glm.hpp>

class Camera {
public:
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    float yawDegrees{-90.0f};
    float pitchDegrees{0.0f};
    float verticalFovDegrees{60.0f};
    float nearPlane{0.1f};
    float farPlane{200.0f};

    [[nodiscard]] glm::vec3 Forward() const;
    [[nodiscard]] glm::vec3 Right() const;
    [[nodiscard]] glm::mat4 ViewMatrix() const;
    [[nodiscard]] glm::mat4 ProjectionMatrix(float aspectRatio) const;
};

