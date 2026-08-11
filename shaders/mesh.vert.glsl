#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 projection;
} camera;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} pushData;

layout(location = 0) out vec3 vertexColor;
layout(location = 1) out vec2 texCoord;
layout(location = 2) out vec4 materialColor;

void main() {
    gl_Position = camera.projection * camera.view * pushData.model * vec4(inPosition, 1.0);
    vertexColor = inColor;
    texCoord = inTexCoord;
    materialColor = pushData.color;
}

