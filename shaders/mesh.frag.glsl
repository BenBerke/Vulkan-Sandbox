#version 450

layout(set = 0, binding = 1) uniform sampler2D materialTexture;

layout(location = 0) in vec3 vertexColor;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec4 materialColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(materialTexture, texCoord) * vec4(vertexColor, 1.0) * materialColor;
}

