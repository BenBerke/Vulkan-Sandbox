cmake_minimum_required(VERSION 3.24)

project(VulkanRendererStarter VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan 1.3 REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_path(STB_INCLUDE_DIR stb_image.h REQUIRED)

if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc was not found. Install the Vulkan SDK and configure CMake again.")
endif()

set(SHADER_SOURCE_DIR "shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled-shaders")

function(compile_shader source stage output_name)
    set(output "${SHADER_BINARY_DIR}/${output_name}")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_BINARY_DIR}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=${stage} "${source}" -o "${output}"
        DEPENDS "${source}"
        VERBATIM
    )
    set(COMPILED_SHADER_OUTPUTS ${COMPILED_SHADER_OUTPUTS} "${output}" PARENT_SCOPE)
endfunction()

compile_shader("${SHADER_SOURCE_DIR}/mesh.vert.glsl" vert mesh.vert.spv)
compile_shader("${SHADER_SOURCE_DIR}/mesh.frag.glsl" frag mesh.frag.spv)

add_custom_target(VulkanRendererShaders ALL DEPENDS ${COMPILED_SHADER_OUTPUTS})

add_executable(VulkanRendererStarter
        src/main.cpp
        src/Application.cpp
        src/Camera.cpp
        src/renderer/VulkanRenderer.cpp
)

add_dependencies(VulkanRendererStarter VulkanRendererShaders)

target_include_directories(VulkanRendererStarter
    PRIVATE
        "include"
        "${STB_INCLUDE_DIR}"
)

target_compile_definitions(VulkanRendererStarter
    PRIVATE
        GLFW_INCLUDE_NONE
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
        VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

target_link_libraries(VulkanRendererStarter
    PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        tinyobjloader::tinyobjloader
)

if(MSVC)
    target_compile_options(VulkanRendererStarter PRIVATE /W4 /permissive-)
else()
    target_compile_options(VulkanRendererStarter PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET VulkanRendererStarter POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "assets"
            "$<TARGET_FILE_DIR:VulkanRendererStarter>/assets"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "$<TARGET_FILE_DIR:VulkanRendererStarter>/shaders"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${COMPILED_SHADER_OUTPUTS}
            "$<TARGET_FILE_DIR:VulkanRendererStarter>/shaders"
    VERBATIM
)

