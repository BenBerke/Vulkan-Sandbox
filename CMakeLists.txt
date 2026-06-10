cmake_minimum_required(VERSION 3.30)
project(VulkanLearning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(VulkanLearning
        main.cpp
)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

# stb from vcpkg is usually header-only, so this finds its include folder.
find_path(STB_INCLUDE_DIRS "stb_image.h")

target_include_directories(VulkanLearning PRIVATE
        ${STB_INCLUDE_DIRS}
)

target_link_libraries(VulkanLearning PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        tinyobjloader::tinyobjloader
)

function (add_slang_shader_target TARGET)
    cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
    set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set (ENTRY_POINTS -entry vertMain -entry fragMain)
    add_custom_command (
            OUTPUT ${SHADERS_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
    add_custom_command (
            OUTPUT  ${SHADERS_DIR}/slang.spv
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
            WORKING_DIRECTORY ${SHADERS_DIR}
            DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
            COMMENT "Compiling Slang Shaders"
            VERBATIM
    )
    add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

set(SHADER_SLANG_SOURCES "${CMAKE_CURRENT_LIST_DIR}/shaders/shader.slang")

add_slang_shader_target(compile_slang_shaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(VulkanLearning compile_slang_shaders)