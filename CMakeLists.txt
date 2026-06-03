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