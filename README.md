# Vulkan Renderer Starter

This is the tutorial demo refactored into a small renderer API. The application owns the window and game loop; `VulkanRenderer` owns Vulkan and exposes resources plus frame and draw operations.

```cpp
Mesh mesh = renderer.LoadMesh("models/room.obj");
Texture texture = renderer.LoadTexture("textures/room.png");
Material material = renderer.CreateMaterial(texture);

if (renderer.BeginFrame(camera)) {
    renderer.DrawMesh(mesh, material, transform);
    renderer.EndFrame();
}
```

The included application loads one cube and submits it three times with different transforms and material colors. Hold the right mouse button to look around. Use `W/A/S/D`, `Space`, and `Left Ctrl` to move; `Escape` closes the window.

## Build on Windows with CLion, MinGW and vcpkg

Install the Vulkan SDK first so CMake can find Vulkan and `glslc`. Install the remaining dependencies using the same vcpkg triplet as your compiler:

```powershell
vcpkg install glfw3 glm stb tinyobjloader --triplet x64-mingw-dynamic
```

Configure and build:

```powershell
cmake -S . -B cmake-build-debug -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-mingw-dynamic

cmake --build cmake-build-debug
```

Run the included scene:

```powershell
./cmake-build-debug/VulkanRendererStarter.exe
```

Or supply an OBJ and texture at runtime:

```powershell
./cmake-build-debug/VulkanRendererStarter.exe models/viking_room.obj textures/viking_room.png
```

The post-build step places `assets/` and compiled SPIR-V shaders beside the executable, so the program does not depend on hard-coded source paths.

## What changed from the tutorial

- The GLFW loop and input live in `Application`.
- Camera view/projection data is updated once per frame.
- Model transforms and material colors use push constants per draw.
- Meshes, textures, and materials are independently loadable renderer resources.
- A frame can contain any number of `DrawMesh` submissions.
- Swapchain recreation returns immediately from a skipped frame and rebuilds depth resources safely.
- The frame fence is reset only once, immediately before submission.
- Textures use one valid mip level; no uninitialized mip levels are allocated.
- Vulkan 1.3 is requested instead of Vulkan 1.4.

The renderer deliberately stops before lighting, glTF, batching, or Tilky integration. The next sensible step is to implement this API behind Tilky's `IRenderer` interface while keeping the current OpenGL backend available.

