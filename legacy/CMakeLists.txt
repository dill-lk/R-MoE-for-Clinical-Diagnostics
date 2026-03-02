cmake_minimum_required(VERSION 3.18)
project(RMoEClinicalEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── NOTE ──────────────────────────────────────────────────────────────────────
# This is the LEGACY C++ engine.  The primary engine is now the Python
# implementation: engine.py (project root).
#
# Sources live in  legacy/src/  and headers in  legacy/include/.
# The llama.cpp submodule remains at the project root (../llama.cpp).
#
# Build from the project root (one level above this file):
#   cmake -S legacy -B legacy/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build legacy/build --config Release -j4
#
# GPU build (CUDA):
#   cmake -S legacy -B legacy/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
#   cmake --build legacy/build --config Release -j4
# ─────────────────────────────────────────────────────────────────────────────

# The llama.cpp submodule is at the project root, one level above legacy/.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")

add_executable(rmoe_engine
    src/main.cpp
    src/DiagnosticEngine.cpp
)

target_include_directories(rmoe_engine PRIVATE include)

target_compile_options(rmoe_engine PRIVATE
    -O3
    -mavx2
    -mfma
    -ffast-math
)

find_package(OpenMP REQUIRED)
target_link_libraries(rmoe_engine PRIVATE OpenMP::OpenMP_CXX)

find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(rmoe_engine PRIVATE nlohmann_json::nlohmann_json)
endif()

# llama.cpp backend integration
if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils (required for mtmd)" FORCE)
    add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp_build)
    target_include_directories(rmoe_engine PRIVATE "${LLAMA_CPP_DIR}/include")

    if(TARGET llama)
        target_link_libraries(rmoe_engine PRIVATE llama)
        target_compile_definitions(rmoe_engine PRIVATE RMOE_HAS_LLAMA=1)
    endif()

    if(TARGET common)
        target_link_libraries(rmoe_engine PRIVATE common)
    endif()

    # mtmd: multimodal (CLIP + LLM vision) support
    if(EXISTS "${LLAMA_CPP_DIR}/tools/mtmd/CMakeLists.txt" AND TARGET llama)
        if(NOT DEFINED LLAMA_INSTALL_VERSION)
            set(LLAMA_INSTALL_VERSION "0.0.0")
        endif()
        add_subdirectory("${LLAMA_CPP_DIR}/tools/mtmd" mtmd_build)
        if(TARGET mtmd)
            target_link_libraries(rmoe_engine PRIVATE mtmd)
            target_include_directories(rmoe_engine PRIVATE "${LLAMA_CPP_DIR}/tools/mtmd")
            target_compile_definitions(rmoe_engine PRIVATE RMOE_HAS_MTMD=1)
        endif()
    endif()
endif()
