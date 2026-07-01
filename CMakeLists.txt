cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(FlagGems LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 20) # for fold-expression and concepts
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------- RPATH settings ---------------------------
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)

# default Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------- project options ---------------------------
option(FLAGGEMS_USE_EXTERNAL_TRITON_JIT "whether to use external triton jit library" OFF)
option(FLAGGEMS_USE_EXTERNAL_PYBIND11 "whether to use external pybind11 library" ON)
option(FLAGGEMS_BUILD_C_EXTENSIONS "whether to build c extension" OFF)
option(FLAGGEMS_BUILD_CTESTS "Whether to build CPP unit tests" ${FLAGGEMS_BUILD_C_EXTENSIONS})
option(FLAGGEMS_INSTALL "Whether to install the package" ${PROJECT_IS_TOP_LEVEL})

if(NOT FLAGGEMS_BUILD_C_EXTENSIONS)
  return()
endif()

# ==============================================================================
# Backend Selection
# ==============================================================================
set(FLAGGEMS_BACKEND "CUDA" CACHE STRING "Target backend: CUDA, IX, MUSA, NPU, GCU")
set_property(CACHE FLAGGEMS_BACKEND PROPERTY STRINGS "CUDA" "IX" "MUSA" "NPU" "GCU")

if(NOT FLAGGEMS_BACKEND MATCHES "^(CUDA|IX|MUSA|NPU|GCU)$")
    message(FATAL_ERROR "Invalid FLAGGEMS_BACKEND: ${FLAGGEMS_BACKEND}")
endif()
message(STATUS "Building FlagGems with backend: ${FLAGGEMS_BACKEND}")

# NPU, MUSA and GCU do not need CUDA language
if(FLAGGEMS_BACKEND STREQUAL "NPU"
   OR FLAGGEMS_BACKEND STREQUAL "MUSA"
   OR FLAGGEMS_BACKEND STREQUAL "GCU")
    project(FlagGems LANGUAGES CXX VERSION 0.1.0)
else()
    project(FlagGems LANGUAGES CUDA CXX VERSION 0.1.0)
endif()

# ------------------------------- Backend compile definitions -------------------------------
if(FLAGGEMS_BACKEND STREQUAL "CUDA")
    add_compile_definitions(FLAGGEMS_USE_CUDA)
elseif(FLAGGEMS_BACKEND STREQUAL "IX")
    add_compile_definitions(FLAGGEMS_USE_IX)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    add_compile_definitions(FLAGGEMS_USE_NPU)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    add_compile_definitions(FLAGGEMS_USE_MUSA)
elseif(FLAGGEMS_BACKEND STREQUAL "GCU")
    add_compile_definitions(FLAGGEMS_USE_GCU)
endif()

message(STATUS "Building flag gems with C extensions")
include(FetchContent)

# dependencies: python (must be before backend includes that need Python_EXECUTABLE)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# --------------------------- Backend-specific dependencies ---------------------------
if(FLAGGEMS_BACKEND STREQUAL "CUDA" OR FLAGGEMS_BACKEND STREQUAL "IX")
    find_package(CUDAToolkit REQUIRED)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    include(BackendNPU)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    include(BackendMUSA)
elseif(FLAGGEMS_BACKEND STREQUAL "GCU")
    include(BackendGCU)
endif()

# torch
find_package(Torch MODULE REQUIRED) # This is the FindTorch.cmake
# dependencies: json
if(FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
  find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
else()
  set(TRITON_JIT_INSTALL ON) # install triton jit
  set(BACKEND ${FLAGGEMS_BACKEND} CACHE STRING "Backend for TritonJIT" FORCE)
  FetchContent_Declare(TritonJIT
    GIT_REPOSITORY https://github.com/flagos-ai/libtriton_jit
    GIT_TAG master
    # SOURCE_DIR <your local source dir of libtriton_jit>
  )
  FetchContent_MakeAvailable(TritonJIT)
  # GluonJITFunction is a JITFunction subclass; accept isinstance() in libtriton_jit scripts.
  if(FLAGGEMS_BUILD_C_EXTENSIONS)
    set(_tj_hdr_path "${TritonJIT_SOURCE_DIR}/include/triton_jit/triton_jit_function.h")
    if(EXISTS "${_tj_hdr_path}")
      file(READ "${_tj_hdr_path}" _tj_hdr_content)
      if(NOT _tj_hdr_content MATCHES "append_global_scratch\\(unsigned grid_x")
        string(REPLACE "#include \"fmt/core.h\""
                     "#include \"fmt/core.h\"\n#include \"torch/torch.h\""
                     _tj_hdr_content "${_tj_hdr_content}")
        string(REPLACE
          "void append_global_scratch() {\n    void* global_scratch = nullptr;\n    this->buf.push_arg(global_scratch);\n  }"
          "void append_global_scratch(unsigned grid_x, unsigned grid_y, unsigned grid_z, unsigned scratch_per_cta) {\n    const int64_t grid_ctas = static_cast<int64_t>(grid_x) * grid_y * grid_z;\n    const int64_t nbytes = std::max<int64_t>(grid_ctas * scratch_per_cta, 256);\n    thread_local at::Tensor scratch;\n    if (!scratch.defined() || scratch.numel() < nbytes) {\n      scratch = at::empty({nbytes}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));\n    }\n    void* global_scratch = scratch.data_ptr();\n    this->buf.push_arg(global_scratch);\n  }"
          _tj_hdr_content "${_tj_hdr_content}")
        string(REPLACE
          "handler.append_global_scratch();\n    handler.append_global_scratch();"
          "constexpr unsigned kDefaultScratchPerCta = 256;\n    handler.append_global_scratch(grid_x, grid_y, grid_z, kDefaultScratchPerCta);\n    handler.append_global_scratch(grid_x, grid_y, grid_z, kDefaultScratchPerCta);"
          _tj_hdr_content "${_tj_hdr_content}")
        file(WRITE "${_tj_hdr_path}" "${_tj_hdr_content}")
      endif()
    endif()
    foreach(_tj_script IN ITEMS gen_ssig.py standalone_compile.py)
      set(_tj_script_path "${TritonJIT_SOURCE_DIR}/scripts/${_tj_script}")
      if(EXISTS "${_tj_script_path}")
        file(READ "${_tj_script_path}" _tj_script_content)
        string(REPLACE "type(fn) is triton.runtime.JITFunction"
                     "isinstance(fn, triton.runtime.JITFunction)"
                     _tj_script_content "${_tj_script_content}")
        if(_tj_script STREQUAL "standalone_compile.py")
          string(REPLACE
            "# STEP1: JITFunction, constants, signature, specialization\n\n    # STEP2:"
            "# STEP1: JITFunction, constants, signature, specialization\n    if getattr(fn, \"is_gluon\", lambda: False)():\n        from triton.experimental.gluon._runtime import GluonASTSource\n\n        src = GluonASTSource(\n            fn=src.fn,\n            signature=src.signature,\n            constexprs=src.constants,\n            attrs=src.attrs,\n        )\n\n    # STEP2:"
            _tj_script_content "${_tj_script_content}")
        endif()
        file(WRITE "${_tj_script_path}" "${_tj_script_content}")
      endif()
    endforeach()
    set(_tj_meta_h_path "${TritonJIT_SOURCE_DIR}/include/triton_jit/kernel_metadata.h")
    if(EXISTS "${_tj_meta_h_path}")
      file(READ "${_tj_meta_h_path}" _tj_meta_h_content)
      if(NOT _tj_meta_h_content MATCHES "load_launch_num_warps")
        string(REPLACE
          "unsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name);\n\n}  // namespace triton_jit"
          "unsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name);\n\n// Launch-time num_warps from compiled metadata (Gluon may differ from compile option).\nunsigned int load_launch_num_warps(const std::string& dir,\n                                     const std::string& kernel_name,\n                                     unsigned int fallback = 0);\n\n}  // namespace triton_jit"
          _tj_meta_h_content "${_tj_meta_h_content}")
        file(WRITE "${_tj_meta_h_path}" "${_tj_meta_h_content}")
      endif()
    endif()
    set(_tj_meta_cpp_path "${TritonJIT_SOURCE_DIR}/src/kernel_metadata.cpp")
    if(EXISTS "${_tj_meta_cpp_path}")
      file(READ "${_tj_meta_cpp_path}" _tj_meta_cpp_content)
      if(NOT _tj_meta_cpp_content MATCHES "load_launch_num_warps")
        string(REPLACE
          "unsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name) {"
          "unsigned int load_launch_num_warps(const std::string& dir,\n                                   const std::string& kernel_name,\n                                   unsigned int fallback) {\n  std::string path = fmt::format(\"{}/{}.json\", dir, kernel_name);\n  std::ifstream f(path);\n  if (!f.is_open()) {\n    return fallback;\n  }\n  try {\n    nlohmann::json j = nlohmann::json::parse(f);\n    return j.value(\"num_warps\", fallback);\n  } catch (const nlohmann::json::exception& e) {\n    LOG(WARNING) << fmt::format(\"Failed to parse metadata {}: {}\", path, e.what());\n    return fallback;\n  }\n}\n\nunsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name) {"
          _tj_meta_cpp_content "${_tj_meta_cpp_content}")
        file(WRITE "${_tj_meta_cpp_path}" "${_tj_meta_cpp_content}")
      endif()
    endif()
    set(_tj_kernel_h_path "${TritonJIT_SOURCE_DIR}/include/triton_jit/triton_kernel.h")
    if(EXISTS "${_tj_kernel_h_path}")
      file(READ "${_tj_kernel_h_path}" _tj_kernel_h_content)
      if(NOT _tj_kernel_h_content MATCHES "load_launch_num_warps")
        string(REPLACE
          "    // Calculate block dimensions using backend-specific warp size\n    unsigned int block_x = num_warps * Backend::WARP_SIZE;"
          "    // Use compiled metadata num_warps when present (Gluon warp_specialize may exceed compile option).\n    int launch_warps = num_warps;\n    if (unsigned int meta_warps = load_launch_num_warps(dir_, kernel_name_, 0); meta_warps > 0) {\n      launch_warps = static_cast<int>(meta_warps);\n    }\n    unsigned int block_x = static_cast<unsigned int>(launch_warps) * Backend::WARP_SIZE;"
          _tj_kernel_h_content "${_tj_kernel_h_content}")
        file(WRITE "${_tj_kernel_h_path}" "${_tj_kernel_h_content}")
      endif()
    endif()
  endif()
endif()

# dependencies: pybind11
if(FLAGGEMS_USE_EXTERNAL_PYBIND11)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
else()
  FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11)
  FetchContent_MakeAvailable(pybind11)
endif()

if(FLAGGEMS_BUILD_CTESTS)
  set(INSTALL_GTEST OFF) # we do not install tests
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

execute_process(
  #COMMAND ${Python_EXECUTABLE}
  COMMAND ${CMAKE_COMMAND} -E env --unset=LD_PRELOAD ${Python_EXECUTABLE}
          -c "import torch; import triton, sys; sys.stdout.write(triton.__version__)"
  RESULT_VARIABLE _ret
  OUTPUT_VARIABLE TRITON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _ret EQUAL 0)
  message(FATAL_ERROR "Triton IS NOT installed properly")
endif()
if(TRITON_VERSION VERSION_GREATER_EQUAL "3.5")
  set(TRITON_GE_3P5 1)
endif()
message(STATUS "Detected Triton version: ${TRITON_VERSION}")

# --------------------------- subdirectories ---------------------------
add_subdirectory(lib)
add_subdirectory(src/flag_gems/csrc)
if(FLAGGEMS_BUILD_CTESTS)
  enable_testing()
  add_subdirectory(ctests)
endif()

# --------------------------- install package ---------------------------
if(FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfigVersion.cmake"
    VERSION "${FlagGems_VERSION_MAJOR}.${FlagGems_VERSION_MINOR}.${FlagGems_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
  )
  # install the FindTorch
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems)
endif()
