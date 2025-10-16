if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")

  set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wshadow -Wconversion -Winline -std=c++23")
  if (FS_FAST)
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -ffast-math -DIGOR_NDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")

  set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23")
  if (FS_FAST)
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -ffast-math -DIGOR_NDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")

  set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23")
  if (FS_FAST)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model fast=2 -ffast-math -DIGOR_NDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model precise")
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "NVHPC")

  set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wshadow -std=c++23")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdpar -Minfo=accel,par,stdpar -DFS_STDPAR -DIGOR_USE_CASSERT")
  if (FS_FAST)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fast -fastsse -Mvect=simd:256")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fastsse -Mvect=simd:256,noassoc")
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined

else()

  message(FATAL_ERROR "Unknown C++ compiler `${CMAKE_CXX_COMPILER}` with ID `${CMAKE_CXX_COMPILER_ID}`")

endif()

if (FS_USE_SCOREP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  set(CMAKE_C_COMPILER_LAUNCHER   scorep)
  set(CMAKE_CXX_COMPILER_LAUNCHER scorep)
endif()
