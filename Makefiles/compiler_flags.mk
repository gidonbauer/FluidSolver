BASENAME_CXX = ${notdir ${CXX}}

ifeq (${BASENAME_CXX}, clang++)

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -Winline -std=c++23
	CXX_RELEASE_FLAGS = -march=native -O3
	CXX_FAST_FLAGS = ${CXX_RELEASE_FLAGS} -ffast-math -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
	CXX_OPENMP_FLAGS = -fopenmp

else ifeq (${BASENAME_CXX}, ${filter ${BASENAME_CXX}, g++ g++-15})

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -march=native -O3
	CXX_FAST_FLAGS = ${CXX_RELEASE_FLAGS} -ffast-math -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
	CXX_OPENMP_FLAGS = -fopenmp

else ifeq (${BASENAME_CXX}, icpx)

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model precise
	CXX_FAST_FLAGS = -O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model fast=2 -ffast-math -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
#	CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined
	CXX_SANITIZER_FLAGS = -fsanitize=thread
	CXX_OPENMP_FLAGS = -qopenmp

else ifeq (${BASENAME_CXX}, nvc++)

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -std=c++23
	CXX_RELEASE_FLAGS = -O3 -fastsse -Mvect=simd:256,noassoc
	CXX_FAST_FLAGS = -O3 -fast -fastsse -Mvect=simd:256
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined
	CXX_OPENMP_FLAGS = -stdpar -Minfo=accel,par,stdpar -DFS_STDPAR -DIGOR_USE_CASSERT

else

  ${error "Unknown C++ compiler `${CXX}`"}

endif

DEBUG    ?= 0
FAST     ?= 0
SANITIZE ?= 0
SCOREP   ?= 0
STDPAR   ?= 0

ifeq (${DEBUG}, 1)
  CXX_FLAGS += ${CXX_DEBUG_FLAGS}
else ifeq (${FAST}, 1)
  CXX_FLAGS += ${CXX_FAST_FLAGS}
else
  CXX_FLAGS += ${CXX_RELEASE_FLAGS}
endif

ifeq (${SANITIZE}, 1)
  CXX_FLAGS += ${CXX_SANITIZER_FLAGS}
endif

ifeq (${SCOREP}, 1)
  CXX_FLAGS += -g -fno-omit-frame-pointer
  CXX := scorep ${CXX}
endif

ifeq (${STDPAR}, 1)
  CXX_FLAGS += -DFS_STDPAR
endif
