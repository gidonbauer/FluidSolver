BASENAME_CXX = ${notdir ${CXX}}

ifeq (${BASENAME_CXX}, clang++)

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -march=native -ffast-math -O3 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

else ifeq (${BASENAME_CXX}, ${filter ${BASENAME_CXX}, g++ g++-15})

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -march=native -ffast-math -O3 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

else ifeq (${BASENAME_CXX}, icpx)

	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -O3 -ffast-math -xSSE4.2 -axCORE-AVX2,AVX -fp-model fast=2 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g
	CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined

else

  ${error "Unknown C++ compiler `${CXX}`"}

endif

DEBUG ?= 0
ifeq (${DEBUG}, 0)
  CXX_FLAGS += ${CXX_RELEASE_FLAGS}
else
  CXX_FLAGS += ${CXX_DEBUG_FLAGS}
endif

SANITIZE ?= 0
ifeq (${SANITIZE}, 1)
  CXX_FLAGS += ${CXX_SANITIZER_FLAGS}
endif

SCOREP ?= 0
ifeq (${SCOREP}, 1)
  CXX_FLAGS += -g
  CXX := scorep ${CXX}
endif
