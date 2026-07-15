cmake_minimum_required(VERSION 3.15)
project(othello_alphazero CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependency-free: standard library only (self-made autograd + MCTS + Othello).
add_executable(othello
    autograd.cpp
    game.cpp
    net.cpp
    mcts.cpp
    pipeline.cpp
    play.cpp
    main.cpp)

if (MSVC)
    target_compile_options(othello PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(othello PRIVATE -O3 -march=native -funsafe-math-optimizations)
    find_package(Threads REQUIRED)
    target_link_libraries(othello PRIVATE Threads::Threads)
endif()
