@REM bash ./compile_data_loader.sh

@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%cd%"
set "SRC_DIR=src\sf18_nnue\data_loader\cpp"
set "BUILD_DIR=build"
set "PGO_DIR=pgo_data"

if "%~1"=="" (
    set "PGO_INPUT=%ROOT_DIR%\.pgo\small.binpack"
) else (
    set "PGO_INPUT=%~1"
)

echo ROOT_DIR: %ROOT_DIR%
echo SRC_DIR: %SRC_DIR%
echo BUILD_DIR: %BUILD_DIR%
echo PGO_DIR: %PGO_DIR%
echo PGO_INPUT: %PGO_INPUT%

echo Cleaning previous build and profile data...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%PGO_DIR%" rmdir /s /q "%PGO_DIR%"

echo Configuring PGO_Generate build...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
  -DCMAKE_BUILD_TYPE=PGO_Generate ^
  -DPGO_PROFILE_DATA_DIR="%ROOT_DIR%\%PGO_DIR%" ^
  -DPGO_INPUT="%PGO_INPUT%" ^
  -DLIB_COPY_DIR="%ROOT_DIR%"

echo Building instrumented default targets...
cmake --build "%BUILD_DIR%" -j

echo Running bench to generate profile data...
cmake --build "%BUILD_DIR%" --target pgo_run -j

:: Merge Clang raw profiles if they exist (utile si vous utilisez Clang sous Windows)
if exist "%PGO_DIR%\*.profraw" (
    echo Merging Clang raw profiles...
    llvm-profdata merge -output="%PGO_DIR%\default.profdata" "%PGO_DIR%\*.profraw"
    if errorlevel 1 (
        echo Error: llvm-profdata failed. Make sure it is in your PATH. >&2
        exit /b 1
    )
)

echo Re-configuring for PGO_Use...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
  -DCMAKE_BUILD_TYPE=PGO_Use ^
  -DPGO_PROFILE_DATA_DIR="%ROOT_DIR%\%PGO_DIR%" ^
  -DCMAKE_INSTALL_PREFIX=".\" ^
  -DLIB_COPY_DIR="%ROOT_DIR%"

echo Building default targets with profile data...
cmake --build "%BUILD_DIR%" -j

echo PGO build complete.

if exist "%PGO_DIR%" rmdir /s /q "%PGO_DIR%"