@echo off
REM E5b: build MSVC + Windows SDK env manually (no vcvarsall/reg.exe; reg is blocked by policy)
REM usage: tools\e5b_msvc_env.bat <exe> [args...]
if not defined MSVC_ROOT set "MSVC_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
if not exist "%MSVC_ROOT%\bin\Hostx64\x64\cl.exe" set "MSVC_ROOT=D:\vc2022\VC\Tools\MSVC\14.44.35207"
if not exist "%MSVC_ROOT%\bin\Hostx64\x64\cl.exe" set "MSVC_ROOT=D:\vs\VC\Tools\MSVC\14.51.36231"

if not defined SDK_ROOT set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
if not defined SDK_VER set "SDK_VER=10.0.26100.0"

set "PATH=%MSVC_ROOT%\bin\Hostx64\x64;%SDK_ROOT%\bin\%SDK_VER%\x64;%PATH%"
set "INCLUDE=%MSVC_ROOT%\include;%SDK_ROOT%\Include\%SDK_VER%\ucrt;%SDK_ROOT%\Include\%SDK_VER%\um;%SDK_ROOT%\Include\%SDK_VER%\shared;%SDK_ROOT%\Include\%SDK_VER%\winrt;%SDK_ROOT%\Include\%SDK_VER%\cppwinrt"
set "LIB=%MSVC_ROOT%\lib\x64;%SDK_ROOT%\Lib\%SDK_VER%\ucrt\x64;%SDK_ROOT%\Lib\%SDK_VER%\um\x64"

set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "TORCH_EXTENSIONS_DIR=D:/model/.cache/torch_ext"
%*
