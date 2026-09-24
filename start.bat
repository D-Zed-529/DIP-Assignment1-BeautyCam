@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到项目虚拟环境：.venv\Scripts\python.exe
    echo 请先按 README.md 的“环境与运行”步骤安装 Python 3.12 和项目依赖。
    pause
    popd
    exit /b 1
)

set "PYTHONUTF8=1"
if /I "%~1"=="--check" (
    "%PYTHON_EXE%" -c "import gui.main_window; print('GUI 模块导入成功')"
) else (
    echo 正在启动 BeautyCam...
    "%PYTHON_EXE%" -m gui.main_window
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [错误] BeautyCam 启动失败，退出码：%EXIT_CODE%
    pause
)
popd
exit /b %EXIT_CODE%
