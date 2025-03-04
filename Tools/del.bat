@echo off
:h
setlocal enabledelayedexpansion
for /d %%a in (%1*) do (
set p="%%a\"
set p=!p:.\"=..\"!
rd !p! || ( call :h !p! & rd !p! )
)
