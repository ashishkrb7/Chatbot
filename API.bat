@echo off
SET LOGFILE=".\log.txt"
(echo====================================================================================================== >> %LOGFILE%)
(echo Script Start Running at - ^ %date% %time% >> %LOGFILE%)
call "C:\ProgramData\Anaconda3\Scripts\activate.bat"
"C:\Users\imash\.conda\envs\mluser\python.exe" ".\app.py"
(echo Script Successfully Executed at - ^ %date% %time% >> %LOGFILE%)
(echo====================================================================================================== >> %LOGFILE%)
pause