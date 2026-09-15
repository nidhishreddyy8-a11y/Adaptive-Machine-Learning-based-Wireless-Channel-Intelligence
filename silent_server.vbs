Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

Dim appDir, pythonExe, logFile

appDir   = "C:\Users\HP\OneDrive\Desktop\Adaptive ML"
pythonExe = "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
logFile  = appDir & "\server.log"

' Kill any existing instance on port 5000
objShell.Run "cmd /c for /f ""tokens=5"" %a in ('netstat -ano 2>nul ^| findstr "":5000 ""') do taskkill /PID %a /F", 0, True

WScript.Sleep 1000

' Launch Flask using python.exe (not pythonw) but hidden window
objShell.CurrentDirectory = appDir
objShell.Run """" & pythonExe & """ app/app.py > """ & logFile & """ 2>&1", 0, False
