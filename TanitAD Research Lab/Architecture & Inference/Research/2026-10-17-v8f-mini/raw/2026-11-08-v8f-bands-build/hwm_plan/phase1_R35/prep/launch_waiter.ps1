$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c "F:\TanitAD-scratch\hwm_plan\phase1_R35_waiter.cmd" < NUL'; ProcessStartupInformation=$si}
"WINPID=$($r.ProcessId) rc=$($r.ReturnValue)"
