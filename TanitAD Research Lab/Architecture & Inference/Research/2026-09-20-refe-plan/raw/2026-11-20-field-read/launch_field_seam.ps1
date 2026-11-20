# Detached launch of run_field_seam.cmd <arm> via WMI, CreateFlags 0x08000200 (CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW).
# Verify by heartbeat: E:\Projects\TanitAD-artifacts\field_read\CHAIN.log gains a line within a minute.
param([Parameter(Mandatory=$true)][ValidateSet('C','F','Cs','Fs')][string]$Arm)
$hold = 'E:\Projects\TanitAD-artifacts\scorer_gym_20261107\field_read'
$cmd = "cmd.exe /c `"`"$hold\run_field_seam.cmd`" $Arm`""
$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 })
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $hold; ProcessStartupInformation = $si }
"launched: ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId) arm=$Arm"
