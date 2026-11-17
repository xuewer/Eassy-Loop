# 终端配置

客户端从项目 .env 加载配置，但不会覆盖进程已有的同名变量。修改 .env 后，已打开的 PyCharm 终端可能仍持有旧值。

在项目根目录的 PowerShell 7 中执行：

~~~powershell
. ./tools/Use-ProjectEnv.ps1
~~~

如果终端执行策略不允许加载脚本，可使用以下交互命令，无需更改执行策略。读取结果进入变量，不回显密钥：

~~~powershell
$readProjectKey = @'
import json
from dotenv import dotenv_values
value = dotenv_values('.env').get('DASHSCOPE_API_KEY')
if not value or not value.strip():
    raise SystemExit('Missing DASHSCOPE_API_KEY in project .env')
print(json.dumps({'key': value.strip()}))
'@
$payload = $readProjectKey | & ./.venv/Scripts/python.exe -B -
if ($LASTEXITCODE -ne 0) { throw 'Could not read project .env' }
$parsed = $payload | ConvertFrom-Json
$env:DASHSCOPE_API_KEY = $parsed.key
Remove-Variable payload,parsed,readProjectKey
Write-Host 'Project key loaded; value hidden.'
~~~

新终端需重新加载；脚本只更新当前进程的 DASHSCOPE_API_KEY，不修改系统环境、端点、DNS、代理或启动配置。不要在共享日志中打印密钥。
