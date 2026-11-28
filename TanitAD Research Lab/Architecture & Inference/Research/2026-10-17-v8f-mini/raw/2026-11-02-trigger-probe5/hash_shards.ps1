Get-ChildItem D:\tanitad\jobs\probe5\shards\shard_*.zip | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + "  " + $_.Name }
