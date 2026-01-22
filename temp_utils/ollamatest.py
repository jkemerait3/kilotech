import subprocess
result = subprocess.run(['ollama', 'run', 'phi'], input=b"Hello", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
print(result.stdout.decode())
print(result.stderr.decode())
