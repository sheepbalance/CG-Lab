import subprocess
import sys

# 运行 Work0/main.py 并捕获输出
result = subprocess.run(
    [sys.executable, 'Work0/main.py'],
    capture_output=True,
    text=True,
    cwd='c:\\Users\\examc\\Documents\\trae_projects\\CG-Lab\\.venv\\src'
)

# 将输出写入文件
with open('run_output.txt', 'w') as f:
    f.write('STDOUT:\n')
    f.write(result.stdout)
    f.write('\nSTDERR:\n')
    f.write(result.stderr)
    f.write('\nReturn code:\n')
    f.write(str(result.returncode))

print('运行完成，输出已写入 run_output.txt')
