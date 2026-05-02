import os

def print_tree(start_path, prefix='', max_depth=5):
    if max_depth < 0:
        return
    files = sorted(os.listdir(start_path))
    for i, name in enumerate(files):
        path = os.path.join(start_path, name)
        connector = '├── ' if i < len(files) - 1 else '└── '
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = '│   ' if i < len(files) - 1 else '    '
            print_tree(path, prefix + extension, max_depth - 1)

# 输出到文件
with open('project_structure.txt', 'w', encoding='utf-8') as f:
    from contextlib import redirect_stdout
    with redirect_stdout(f):
        print_tree('.', max_depth=5)  # 根目录向下递归5层
