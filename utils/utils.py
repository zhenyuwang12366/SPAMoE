import re

def to_snake_lower(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # 驼峰 -> 下划线边界
    s = re.sub(r'([A-Z]+)([A-Z][a-z0-9])', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    parts = [p for p in s.split('_') if p]
    if not parts:
        return s
    if parts[0] == 'style':
        # 与原脚本一致：合并 style 子类型，仅区分 a/b
        suffix = parts[-1] if len(parts) >= 2 else 'a'
        if len(parts) >= 3 and parts[1] == 'style':
            return '_'.join(parts[:3])  # style_style_a/b
        return f"style_style_{suffix}"
    return '_'.join(parts)