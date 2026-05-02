"""
验证当前环境中的ExpertFactory类是否支持local类型专家
"""

import os
import sys
import inspect

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入ExpertFactory
from neuralop.models.expert_factory import ExpertFactory

# 检查是否有create_local_expert方法
has_local_expert = hasattr(ExpertFactory, 'create_local_expert')
print(f"ExpertFactory是否有create_local_expert方法: {has_local_expert}")

# 检查create_expert_ensemble方法是否处理local类型
source_code = inspect.getsource(ExpertFactory.create_expert_ensemble)
handles_local = "elif expert_type == 'local':" in source_code
print(f"create_expert_ensemble方法是否处理local类型: {handles_local}")

# 如果不支持local类型，显示文件位置
if not (has_local_expert and handles_local):
    import neuralop.models.expert_factory
    file_path = inspect.getfile(neuralop.models.expert_factory)
    print(f"ExpertFactory文件位置: {file_path}")
    print("ExpertFactory类可能需要更新以支持local类型专家。")
else:
    print("ExpertFactory类已经支持local类型专家，配置应该可以正常工作。")

# 显示配置
try:
    from config.seismic_moe_config import SeismicMOEConfig
    config = SeismicMOEConfig()
    print("\n配置中的专家类型:")
    for i, expert_config in enumerate(config.expert_configs):
        print(f"专家 {i+1} 类型: {expert_config.get('type', 'unknown')}")
except Exception as e:
    print(f"加载配置时出错: {e}")

print("\n如果验证通过但仍然出错，可能需要将修改后的expert_factory.py文件复制到运行环境中。") 