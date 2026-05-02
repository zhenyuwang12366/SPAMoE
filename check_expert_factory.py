"""
Verify whether ExpertFactory in the current environment supports local-type experts.
"""

import os
import sys
import inspect

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuralop.models.expert_factory import ExpertFactory

has_local_expert = hasattr(ExpertFactory, 'create_local_expert')
print(f"ExpertFactory has create_local_expert: {has_local_expert}")

source_code = inspect.getsource(ExpertFactory.create_expert_ensemble)
handles_local = "elif expert_type == 'local':" in source_code
print(f"create_expert_ensemble handles 'local' type: {handles_local}")

if not (has_local_expert and handles_local):
    import neuralop.models.expert_factory
    file_path = inspect.getfile(neuralop.models.expert_factory)
    print(f"ExpertFactory file path: {file_path}")
    print("ExpertFactory may need updates to support local-type experts.")
else:
    print("ExpertFactory already supports local-type experts; configuration should work.")

try:
    from config.seismic_moe_config import SeismicMOEConfig
    config = SeismicMOEConfig()
    print("\nExpert types in config:")
    for i, expert_config in enumerate(config.expert_configs):
        print(f"Expert {i+1} type: {expert_config.get('type', 'unknown')}")
except Exception as e:
    print(f"Error loading config: {e}")

print("\nIf verification passes but you still see errors, copy the modified expert_factory.py into your runtime environment.")
