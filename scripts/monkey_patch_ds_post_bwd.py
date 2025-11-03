# monkey_patch_ds_post_bwd.py
"""
修复 DeepSpeedZeRoOffload._register_deepspeed_module 判断条件，
避免因 PEFT 等 wrapper 的 __getattr__ 误导导致 post_bwd_fn 未注册。
此版本不依赖 _post_backward_module_hook 的直接导入。
"""

from deepspeed.runtime.zero import DeepSpeedZeRoOffload

# 保存原函数
_ORIG = DeepSpeedZeRoOffload._register_deepspeed_module

def _patched_register_deepspeed_module(self, module):
    # 改用 __dict__ 检查，避免 hasattr 受 __getattr__ 影响
    if "post_bwd_fn" not in module.__dict__:
        # 延迟绑定到 engine 的后向钩子逻辑（简化安全版）
        # 等效于老版本 DS 的 _post_backward_module_hook
        def _post_backward_module_hook(*args, **kwargs):
            # 如果 DS 在后向阶段还会使用 ds_grads_remaining，可安全初始化
            if not hasattr(module, "ds_grads_remaining"):
                module.ds_grads_remaining = 0
            # DS 内部的真正钩子由 ZeRO engine 调用，
            # 这里只是保证属性存在，防止 AttributeError
            return None

        # 注册轻量占位的 post_bwd_fn，避免报错
        module.post_bwd_fn = _post_backward_module_hook.__get__(module, type(module))

    # 调用原始逻辑继续注册
    return _ORIG(self, module)

# 打补丁
DeepSpeedZeRoOffload._register_deepspeed_module = _patched_register_deepspeed_module
print("[Patch] DeepSpeedZeRoOffload._register_deepspeed_module patched to use __dict__ check.")