# Exp

1. 单个专家模型在单数据集上的表现。10 * 4 = 40
2. 单个专家模型在全数据集上的表现。1 * 4 = 4
3. OpenFWI其他模型在单数据集上的表现，网站上面抄
4. 数值方法在单数据集上的表现

5. 没有encoder(直接resize)/不同encoder(vit/convnext)的单专家在单数据集上的表现 3 * 10 = 30
6. encoder在全数据集上的训练, 使用encoder+fno直接训练，仅保存encoder 1
7. moe在全数据集上的表现
   1. velocity_type
   2. group
      1. 普通路由 + Sum/Attention
      2. 普通路由 + 强弱激活融合 + Sum/Attention
      3. 任务感知路由 + Sum  <--->   普通路由 + Sum
8. Marmousi 数据集进行推理，与数值方法对比

体现模型优势：

1. 对于单速度图类型效果好
2. 同时具有泛化性，能够很好的处理未知地形
