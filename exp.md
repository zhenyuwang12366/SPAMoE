1. 单个专家模型在单数据集上的表现。10 * 4 = 40
2. 单个专家模型在全数据集上的表现。1 * 4 = 4
3. OpenFWI其他模型在单数据集上的表现
4. OpenFWI在全数据集上的表现
5. 数值方法在单数据集上的表现
6. 数值方法在全数据集上的表现

7. 没有encoder(直接resize)/不同encoder的单专家在单数据集上的表现
8. encoder在全数据集上的训练, 使用encoder+fno直接训练，仅保存encoder
9. moe在全数据集上的表现 

两个MoE模式：group 或 velocity_type

- 对于group模式：
  - 两种路由方式：普通路由 或 任务感知路由
  - 四种组间融合方式：Linear, Attention, Sum, 强弱激活融合
    - 对于强弱激活融合方式：
      - 强专家类内：四种融合方式：Linear, Attention, Sum, Mean
      - 弱专家类内：四种融合方式：Linear, Attention, Sum, Mean

10.    Marmousi数据集进行推理，与数值方法对比

体现模型优势：
1. 对于单速度图类型效果好
2. 同时具有泛化性，能够很好的处理未知地形