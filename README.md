# 说明
## 项目1：PracticalTraining2
**内容**

* 分割、分割分类联合学习

**文件介绍**

1. 实验早期的不同网络设计实验：`previous文件夹下文件`
2. 基本文件：`layers.py（基本网络结构）、Load.py（数据加载）` 
3. 损失函数文件：`loss_dice.py、loss_msssim.py `
4. 独立的分割：`Train.py、Test.py、UNet_3Plus.py `
5. 分割分类联合学习：`Train_Stage2.py、Test_Stage2.py、Model.py` 
6. 测试模块：`detect.py `
7. 其他不重要的：`Others.py`


**备注**
- 网络训练时的模型、损失变化图等，都将输出到`data/outputs`文件夹下
- 网络训练无需预处理过程，但需指定与`所给训练集`文件夹结构类似的训练集
- 受实验条件限制，目前未进行联合训练，联合学习模型`class Model`的训练被割裂

## 项目2：classify
**内容**

* 独立的分类