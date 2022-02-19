# CLN-SOD-
context-aware learning network for salient object detection


针对换天现有模型分割后会出现一些前景与背景之间的误分，自研一种基于上下文学习的显著目标检测网络（context-aware learning network for salient object detection），该网络通过设计了一个互相关图（cross correlation map）来学习图像上下文中的相关信息，并利用互相关损失函数来不断学习和优化相关效果，以此达到减少前景与背景的误分。具体来说，该网格首先通过一个网络主干，然后提取此时的网络特征，通过卷积和sigmoid函数生成一个内容感知图（context aware map），将内容感知图分为类内感知和类间感知，同时提升类内准确度、降低类间误判度。同时对图像标签，即真实结果（groundtruth），进行缩放，然后热编码，将结果转置并与自身相乘后建立一个相关矩阵（A=LLT），该矩阵即为互相关图，将内容感知图与互相关图放入互相关损失函数来进行相关性学习，此外，对网络主干提取的特征、网络主干提取的特征与内容感知图的乘积，合并后进行卷积，最后通过sigmoid函数生成预测结果，将真实结果与预测结果送入二值交叉熵损失函数作为分割损失。网络对互相关损失函数和二值交叉熵损失函数不断进行梯度负反馈调节，最终达到生成精细度高的分割结果，同时减少前景与背景的误分。

网络结构如下图所示：

![image](https://github.com/lzylyx/CLN-SOD-/blob/main/fig/model_struct.png)