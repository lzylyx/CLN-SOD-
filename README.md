# CLN-SOD-
context-aware learning network for salient object detection


��Ի�������ģ�ͷָ������һЩǰ���뱳��֮�����֣�����һ�ֻ���������ѧϰ������Ŀ�������磨context-aware learning network for salient object detection����������ͨ�������һ�������ͼ��cross correlation map����ѧϰͼ���������е������Ϣ�������û������ʧ����������ѧϰ���Ż����Ч�����Դ˴ﵽ����ǰ���뱳������֡�������˵������������ͨ��һ���������ɣ�Ȼ����ȡ��ʱ������������ͨ�������sigmoid��������һ�����ݸ�֪ͼ��context aware map���������ݸ�֪ͼ��Ϊ���ڸ�֪������֪��ͬʱ��������׼ȷ�ȡ�����������жȡ�ͬʱ��ͼ���ǩ������ʵ�����groundtruth�����������ţ�Ȼ���ȱ��룬�����ת�ò���������˺���һ����ؾ���A=LLT�����þ���Ϊ�����ͼ�������ݸ�֪ͼ�뻥���ͼ���뻥�����ʧ���������������ѧϰ�����⣬������������ȡ������������������ȡ�����������ݸ�֪ͼ�ĳ˻����ϲ�����о�������ͨ��sigmoid��������Ԥ����������ʵ�����Ԥ���������ֵ��������ʧ������Ϊ�ָ���ʧ������Ի������ʧ�����Ͷ�ֵ��������ʧ�������Ͻ����ݶȸ��������ڣ����մﵽ���ɾ�ϸ�ȸߵķָ�����ͬʱ����ǰ���뱳������֡�

����ṹ����ͼ��ʾ��

![image](https://github.com/lzylyx/CLN-SOD-/blob/main/fig/model_struct.png)

������չʾ��

<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t1.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t1.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t1_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t2.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t2.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t2_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t3.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t3.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t3_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t4.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t4.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t4_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t5.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t5.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t5_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t6.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t6.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t6_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t7.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t7.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t7_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t8.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t8.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t8_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/test_images/t9.jpg" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t9.png" width="100" height="150" border="2" hspace="10">
<img src="https://github.com/lzylyx/CLN-SOD-/blob/main/fusion_results/t9_fusion.jpg" width="100" height="150" border="2" hspace="10"><br/>


ʹ��˵����

ģ����1��TITAN X GPU�Ͻ���ѵ������֤���������ÿ����PyTorch1.1.0��

���������

���ṩ�Ĵ������ѧ���о�Ŀ��ʹ�á�������Ըü�������Ȥ����Email��ϵLzy_Lyx@163.com��  


