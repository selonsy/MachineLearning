clear;
clc
% ����training sample
% MU1 = [1 2];
% MU2 = [4 6];
% SIGMA1 = [4 4; 4 9];
% SIGMA2 = [4 2; 2 4];

% MU1 = [0 0];
% MU2 = [1 0];
% SIGMA1 = [1 -0.5; -0.5 1];
% SIGMA2 = [0.5 0; 0 1];

% ����training sample
MU1 = [1 3];
MU2 = [3 1];
SIGMA1 = [1.5,0;0,1];
SIGMA2 = [1,0.5;0.5,2];

M1 = mvnrnd(MU1,SIGMA1,100);
M2 = mvnrnd(MU2,SIGMA2,100);

plot(M1(:,1),M1(:,2),'bO',M2(:,1),M2(:,2),'r*')

% 
% %����testing sample
% TEST1 = mvnrnd(MU1,SIGMA1,50);
% TEST2 = mvnrnd(MU2,SIGMA2,50);
%  
% %�������ļ���
% %�м��C
% C = (MU1+MU2)/2;
% C_M = repmat(C,50,1);
%  
% %MUi vector
% TRAIN_V = MU1 - MU2;
% TRAIN_V_M = repmat(TRAIN_V,50,1);
%  
% %TEST vector
% TEST1_V = TEST1 - C_M;
% TEST2_V = TEST2 - C_M;
%  
% %Ԥ���һ�����Լ�
% num1 = 0;
% for (i=1:50)
%     d = dot(TRAIN_V,TEST1_V(i,:));
%     if d >0
%         num1 = num1 + 1;
%     end
% end
%  
% disp(['���Լ�1������������Ϊ��',num2str(length(TEST1_V)),'��ȷ���������Ϊ��',num2str(num1)])
% disp(['���Լ�1��Ԥ��׼ȷ��Ϊ��',num2str(num1/length(TEST1_V))])
%  
% num2 = 0;
% for (i=1:50)
%     d = dot(TRAIN_V,TEST2_V(i,:));
%     if d <0
%         num2 = num2 + 1;
%     end
% end
%  
% disp(['���Լ�2������������Ϊ��',num2str(length(TEST2_V)),'��ȷ���������Ϊ��',num2str(num2)])
% disp(['���Լ�2��Ԥ��׼ȷ��Ϊ��',num2str(num2/length(TEST2_V))])
%  
% %����������ֵ���ߵ�б��
% K = TRAIN_V(2)/TRAIN_V(1);
% %����������ֵ���ߵ��д��ߵ�б��
% k = K/(-1);
%  
% x = min(TEST1):0.1:max(TEST2);
% y = k*(x-C(1))+C(2);
%  
%  plot(TEST1,TEST2,'O',MU1,MU2,'o',x,y)