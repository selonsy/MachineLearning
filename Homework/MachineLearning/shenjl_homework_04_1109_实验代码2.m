function self_mvnrnd(varargin)  %可自定义参数的函数
if(nargin==8)%判定输入参数是否为8
w1=mvnrnd(varargin{1},varargin{2},varargin{3});%第一类
w2=mvnrnd(varargin{5},varargin{6},varargin{7});%第二类
save('w1.mat','w1');save('w2.mat','w2');
figure(1);
plot(w1(:,1),w1(:,2),'bo');%蓝色o为第一类
hold on
plot(w2(:,1),w2(:,2),'g*');%绿色*为第二类
title('200个随机样本，蓝色o为第一类，绿色*为第二类');
w=[w1;w2];
save('w.mat','w');
n1=0;%第一类正确个数
n2=0;%第二类正确个数
figure(2);
%贝叶斯分类器
m=1;
n=1;
R1=[];
R2=[];
R=[];
for i=1:(varargin{3}+varargin{7})
    x=w(i,1);
    y=w(i,2);
    g1=mvnpdf([x,y],varargin{1},varargin{2})*varargin{4};
    g2=mvnpdf([x,y],varargin{5},varargin{6})*varargin{8};
    if g1>g2
        if 1<=i&&i<=varargin{3}
            n1=n1+1;%第一类正确个数
            plot(x,y,'bo');%蓝色o表示正确分为第一类的样本
            hold on;
        else
            plot(x,y,'r^');% 红色的上三角形表示第二类错误分为第一类 selonsy                           
            hold on;
        end
        R1(m,1)=x;R1(m,2)=y;m=m+1;
        R(i)=1;
    else
        if varargin{3}<=i&&i<=(varargin{3}+varargin{7})
            n2=n2+1;%第二类正确个数
            plot(x,y,'g*');%绿色*表示正确分为第二类的样本
            hold on;
        else
            plot(x,y,'rv');% 红色的下三角形表示第一类错误分为第二类 selonsy                           
            hold on;
        end
        R2(n,1)=x;R2(n,2)=y;n=n+1;
        R(i)=0;
    end
end
save('R1.mat','R1');
save('R2.mat','R2');
save('R.mat','R');
r1_rate=n1/varargin{3};%第一类正确率
r2_rate=n2/varargin{7};%第二类正确率
gtext(['第一类正确率：',num2str(r1_rate*100),'%']);
gtext(['第二类正确率：',num2str(r2_rate*100),'%']);
title('最小错误率贝叶斯分类器');
else disp('只能输入参数个数为8');
end