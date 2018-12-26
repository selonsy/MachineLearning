function self_mvnrnd(varargin)  %���Զ�������ĺ���
if(nargin==8)%�ж���������Ƿ�Ϊ8
w1=mvnrnd(varargin{1},varargin{2},varargin{3});%��һ��
w2=mvnrnd(varargin{5},varargin{6},varargin{7});%�ڶ���
save('w1.mat','w1');save('w2.mat','w2');
figure(1);
plot(w1(:,1),w1(:,2),'bo');%��ɫoΪ��һ��
hold on
plot(w2(:,1),w2(:,2),'g*');%��ɫ*Ϊ�ڶ���
title('200�������������ɫoΪ��һ�࣬��ɫ*Ϊ�ڶ���');
w=[w1;w2];
save('w.mat','w');
n1=0;%��һ����ȷ����
n2=0;%�ڶ�����ȷ����
figure(2);
%��Ҷ˹������
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
            n1=n1+1;%��һ����ȷ����
            plot(x,y,'bo');%��ɫo��ʾ��ȷ��Ϊ��һ�������
            hold on;
        else
            plot(x,y,'r^');% ��ɫ���������α�ʾ�ڶ�������Ϊ��һ�� selonsy                           
            hold on;
        end
        R1(m,1)=x;R1(m,2)=y;m=m+1;
        R(i)=1;
    else
        if varargin{3}<=i&&i<=(varargin{3}+varargin{7})
            n2=n2+1;%�ڶ�����ȷ����
            plot(x,y,'g*');%��ɫ*��ʾ��ȷ��Ϊ�ڶ��������
            hold on;
        else
            plot(x,y,'rv');% ��ɫ���������α�ʾ��һ������Ϊ�ڶ��� selonsy                           
            hold on;
        end
        R2(n,1)=x;R2(n,2)=y;n=n+1;
        R(i)=0;
    end
end
save('R1.mat','R1');
save('R2.mat','R2');
save('R.mat','R');
r1_rate=n1/varargin{3};%��һ����ȷ��
r2_rate=n2/varargin{7};%�ڶ�����ȷ��
gtext(['��һ����ȷ�ʣ�',num2str(r1_rate*100),'%']);
gtext(['�ڶ�����ȷ�ʣ�',num2str(r2_rate*100),'%']);
title('��С�����ʱ�Ҷ˹������');
else disp('ֻ�������������Ϊ8');
end