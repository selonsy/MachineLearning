clear;
clf;
clc;

% -----ģ��������Һ����Լ�����������ĺ���----%
N = 10;
NN = 666;
x = linspace(0, 1, N);  % ����ָ��,����0��1֮���N����ʸ��
x_fits = linspace(0,1,NN);
sn = sin(2*pi*x); % sn �������Һ���
sn_fits = sin(2*pi*x_fits);  % sn_fits���ڻ����⻬��sn����
xn = sn + 0.25*randn(1,N);  % xn����sn�����õ��ĺ���,�����������ϸ�˹�ֲ�
                            % randn(1,N)��ʾ����1*N��,����Ϊ0,��׼��Ϊ1����̬�ֲ���.
plot(x_fits,sn_fits, 'g', x, xn, 'bo', 'LineWidth',2);  % 
legend('s(n)', 'x(n)');  % ��ʶͼ��
% set(get(gca,'title'),'fontname','����')
% title('ģ��������Һ����Լ�����������ĺ���')  % ע:���ı��������

% -----����ʵ�ֲ�ͬ�����Ķ���ʽ���-----%
i=1;
figure;
for M=[0 1 3 9]
    % w����M�׶���ʽ��ϵ��
    w = polyfit(x, xn, M);  % ����wΪ�ݴδӸߵ��͵Ķ���ʽϵ������w
    y = polyval(w, x_fits);  % ���ض�Ӧ�Ա���x_fits�ڸ���ϵ��w�Ķ���ʽ��ֵ
    
    subplot(2,2,i);   % ����2*2��С�ĺϲ���ͼ
    i=i+1;
    plot(x_fits, sn_fits, 'g', x, xn, 'bo', x_fits, y, 'r', 'LineWidth',2);
    str = ['M=' mat2str(M)];  % ��ʶM�Ľ���,mat2str������ת��Ϊ�ַ���
    text(0.6, 0.8, str);  % ��ͼ��ָ��λ����ʾ�ַ���str
    legend('s(n)', 'x(n)', 'y');  % ��ʶͼ��
end

% -----ģ�����9�׶���ʽ��ϲ�ͬ���ݼ��ı���-----%
i = 1;
figure;
for N=[10,20,50,100]
    x = linspace(0, 1, N);
    sn = sin(2*pi*x);
    xn = sn + 0.25*randn(1,N);
    w = polyfit(x, xn, 9);
    y = polyval(w, x_fits);
    
    subplot(2,2,i);
    i = i+1;
    plot(x_fits, sn_fits,'g', x, xn, 'bo', x_fits, y, 'r', 'LineWidth',2);
    str = ['N=' mat2str(N)];
    text(0.6, 0.8, str);    
    legend('s(n)', 'x(n)', 'y');  
end
