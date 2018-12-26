clear;
clf;
clc;

% -----模拟仿真正弦函数以及增加噪声后的函数----%
N = 10;
NN = 666;
x = linspace(0, 1, N);  % 均分指令,产生0到1之间的N个行矢量
x_fits = linspace(0,1,NN);
sn = sin(2*pi*x); % sn 代表正弦函数
sn_fits = sin(2*pi*x_fits);  % sn_fits用于画出光滑的sn曲线
xn = sn + 0.25*randn(1,N);  % xn代表sn加噪后得到的函数,所加噪声符合高斯分布
                            % randn(1,N)表示生成1*N的,期望为0,标准差为1的正态分布量.
plot(x_fits,sn_fits, 'g', x, xn, 'bo', 'LineWidth',2);  % 
legend('s(n)', 'x(n)');  % 标识图例
% set(get(gca,'title'),'fontname','宋体')
% title('模拟仿真正弦函数以及增加噪声后的函数')  % 注:中文标题会乱码

% -----仿真实现不同阶数的多项式拟合-----%
i=1;
figure;
for M=[0 1 3 9]
    % w代表M阶多项式的系数
    w = polyfit(x, xn, M);  % 返回w为幂次从高到低的多项式系数向量w
    y = polyval(w, x_fits);  % 返回对应自变量x_fits在给定系数w的多项式的值
    
    subplot(2,2,i);   % 生成2*2大小的合并子图
    i=i+1;
    plot(x_fits, sn_fits, 'g', x, xn, 'bo', x_fits, y, 'r', 'LineWidth',2);
    str = ['M=' mat2str(M)];  % 标识M的阶数,mat2str将矩阵转化为字符串
    text(0.6, 0.8, str);  % 在图中指定位置显示字符串str
    legend('s(n)', 'x(n)', 'y');  % 标识图例
end

% -----模拟仿真9阶多项式拟合不同数据集的表现-----%
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
