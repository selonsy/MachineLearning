#define _CRT_SECURE_NO_WARNINGS
#include <map>
#include <iostream>
#include <string>

using namespace std;

//求共同好友数量
/*
描述
Alice有N个朋友，Bob有M个朋友。 他们想知道有多少人是他们的共同朋友。 你能帮助他们吗？
输入:
输入以整数T（T <= 20）开始，表示测试样例的数量。 每个测试样例包含三行。 第一行包含两个整数N和M（1 <= N，M <= 50）。 第二行包含N个名字，代表Alice的朋友。 第三行包含M个名字，代表Bob的朋友。 每个名字的长度在3到10之间。 每个名称只包含小写英文字母（'a' - 'z'）。 保证两个不同的人不会共享相同的名称。
输出:
对于每个样例，用一行输出Alice和Bob的共同朋友的数目。
输入样例:
2
4 5
tom lucy kate eva
eva rose joe carl tom
3 3
jimmy jeff emily
zhanglan xiemin xiaran
输出样例
2
0
*/
void findCommonFriendsOfAlice()
{
	map<string, int> friends;
	string str;
	int n, a, b, i, count;
	scanf("%d", &n);

	while (n--)
	{
		count = 0;
		scanf("%d %d", &a, &b);
		for (i = 0; i < a; i++)
		{
			cin >> str;
			friends[str] += 1;
		}

		for (i = 0; i < b; i++)
		{
			cin >> str;
			if (friends[str] > 0)
			{
				count++;
			}
		}
		printf("%d\n", count);
	}
}

//求最长公共子串
/*
1.求最长公共子串
题目:给定两个字符串A和B，同时给定两串的长度n和m。
要求:时间复杂度为O(m*n)的算法(这里的m和n为两串的长度).
样例：给定两个字符串A和B，同时给定两串的长度n和m。
输入:"1AB2345CD",9,"12345EF",7
返回：4

分析:
首先，所求是最大子串，不是子序列，说明是连续的。
LS问题，用动态规划思想，把大问题分解成若干小问题，用矩阵记录状态结果：
若s是两个字符串s1和s2敏感词有的字符，s1中s左侧的字符串为s1'，s2中s左侧的字符串为s2'，那么截止到s为止，LS(s1, s2) = LS(s1', s2')+1，
用矩阵记录结果，例如bab和caba，矩阵如下：
	b	a	b
c	0	0	0
a	0	1	0
b	1	0	2
a	0	2	0
我们用全局变量max记录最大值，同时可以记录最后一次+1时的行号或列号，结合max可以推算出子串。
*/
void maxSubstring()
{
	char A[50], B[50];
	int n, m;
	int max = 0;
	int i, j;

	//用于输出最后的公共子串
	int a, b;
	char C[50] = { '\0' };

	printf("Please input your string and length:");
	scanf("%s%d%s%d", A, &n, B, &m);

	if (n == 0 || m == 0) {
		max = 0;
	}

	int matrix[10][10];//吐槽下这个大小非要常量值
	memset(matrix, 0, sizeof(matrix));


	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			if (A[i] == B[j]) {
				if (i == 0 || j == 0) {
					matrix[i][j] = 1;
				}
				else {
					matrix[i][j] = matrix[i - 1][j - 1] + 1;
				}
				max = (max > matrix[i][j] ? max : matrix[i][j]);
				a = i;
				b = j;
			}
		}
	}
	printf("The max length is %d\n", max);

	for (i = a-max+1; i <= a; i++)
	{
		//字符串拼接
		char temp[2];
		sprintf(temp, "%c", A[i]);
		strcat(C, temp);

		printf("%c",A[i]);
	}
	printf("\nThe common string is %s", C);
}

//二分查找
/**
*  折半查找函数
*
*  @param arr   数组
*  @param len   数组长度
*  @param value 查找元素
*
*  @return 返回查找元素的位置
*/
int searchItem(int arr[], int len, int value) {
	int low = 0, high = len - 1, mid;
	while (low <= high) {
		mid = (low + high) / 2;
		if (value > arr[mid]) {
			low = mid + 1;
		}
		else if (value < arr[mid]) {
			high = mid - 1;
		}
		else {
			return mid;
		}
	}
	return -1;
}
void searchItem_test(int argc, const char * argv[]) {
	//数组必须是有序数组  
	int a[10] = { 1,2,31,45,52,62,73,86,90,100 };
	//查找86元素  
	int l = searchItem(a, 10, 86);
	printf("loc = %d\n", l);
	return ;
}

//正整数分解质因数
/*
2.实现一个正整数分解质因数的功能.
例如：输入90,打印出90=2*3*3*5。
*/
void integerDecompose()
{
	int m, i;
	printf("please imput a mumber:\m");
	scanf("%d", &m);
	printf("%d=", m);
	for (i = 2; i <= m; i++)
	{
		while (m != i)
		{
			if (m%i == 0)
			{
				printf("%d*", i);
				m = m / i;
			}
			else
				break;
		}
	}
	printf("%d", m);
}

//多项式相乘
/*
3.多项式的乘法
exp:
A(X)=2X^3+4
B(x)=4X^4+2X^3
C(X)=A(x)*B(x)=8X^7+4X^6+16X^4+8X^3
思路：
1.创建两个链表,用于存储两个多项式 用链式存储 系数域+指数域+指针域
2.完成两个多项式的乘法
3.打印出新结果

链式相乘：
思路：
1.因为两个链表都是指数递减，所以A(X)递减，B(x)逆置下，递增,why do this?
2.先获取两个最大的指数和 exp_max. 这样的话余下的指数就是都在0~7之间了。
关键来了，遍历相乘本质并不难，但是如何可以找到所有的指数呢？而且还要开辟新的节点来存储没有的指数
解决：用一个新的链来存储结果，从exp_max开始向下查找,每一个可能指数都要遍历到。
这里指数升序+降序的排列就很精妙了。
for(k=exp_max;k>=0;k--)
{
相乘;
判断是否还有同类的系数，有就相加;
}
如何判断呢？就是在步进查找。
若是当前k值,表明该指数找到了，此时就是a,b都后继一位，因为只有这种组合才可能有同样系数
若是当前指数<k,表明则表明要增加系数和，只有a增加
若是当前指数>k,表明要减少系数和，只有b增加
这也就看出了，a,b两个链表指数一个升序一个降序的好处了。这种思路很好

3.归纳总结下：
3.1 求k=exp_max;
3.2 逆置b
3.3 遍历查找 怎么做循环又是个问题
一旦查找到了 =k的情况,然后就继续搜索其他可能性 直到都到NULL节点
*/
typedef struct node
{
	float coef;	//系数
	int exp;	//指数
	struct node *next; //指针域
}listnode;
listnode *CreateList(int n)
{
	listnode *head = (listnode *)malloc(sizeof(listnode)), *p, *pre = head;//head指向头结点，p指向新开辟的节点
	float coef; //系数
	int exp;	//指数
	if (NULL == head)
	{
		printf("开辟头结点失败\n");
		exit(-1);
	}
	head->next = NULL;
	for (int i = 0; i<n; i++)
	{
		if (NULL == (p = (listnode *)malloc(sizeof(listnode))))
		{
			printf("新结点malloc失败\n");
			exit(-1);
		}
		printf("请输入第%d个系数和指数\n", i + 1);
		scanf("%f,%d", &coef, &exp);
		p->coef = coef;
		p->exp = exp;  //更新节点数据
		p->next = NULL;
		//插入节点
		pre->next = p;
		pre = p;
	}
	return  head;	//这里是返回堆的内存区 不是局部变量
}
int printflist(listnode *head)
{
	listnode *p = head->next;
	while (p->next != NULL)
	{
		printf("%1.1f*X^%d+", p->coef, p->exp);// %1.1f格式输出小数点后只保留一位
		p = p->next;
	}
	printf("%1.1f*X^%d\n", p->coef, p->exp);
	return 0;
}
int InverseList(listnode *head)	//逆置链表
{
	listnode *p = head->next, *q; //p指向正要逆置的节点，q指向下一个待逆置的节点
	head->next = NULL;
	while (p)	//当前节点不为空
	{
		q = p->next;//保存下一个节点
		p->next = head->next; //先更新逆置点的 next
		head->next = p;		//在更新head->next 
		p = q;		   //下一轮
	}
	return 0;
}
listnode *MultiplisePoly(listnode *head_a, listnode *head_b)//链式相乘
{
	listnode *head_c, *pa = head_a, *pb = head_b, *pc, *newnode;
	int exp_max; //指数之和最大值
	if (pa->next != NULL && pb->next != NULL)
		exp_max = pa->next->exp + pb->next->exp; //获取最大指数和 
	else return NULL;
	//初始化链表C头结点
	head_c = (listnode *)malloc(sizeof(listnode));
	if (NULL == head_c)
	{
		printf("开辟链表C失败\n");
		exit(-1);//exit用于在程序运行的过程中随时结束程序，函数原型void exit(int status)
	}
	head_c->coef = 0.0;
	head_c->exp = 0;
	head_c->next = NULL;
	pc = head_c;
	InverseList(head_b);	//逆置b链表
	float ceof = 0.0;
	for (int k = exp_max; k >= 0; k--)
	{
		pa = head_a->next; //恢复pa的指向
		while (pa != NULL && pa->exp>k) //首先查找pa的位置 找不大于k的
			pa = pa->next;
		pb = head_b->next;//恢复Pb的指向
		while (pa != NULL && pb != NULL && pa->exp + pb->exp<k)//然后在查找pb的位置 pa+pb的指数和不大于k
			pb = pb->next;
		//经过上面两轮后 pa+pb 的exp<=k
		while (pa != NULL && pb != NULL)//此循环进入后,找到所有的同指数的和相加
		{
			if (k == pa->exp + pb->exp)  //目的就是找等于K
			{
				ceof += pa->coef*pb->coef;
				pa = pa->next;
				pb = pb->next;
			}
			else
			{
				if (pa->exp + pb->exp<k) //小于k 增加pb
					pb = pb->next;
				else
					pa = pa->next; //大于k 减小pa
			}

		}
		if (ceof != 0.0)	//有系数了 就将此节点插入到c链表中
		{
			if (NULL == (newnode = (listnode *)malloc(sizeof(listnode))))
			{
				printf("链表C节点开辟失败");
				exit(-1);
			}
			newnode->coef = ceof;
			newnode->exp = k;
			newnode->next = NULL; //插入节点数据

			pc->next = newnode;
			pc = newnode;			//插入节点		
			ceof = 0.0;
		}
	}

	InverseList(head_b);
	return head_c;
}

void duo_xiang_shi_cheng()
{
	printf("  链表实现多项式的乘法   \n");
	printf("----by lynnbest ----\n\n");

	int n, m;
	printf("请输入A(X)的项数(降幂排列)\n");
	scanf("%d", &m);
	listnode *head_a = CreateList(m);
	printf("A(X)=");
	printflist(head_a);
	printf("请输入B(X)的项数(降幂排列)\n");
	scanf("%d", &n);
	listnode *head_b = CreateList(n);
	//  InverseList(head_b);  
	printf("B(X)=");
	printflist(head_b);
	listnode *head_c = MultiplisePoly(head_a, head_b);
	printf("C(X)=A(X)*B(X)=");
	printflist(head_c);

	//以下为沈金龙版本
	/*int* arr;
	arr = (int*)malloc(sizeof(int)*m*n);
	memset(arr, 0, sizeof(arr));*/
	float arr[100];
	memset(arr, 0, sizeof(arr));
	int i, j;
	listnode *pa = head_a, *pb = head_b;
	for (i = 0; i < m&&pa; i++, pa = pa->next)
	{
		for (j = 0; j < n&&pb; j++, pb = pb->next)
		{
			float coef = pa->next->coef*pb->next->coef;
			int exp = pa->next->exp + pb->next->exp;
			arr[exp] = arr[exp] + coef;
		}
		pb = head_b;
	}
	pa = head_a;
	printf("\nshenjinlong\n");
	for (i = 0; i < 100; i++)
	{
		if (arr[i] > 0)
		{
			printf("%1.1f*X^%d+", arr[i], i);
		}
	}
	printf("\nshenjinlong\n");

	return;
}

//求平面坐标距离
/*
给定平面十个点的坐标，求相互距离中的最小值
*/
typedef struct vector
{
	int v;//纵坐标
	int h;//横坐标
}Vector;
void minDistanceOfPlane()
{
	Vector arr[10] = { { 1,1 },{ 2,2 },{ 3,3 },{ 4,4 },{ 5,5 },{ 1,5 },{ 2,6 },{ 3,7 },{ 4,8 },{ 5,9 } };
	int i, j;
	float min = 100;
	for (i = 0; i < 10; i++)
	{
		for (j = i + 1; j < 10; j++)
		{
			float x = sqrt(pow((arr[i].h - arr[j].h), 2) + pow((arr[i].v - arr[j].v), 2));
			min = x < min ? x : min;
		}
	}
	printf("The min distance is %.2f\n", min);
}

//轮流挑石头问题
/*
Description
有一堆石头，数量为n，鑫胖和dalao两个人轮流拿石头，每次能选择拿一块或两块石头。dalao先手。游戏规则为谁拿了最后一块石头，则谁赢。 假设鑫胖和dalao都很聪明，IQ=250，每次都能做出最佳的选择。 例如，有1-2块石头，则先手的dalao获胜。(直接拿所有石头) 如果有3块石头，则鑫胖获胜。(不管dalao拿几块，鑫胖都能直接拿剩下的石头)
Input
输入的第一行为t，表示t组数据 接下来t行，每行输入一个n，表示有n块石头。
Output
输出谁赢 eg： xinpang win! dalao win!
Simple Input
2 2 3
Somple Output
dalao win!
dalao win!
xinpang win!

分析:谁面临只剩下三个的情况,谁就输了.
*/
void pickStone()
{
	int a, b;
	scanf("%d", &a);

	while (a--)
	{
		scanf("%d", &b);
		if (b % 3 != 0)
		{
			printf("dalao win!\n");
		}
		else
		{
			printf("xinpang win!\n");
		}
	}
}

//求最大公约数和最小公倍数
/*
Description
Give you two integers, and calculate their GCD(Greatest Common Divisor) and LCM(Least Common Multiple). If both of them are positive, print the GCD and LCM. If some are invalid(zero or negative), print invalid.
Sample Input 1
2 10
Sample Output 1
2 10
Sample Input 2
14 0
Sample Output 2
invalid
*/
int gcd(int a, int b)
{
	if (b == 0)
	{
		return a;
	}
	else
	{
		return gcd(b, a%b);//递归
	}
}
int lcm(int a, int b)
{
	return a / gcd(a, b)*b;//两数积除以两数最大公约数
}
void gcd_lcm()
{
	int a, b;
	scanf("%d%d", &a, &b);
	if (a > 0 && b > 0)
		printf("%d %d\n", gcd(a, b), lcm(a, b));
	else
		printf("Invalid\n");
}

//指令移动小球
/*
题目描述
你有一些小球，从左到右依次编号为1,2,3,...,n. 你可以执行两种指令（1或者2）。其中， 1 X Y表示把小球X移动到小球Y的左边， 2 X Y表示把小球X移动到小球Y右边。 指令保证合法，即X不等于Y。 例如，初始状态1,2,3,4,5,6的小球执行1 1 4后，小球1被移动到小球4的左边，即2,3,1,4,5,6。如果再执行2 3 5，结点3将会移到5的右边，即2,1,4,5,3,6。
输入格式
第一行为一个整数t（0<t<10），表示测试用例个数。每个测试用例的第一行为两个整数n（1<n<=500000）和m（0<m<100000），n表示小球的个数，m为指令条数，以下m行每行为一条指令。
输出格式
为每个测试用例单独输出一行，从左到右输出最后序列，每个数字后面跟一个空格。
样例输入
2
6 2
1 1 4
2 3 5
5 1
2 1 5
样例输出
2 1 4 5 3 6
2 3 4 5 1
*/
typedef struct ball_node
{
	int num;
	struct ball_node * next;
}BallNode;
BallNode * create_list(int num)
{
	return NULL;
}
void move_ball(int* nums, int type, int obj, int des, int ball_num)
{
	int i=0;
	for (; i < ball_num; i++)
	{
		if (nums[i] == obj)break;
	}
	for (; i < ball_num; i++)
	{
		nums[i] = nums[i + 1];
		if (nums[i + 1] == des)
		{
			if (1 == type)
			{
				nums[i] = obj;
			}
			else if(2==type) 
			{
				nums[i + 1] = obj;
			}
			break;
		}
	}
}
void moveBallByOrder()
{
	int sample_num, ball_num, order_num;
	int order_type, obj, des;
	int i;
	scanf("%d", &sample_num);
	while (sample_num--)//常用来处理循环输入
	{
		scanf("%d%d", &ball_num, &order_num);
		int * nums = (int*)malloc(sizeof(int)*ball_num);
		for (i = 0; i < ball_num; i++)
		{
			nums[i] = i + 1;
		}
		while (order_num--)
		{
			scanf("%d%d%d", &order_type, &obj, &des);
			move_ball(nums, order_type, obj, des, ball_num);
		}
		for (i = 0; i < ball_num; i++)
		{
			printf("%d ", nums[i]);
		}
		printf("\n");
	}
}

//丢扑克牌游戏
/*
题目描述
桌上有一叠牌，从第一张牌（即位于顶面的牌）开始从上往下依次编号为1~n。当至少还剩两张牌时进行以下操作：把第一张牌扔掉，然后把新的第一张放到整叠牌的最后。输入n,输出每次扔掉的牌，以及最后剩下的牌。
输入格式
第一行为一个整数t（0<t<40），表示测试用例个数。以下t行每行包含一个整数n（0<n<40），为一个测试用例的牌数。
输出格式
为每个测试用例单独输出一行，该行中依次输出每次扔掉的牌以及最后剩下的牌，每张牌后跟着一个空格。
样例输入
2
7
4
样例输出
1 3 5 7 4 2 6
1 3 2 4
*/
int a[100];
void left_move(int n)
{
	int i;
	for (i = 0; i<n - 1; i++)
	{
		a[i] = a[i + 1];
	}
	a[n - 1] = a[0];
	for (i = 0; i<n - 1; i++)
	{
		a[i] = a[i + 1];
	}
}
void init(int n)
{
	int i;
	for (i = 0; i<n; i++)
	{
		a[i] = i + 1;
	}
}
void thorwPoker()
{
#if 0
	freopen("input", "r", stdin);
	freopen("output", "w", stdout);
#endif

	int n, count, m;
	scanf("%d", &n);
	while (n--)
	{
		scanf("%d", &m);
		init(40);
		count = m;
		while (count != 1)
		{
			printf("%d ", a[0]);
			left_move(count); //类似于数组左移
			count--;
		}
		printf("%d \n", a[0]);//输出最后一张牌
	}
}

//猴子选大王
/*
题目描述
猴子选大王，有N只猴子，从1～N进行编号。它们按照编号的顺时针方向，排成一个圆圈，然后从第一只猴子开始报数。第一只猴子报1，以后每只猴子报的数字都是它前面猴子所报数字加1。如果一只猴子报的数字是M，则该猴子出列，下一只猴子重新从1开始报数。剩下的猴子继续排成一个圆圈报数，直到全部的猴子都出列为止。最后一个出列的猴子胜出。
输入格式
The first line is an integer t, indicating the number of test cases. Then there are t lines and each line contains two positive integer N(0<N<=100) and M(0<M<=100).
输出格式
For each test case, print out the number of the Monkey King.
样例输入
2
5 2
4 3
样例输出
3
1
*/
void pickMonkeyKing()
{
	int t, i, count, s, m, n;
	scanf("%d", &t);
	while (t--)
	{
		scanf("%d %d", &n, &m);
		int * nums = (int*)malloc(sizeof(int)*n);//动态声明数组
		memset(nums, 1, sizeof(int)*n);//批处理为1
		count = n;
		s = 1;
		while (count != 1)
		{
			for (i = 1; i <= n; i++)
			{
				if (!nums[i]) continue;//遇到0的就跳过

				if (s == m)//等于目标值,将其置0,s重新计算
				{
					nums[i] = 0;
					count--;
					s = 0;
				}
				s++;
			}

		}
		for (i = 1; i <= n; i++)
		{
			if (nums[i])
			{
				break;
			}
		}
		printf("%d\n", i);
	}
}

//求日期为当年第几天
/*
Description
已知当前日期，请计算这是这一年的第几天。例如2012年12月31日是2012年的第366天（2012是闰年）。 提示：四年一闰，百年不闰，四百年再闰，即能整除4且不能整除100的年为闰年（2004年就是闰年,1900年不是闰年）；世纪年能整除400的是闰年（2000年是闰年，1900年不是闰年）。
Input
分别输入年Y，月M，日D，以整数形式输入，其中1900<=Y<=3000，1<=M<=12，1<=D<=31，三个整数之间用空格隔开。
Output
输出一个整数N，表示当前是一年中的第几天，末尾有一个换行符
Sample Input
2012 12 1
Sample Output
336
*/
char month_day[13] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
int is_leap_year(int y)
{
	if ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)
	{
		return 1;
	}

	return 0;
}
void countDaysOfYear()
{
	int year, month, day, i, sum;
	//循环从输入流读取值，直到遇到EOF为止，等同于while(scanf("%d %d %d", &year, &month, &day)!=EOF)。
	while (~scanf("%d,%d,%d", &year, &month, &day))
	{
		sum = 0;
		for (i = 1; i<month; i++)
		{
			sum += month_day[i];
		}

		sum += day;

		if (month>2 && is_leap_year(year))
		{
			sum++;
		}
		printf("%d\n", sum);
	}
}

//输出所有的完数
/*
Description:
编写程序，输入一个正整数M，输出[1,M]内的所有“完数”。每输出一个数（包括最后一个数）均以换行结束。
所谓“完数”，是指一个数恰好等于它的因子值（因子不包括该数本身）之和。例如6是完数，因为6=1+2+3。
Sample Input:
100
Sample Output:
6
28
*/
void outputWanShu()
{
	int m;
	scanf("%d", &m);
	int temp = 0, i, j;
	for (i = 1; i <= m; i++)
	{
		for (j = 1; j <= i / 2; j++)
		{
			if (i%j == 0)
			{
				temp += j;
			}
		}
		if (temp == i)
		{
			printf("%d\n", i);
		}
		temp = 0;
	}
}

//砝码称重问题
/*
递归函数IsMeasurable(int target, int weights[], int nWeights)，填空题。给定n个砝码，然后再给定指定重量，看是否能用这几个砝码就能确定这个重量
比如，给定砝码分别，1KG 3KG， 那么通过这两个砝码，我们能准确称量1KG 2KG 3KG 4KG的物品。（2KG物品称量的方法就是把1KG的砝码放到和物品一起，然后等于3KG的砝码）
*/
int IsMeasurable(int target, int weights[], int nWeights)
{

	if (nWeights == 0)
	{
		if (target == 0) return 1;
		else return 0;
	}
	int a = IsMeasurable(target - weights[nWeights - 1], weights, nWeights - 1);
	int b = IsMeasurable(target, weights, nWeights - 1);
	int c = IsMeasurable(target + weights[nWeights - 1], weights, nWeights - 1);
	return (a || b || c);
}
void measureByWeight()
{
	int weights[] = { 1,3,5,7,9 };
	int nWeights = 5;
	printf("%d\n", IsMeasurable(20, weights, nWeights));
}

//统计字符数量问题
/*
编程题，编一个子程序，接收三个数组，前两个char型，后一个int型，第二个初始化为字母表，并统计第一个数组上限为1000的字符中各字母出现的频率，记录在第三个数组的对应项中。
题目中主程序给的数组都是没有赋值的，要自己赋值，然后……记得大小写字母都要统计啊
*/
int* _countChars(char des[], char ori[], int num[])
{
	int i;
	for (i = 0; des[i] != NULL; i++)
	{
		num[des[i]]++;
	}
	return num;
}
void countChars()
{
	int i, num[100];
	char des[1000];
	printf("Please input your chars:\n");
	scanf("%s", des);
	memset(num, 0, sizeof(num));
	int* p = _countChars(des, des, num);//接收返回数组的首地址
	for (i = 0; i < 100; i++)
	{
		if (*(p + i) != 0)
			printf("%c:%d\n", i, *(p + i));
	}
}

//字母数字排序问题
/*
设计一个函数，输入一个字符串，将该字符中的数字排在左边，字母排在右边。
*/
void changeStrNumOrder(char * str)
{
	int len = strlen(str);//strlen不计算\0的长度
	int i, j = len - 1;
	char temp;
	for (i = 0; i < len - 1 && i<j; i++)
	{
		if (str[i] >= 48 && str[i] <= 67)
		{
		}
		else if ((str[i] >= 65 && str[i] <= 90) || (str[i] >= 97 && str[i] <= 122))
		{
			temp = str[j];
			str[j] = str[i];
			str[i] = temp;
			i--;
			j--;
		}
	}
	puts(str);
}
void test_changeStrNumOrder()
{
	char str[] = "asd546djch789dgt";
	changeStrNumOrder(str);
}

//个人所得税计算问题
/*
1.计算税率，题目给出了工资salary和保险insurance，根据工资加保险的大小范围不同，所缴纳的税率不同，根据题目要求输出需要交纳的税率.

具体题目不清晰,我将其理解为以下内容:
给定一个人的当月工资,求出他当月需要缴纳的个人所得税,并计算其税率.
//应缴个人所得税的计算公式为=（月应税收入-3500）*税率-速算扣除数
含税级距						税率(%)				速算扣除数
不超过1,500元的               3					0
超过1,500元至4,500元的部分		10					105
超过4,500元至9,000元的部分		20					555
超过9,000元至35,000元的部分	25					1,005
超过35,000元至55,000元的部分	30					2,755
超过55,000元至80,000元的部分	35					5,505
超过80,000元的部分				45					13,505
*/
void personalRate()
{
	float money, insurance;
	printf("Please input your income(monthly):");
	scanf_s("%f", &money);
	printf("Please input your insurance cost(monthly):");
	scanf_s("%f", &insurance);

	if (money < 0 || insurance < 0)printf("your input is error!\n");

	float money_rate = money - insurance - 3500; //保险不交税,3500是免征额
	float money_pay;

	if (money_rate <= 0)printf("your income is too low,you don't need to pay!\n");

	if (money_rate <= 1500)money_pay = money_rate*0.03;
	else if (money_rate > 1500 && money_rate <= 4500)money_pay = money_rate*0.1 - 105;
	else if (money_rate > 4500 && money_rate <= 9000)money_pay = money_rate*0.2 - 555;
	else if (money_rate > 9000 && money_rate <= 35000)money_pay = money_rate*0.25 - 1005;
	else if (money_rate > 35000 && money_rate <= 55000)money_pay = money_rate*0.3 - 2755;
	else if (money_rate > 55000 && money_rate <= 80000)money_pay = money_rate*0.35 - 5505;
	else money_pay = money_rate*0.45 - 13505;

	printf("you need to pay %.2f,the rate is %.2f\%", money_pay, (money_pay / money) * 100);
}

//象棋马走日遍历问题
/*
2.中国象棋之马：这个题坑了我一个小时，大致说的是一个8*8的方阵里面，用户随意输入1到64中的一个数字，这个数字就是中国象棋中马的初始位置，大家都知道马只能走日字，请设计一个算法，给出一个路径，这个路径描述出马走遍棋盘上所有点所经过的路径，而且并不重复。总之这个题我最后没做出来。

分析如下:
>给定一个数字,可能有几种位置,比如48,那么有可能是横六竖八,也可能是横八竖六.
>给定一个居中的点,如横四竖四,那么他可能有八个位置可以走,一方两个.
>采用二维数组,循环操作,走过的路不再走.
>停止条件采用走过的点累积六十四个.
>二维数组的值可以采用先初始化为0,后面走一个加一的方式,比如第一步a[i][j]=1,第二步a[i][j]=2
>输出结果可以采用链表的方式

进一步分析:广度优先遍历
首先知道马在棋盘是怎么走的，根据国际象棋规则，马在一个起始位置共有8个可用的行动位置，当然边界方面需要另外考虑，我们的马的行走必须考虑这8种类可能性，排除不能使用的位置，走可用的位置，当8个位置不可以使用的时，需要考虑返回上一步，这点有点像图的广度优先遍历相同，当马走完所有位置，同时没有可用的位置用于行走的时候遍历结束。

进进一步分析:贪心算法/局部最优调整
其实马踏棋盘的问题很早就有人提出，且早在1823年，J.C.Warnsdorff就提出了一个有名的算法。在每个结点对其子结点进行选取时，优先选择‘出口’最小的进行搜索，‘出口’的意思是在这些子结点中它们的可行子结点的个数，也就是‘孙子’结点越少的越优先跳，为什么要这样选取，这是一种局部调整最优的做法，如果优先选择出口多的子结点，那出口少的子结点就会越来越多，很可能出现‘死’结点（顾名思义就是没有出口又没有跳过的结点），这样对下面的搜索纯粹是徒劳，这样会浪费很多无用的时间，反过来如果每次都优先选择出口少的结点跳，那出口少的结点就会越来越少，这样跳成功的机会就更大一些。这种算法称为为贪心算法，也叫贪婪算法或启发式算法，它对整个求解过程的局部做最优调整，它只适用于求较优解或者部分解，而不能求最优解。这样的调整方法叫贪心策略，至于什么问题需要什么样的贪心策略是不确定的，具体问题具体分析。实验可以证明马遍历问题在运用到了上面的贪心策略之后求解速率有非常明显的提高，如果只要求出一个解甚至不用回溯就可以完成，因为在这个算法提出的时候世界上还没有计算机，这种方法完全可以用手工求出解来，其效率可想而知。

算法思想步骤概括为下面的几点：
①找到一个位置的各个方向的出口（OUT1）。
②对各个方向的出口进行二次出口（OUT2）的搜索。
③记录每个方向上出口（OUT1）的二次出口（OUT2），（OUT2）最小的的那个（OUT1）即为接下来要访问的位置。
④将每次访问过的位置放入容器。
*/
//棋盘数据：  
const int WIDTH = 8;       //棋盘宽和高(国际象棋)
const int HEIGHT = 8;
int board[8][8];  //棋盘数组保存数据为每个位置对应马的路线的第几步  
				  //dir为马的八个方向  
const int dir[8][2] = { { -2,-1 },{ -2,1 },{ -1,2 },{ 1,2 },{ 2,1 },{ 2,-1 },{ 1,-2 },{ -1,-2 } };
//求(i,j)位置的出口,并返回所有出口和对应的出口个数  
int exitn(int i, int j, int s, int a[])
{
	int k, i1, j1;
	int count;  //计数出口的个数  
	for (count = k = 0; k<8; k++)
	{
		i1 = i + dir[(s + k) % 8][0];  //八方向横坐标和纵坐标，(s+k)%8让索引保持在八方向上  
		j1 = j + dir[(s + k) % 8][1];
		if (i1 >= 0 && i1<HEIGHT && j1 >= 0 && j1<WIDTH && board[i1][j1] == 0) //在棋盘的范围内并且没有走过  
			a[count++] = (s + k) % 8;
	}
	return count;
}
//将(i,j)节点以s开始的下一个节点序列中，节点数最小的一个返回  //next(istartX, istartY, start)
int next(int i, int j, int s)
{
	int m, k, go, min, a[8], b[8], temp;
	m = exitn(i, j, s, a);  //当前位置有出口数  
	if (m == 0)                //没有  
		return -1;
	for (min = 8, k = 0; k<m; k++)       //搜寻出口最少的位置  
	{
		temp = exitn(i + dir[a[k]][0], j + dir[a[k]][1], s, b);
		if (temp < min)             //保存最小出口  
		{
			min = temp;
			go = a[k];             //保存方向索引  
		}
	}
	return go;             //返回最少出口位置的方向索引  
}
void horseChess()
{
	int num, istartX, istartY;
	while (1)
	{
		printf("Please input your number:");
		scanf_s("%d", &num);
		if (num <= WIDTH*HEIGHT && num >= 1)break;
	}

	istartX = num % 8 == 0 ? num / 8 - 1 : num / 8;		//起始横坐标
	istartY = num % 8 == 0 ? 7 : (num % 8) - 1;			//起始纵坐标


	int step, flag, start = 1;    //step表示第几步,flag标记下一步的方向，start表示方向索引的初始值  
	memset(board, 0, sizeof(int)*WIDTH*HEIGHT);  //初始化棋盘，0表示没有走过
	board[istartX][istartY] = 1;                 //起始位置的第一步，1表示第一步，N表示第N步  

	printf("\n棋盘中的轨迹：（数字表示第几步，所在位置为棋盘位置）\n");
	for (step = 2; step <= WIDTH*HEIGHT; step++)  //从第二步开始，直到走满整个棋盘  
	{
		if ((flag = next(istartX, istartY, start)) == -1) //返回-1，没有找到出口  
			break;
		istartX += dir[flag][0]; //下一步的起始坐标  
		istartY += dir[flag][1];
		board[istartX][istartY] = step; //保存当前步到棋盘作为标记  
	}
	int i, j;
	for (i = 0; i<HEIGHT; i++) //输出棋盘保存的路径  
	{
		for (j = 0; j<WIDTH; j++)
			printf("%5d", board[i][j]); //5格对齐  
		printf("\n");
	}
}

typedef struct chess {
	int heng;
	int shu;
	struct chess* next;
}Chess, *PChess;
PChess reverse(PChess head) {
	PChess p, q, pr;
	p = head->next;
	q = NULL;
	head->next = NULL;
	while (p) {
		pr = p->next;
		p->next = q;
		q = p;
		p = pr;
	}
	head->next = q;
	return head;
}

/*
可能的变种题:
设有下图所示的一个棋盘，在棋盘上的A点，有一个中国象棋的马，并约定马走的规则,从A点跳至B点。
规则：1. 马走日字 2. 马只能向右走。找出从A跳到B的某一途径。
1）编写一个函数，输入A点的x和 y值，按规则跳马至B点。
2）在main函数中通过键盘输入x,y的初值，打印马跳过的轨迹和步数。
3）采用递归的思路设计该函数。
*/
//棋盘尺寸
#define X 8
#define Y 8
//B点坐标
#define BX 0
#define BY 7
int pos[100];
int idx = 0;
int drt[4][2] = { { 2,1 },{ 1,2 },{ -1,2 },{ -2,1 } };//仅允许向右侧跳
int expand(int x, int y)
{
	int i, xn, yn;

	if (x == BX && y == BY)
		return 1;
	if (x<0 || y<0 || x >= X || y >= Y)
		return 0;

	for (i = 0; i<4; i++)
	{
		xn = x + drt[i][0];
		yn = y + drt[i][1];
		pos[2 * idx] = xn;
		pos[2 * idx + 1] = yn;
		idx++;
		if (expand(xn, yn))
			return 1;
		else
			idx--;
	}
	return 0;
}
void horseChess1()
{
	int x, y, i, j;
	int ary[X][Y];

	memset(ary, 0, sizeof(ary));

	printf("请输入起始点(x,y)\n");
	scanf_s("%d,%d", &x, &y);
	pos[0] = x;
	pos[1] = y;
	idx = 1;

	if (expand(x, y))
	{
		for (i = 0; i < idx; i++)
			ary[pos[2 * i]][pos[2 * i + 1]] = i + 1;
		for (i = 0; i < X; i++)
		{
			for (j = 0; j < Y; j++)
				printf("%d ", ary[i][j]);
			printf("\n");
		}
		printf("共计%d步\n", idx);
	}
	else
	{
		printf("不能从A到达B\n");
	}


	//int num, a[8][8], i, j, m, n, k = 0;
	//printf("Please input your number:");
	//scanf_s("%d", &num);

	////初始化二维数组
	//for (m = 0; m < 8; m++)
	//{
	//	for (n = 0; n < 8; n++)
	//	{
	//		a[m][n] = 0;
	//	}
	//}

	//i = num % 8 == 0 ? num / 8 - 1 : num / 8;		//起始横坐标
	//j = num % 8 == 0 ? 7 : (num % 8) - 1;	//起始纵坐标

	//a[i][j] = 1;		//第一步
	//k++;
	//PChess head = (PChess)malloc(sizeof(Chess));  //初始化头结点
	//head->heng = i;
	//head->shu = j;
	//head->next = NULL;

	//int count = 0;
	//for (m = 0; m < 64; m++)
	//{
	//	int flag = 0;
	//	count++;
	//	if ((i - 2) >= 0 && (j - 1) >= 0 && a[i - 2][j - 1] == 0)
	//	{
	//		i = i - 2; j = j - 1;
	//	}
	//	else if ((i - 2) >= 0 && (j + 1) >= 0 && a[i - 2][j + 1] == 0)
	//	{
	//		i = i - 2; j = j + 1;
	//	}
	//	else if ((i - 1) >= 0 && (j - 2) >= 0 && a[i - 1][j - 2] == 0)
	//	{
	//		i = i - 1; j = j - 2;
	//	}
	//	else if ((i + 1) >= 0 && (j - 2) >= 0 && a[i + 1][j - 2] == 0)
	//	{
	//		i = i + 1; j = j - 2;
	//	}
	//	else if ((i + 2) >= 0 && (j - 1) >= 0 && a[i + 2][j - 1] == 0)
	//	{
	//		i = i + 2; j = j - 1;
	//	}
	//	else if ((i + 2) >= 0 && (j + 1) >= 0 && a[i + 2][j + 1] == 0)
	//	{
	//		i = i + 2; j = j + 1;
	//	}
	//	else if ((i - 1) >= 0 && (j + 2) >= 0 && a[i - 1][j + 2] == 0)
	//	{
	//		i = i - 1; j = j + 2;
	//	}
	//	else if ((i + 1) >= 0 && (j + 2) >= 0 && a[i + 1][j + 2] == 0)
	//	{
	//		i = i + 1; j = j + 2;
	//	}
	//	else { flag = 1; }

	//	if (flag == 0)
	//	{
	//		//可以跳
	//		a[i][j] = ++k;
	//		PChess node = (PChess)malloc(sizeof(Chess));
	//		node->heng = i;
	//		node->shu = j;
	//		node->next = head;
	//		head = node;
	//	}
	//	/*else
	//	{
	//		if (k == 64)break;
	//		else m = 0;

	//		if (count == 100000) { printf("over time 10000!\n"); break; }
	//	}*/
	//}

	////打印出马跳动的轨迹
	//for ( i = 0; i < 8; i++)
	//{
	//	for (j = 0; j < 8; j++)
	//	{
	//		printf("%4d ",a[i][j]);
	//	}
	//	printf("\n");
	//}

	////按先后顺序输出
	//head = reverse(head);
	//PChess p = head;
	//k = 0;
	//while (p)
	//{
	//	printf("横 %d,竖 %d\n", p->heng, p->shu);
	//	p = p->next;
	//	k++;
	//}
	//printf("total is %d",k);
}

//螺旋数组问题
/*
3.输出螺旋数组:
Input array size N (1~20)
Using Two-Dimensional array
输出一个数字n，输出n阶螺旋数组。
所谓螺旋就是指下面举得这个例子,此时n为4
1   2  3  4
12  13 14 5
11  16 15 6
10  9  8  7
*/
void spiralArray(int num)
{
	int a[20][20], count, i, j, k, kend, lend;

	kend = num / 2;//轮数,也即圈数
	lend = num;
	count = 1;
	for (i = 0, j = 0, k = 0; k<kend; k++)
	{
		//一轮即一个外环圈
		a[i][j] = count++;
		//输出顶行
		for (j++; j<lend; j++)
			a[i][j] = count++;
		//输出最右一列(除顶行的末元素)
		for (i++, j--; i<lend; i++)
			a[i][j] = count++;
		//输出最底一行(除最右行的末元素)
		for (i--, j--; j >= k; j--)
			a[i][j] = count++;
		//输出第一列(除顶行和底行的头元素)
		for (i--, j++; i>k; i--)
			a[i][j] = count++;

		i++; j++; lend--;
	}
	//奇数螺旋组的中心元素
	if (num % 2 != 0) a[i][j] = count;
	for (i = 0; i<num; i++)
	{
		for (j = 0; j<num; j++)
			printf("%4d ", a[i][j]);
		printf("\n");
	}
}
void spiralArray1(int num)
{
	//c++版本
	/*int num, count, i, j, k, kend, lend;
	printf("Input a number[1-20]: ");
	scanf("%d", &num);

	int *a = new int[num*num];
	kend = num / 2;
	lend = num;
	count = 1;
	for (i = 0, j = 0, k = 0; k<kend; k++)
	{
	a[i*num + j] = count++;

	for (j++; j<lend; j++)
	a[i*num + j] = count++;

	for (i++, j--; i<lend; i++)
	a[i*num + j] = count++;

	for (i--, j--; j >= k; j--)
	a[i*num + j] = count++;

	for (i--, j++; i>k; i--)
	a[i*num + j] = count++;

	i++; j++; lend--;
	}
	if (num % 2 != 0) a[i*num + j] = count;
	for (i = 0; i<num; i++)
	{
	for (j = 0; j<num; j++)
	printf("%4d ", a[i*num + j]);
	printf("\n");
	}
	delete[]a;*/
}
void spiralArray2(int num)
{
	//C++版本
	//int **a;
	//int num, count, i, j, k, kend, lend;
	//printf("Input a number[1-20]: ");
	//scanf("%d", &num);

	//a = new int *[num]; //动态申请指针数组  
	//for (i = 0; i<num; i++)
	//{
	//	a[i] = new int[num];    //动态申请数组  
	//}

	//kend = num / 2;
	//lend = num;
	//count = 1;
	//for (i = 0, j = 0, k = 0; k<kend; k++)
	//{
	//	a[i][j] = count++;

	//	for (j++; j<lend; j++)
	//		a[i][j] = count++;

	//	for (i++, j--; i<lend; i++)
	//		a[i][j] = count++;

	//	for (i--, j--; j >= k; j--)
	//		a[i][j] = count++;

	//	for (i--, j++; i>k; i--)
	//		a[i][j] = count++;

	//	i++; j++; lend--;
	//}
	//if (num % 2 != 0) a[i][j] = count;
	//for (i = 0; i<num; i++)
	//{
	//	for (j = 0; j<num; j++)
	//		printf("%4d ", a[i][j]);
	//	printf("\n");
	//}

	//for (i = 0; i<num; i++)
	//{
	//	delete[]a[i];
	//}
	//delete[]a;
}

//括号匹配问题
/*
5.括号匹配，就是数据结构中的内容，大家应该都知道，这次只不过要你编程实现.

分析如下:
1.遇到[和(直接压入栈.
2.遇到]和)则从栈顶往下尝试匹配,遇到第一个[和(就表示匹配成功,栈顶指针下移.如果一直到栈底都没匹配到,则直接跳出循环,返回失败.
*/

int MatchCheck(char a[], int len) {
	int flag = 0;
	char s[10000];
	int top, i;
	char temp;
	// 初始化一个栈
	top = 0;
	for (i = 0; i<len; i++) {
		if (a[i] == '[') { // 如果是左括号直接入栈
			s[++top] = a[i];
			continue;
		}
		if (a[i] == ']') { // 如果是右括号，则尝试匹配
			temp = s[top];
			if (temp == '[') {
				flag = 1;
				top--;
				continue;
			}
			else {
				flag = 0;
				break;
			}
		}

		if (a[i] == '(') { // 如果是左括号直接入栈
			s[++top] = a[i];
			continue;
		}
		if (a[i] == ')') { // 如果是右括号，则尝试匹配
			temp = s[top];
			if (temp == '(') {
				flag = 1;
				top--;
				continue;
			}
			else {
				flag = 0;
				break;
			}
		}
	}
	if (flag && (top == 0)) {
		return 1;
	}
	else {
		return 0;
	}

}
void bracketsMatch()
{
	int result;
	char a[100];
	scanf("%s", &a);
	result = MatchCheck(a, strlen(a));
	if (result)printf("match success!\n");
	else printf("failed!\n");
	return;
}

//互素问题
/*
题目:还有一个是，素数对，给一组数据，问其中互素的元素最多有多少。

互素,又称互质,最早是初等数论中的概念：
若n个整数a1,a2,…,an的最大公因数为1,就称这n个整数互素．

素数(质数)定义为在大于1的自然数中，除了1和它本身以外不再有其他因数。

*/
int isPrimeNumber(int num)
{
	int i, k = (int)sqrt(num);
	for (i = 2; i <= k; i++)
		if (num%i == 0)
			break;
	if (i > k)return 1;
	//printf("%d 是素数。\n", num);
	else return 0;
	//printf("%d 不是素数。\n", num);
}
void primeNumber()
{
	int max = 100;

	int i;
	int m, n;
	for (m = 2; m <= max; m++)
	{
		//isPrimeNumber(m);
		/*for (n = 2; n<m; n++)
		{
		if (m%n == 0)
		break;
		}
		if (m == n)
		printf("%d  ", m);*/
		if (isPrimeNumber(m))printf("%4d", m);
	}
	return;
}

//字符串最大距离问题
/*
给出一个字符串，求出字符串中不同字符相差的最大距离。
*/
void findMaxDistanceOfStringChar()
{

}