#pragma region 练习函数
int is_leap_year(int year)
{
	if (year % 4 == 0 && year % 100 != 0)
		return 1;
	else if (year % 400 == 0)
		return 1;
	else
		return 0;
}
int shui_xian_hua()
{
	//定义三位数num,个位数sd,十位数td,百位数hd
	int num, sd, td, hd;
	//循环所有三位数
	for (num = 100; num<1000; num++)
	{
		//获取三位数字num百位上的数字
		hd = num / 100;
		//获取三位数字num十位上的数字
		td = (num / 10) % 10;
		//获取三位数字num个位上的数字
		sd = num % 10;
		//水仙花数的条件是什么？
		if (hd*hd*hd + td*td*td + sd*sd*sd == num)
		{
			printf("水仙花数字：%d\n", num);
		}
	}
	return 0;
}
int add1()
{
	int i = 1;
	int sum = 0;
	while (i <= 100)
	{
		sum += i % 2 == 0 ? -i : i;
		i++;
	}
	printf("sum=%d\n", sum);
	return 0;
}
int triangle_print(int num)
{
	int m, n;
	for (m = 0; m < num; m++)
	{
		for (n = 0; n < num - m - 1; n++)
		{
			printf(" ");
		}
		for (n = 0; n < 2 * m - 1; n++)
		{
			printf("*");
		}
		printf("\n");
	}
	return 0;
}
int triangle_print1(int num)
{
	int m, n;
	for (m = 0; m < num; m++)
	{
		for (n = 0; n < 2 * m - 1; n++)
		{
			printf("*");
		}
		printf("\n");
	}
	return 0;
}
int multi_9()
{
	int i, j;
	for (i = 9; i > 0; i--)
	{
		for (j = 1; j <= i; j++)
		{
			printf("%d*%d=%d ", i, j, i*j);
		}
		printf("\n");
	}
	return 0;
}
int su_shu(int max)
{
	int i;
	int m, n;
	for (m = 2; m <= max; m++)
	{
		for (n = 2; n<m; n++)
		{
			if (m%n == 0)
				break;
		}
		if (m == n)
			printf("%d  ", m);
	}
	return 0;
}
int the_day_of_year(int year, int month, int day)
{
	int sum = 0;
	switch (month)
	{
	case 1:
		sum = 0; break;
	case 2:
		sum = 31; break;
	case 3:
		sum = 59; break;
	case 4:
		sum = 90; break;
	case 5:
		sum = 120; break;
	case 6:
		sum = 151; break;
	case 7:
		sum = 181; break;
	case 8:
		sum = 212; break;
	case 9:
		sum = 243; break;
	case 10:
		sum = 273; break;
	case 11:
		sum = 304; break;
	case 12:
		sum = 334; break;
	default:
		printf("一年只有十二个月哦亲~~");
		break;
	}
	//判断是否闰年
	int is_leap_year(year);
	if (is_leap_year && month > 2)sum++;
	if (day > 0)sum += day;
	printf("%d年%d月%d日是该年的第%d天\n", year, month, day, sum);
	return 0;
}
int factorial(int n)
{
	int result;
	if (n < 0)printf("对不起,您的输入有误,请重新输入!\n");
	else if (n == 1 || n == 0)result = 1;
	else
		result = factorial(n - 1)*n;
	return result;
}
int monkey_eat_peach(int n)
{
	int num; //定义所剩桃子数
	if (n == 10)
	{
		num = 1;
		printf("第%d天所剩桃子%d个\n", n, num);
	}
	else
	{
		num = (monkey_eat_peach(n + 1) + 1) * 2;//这里是不应该用递归呢？
		printf("第%d天所剩桃子%d个\n", n, num); //天数，所剩桃子个数
	}
	return num;
}
int factorial_test(int n)
{
	if (n == 1)
	{
		return 10;
	}
	else
	{
		return factorial_test(n - 1) + 2;
	}
}
int xiaoming_dache(int total_length, int time_go, int time_back)
{
	if (time_go < 0 || time_go>23 || time_back < 0 || time_back>23)
	{
		printf("回家的时间不对吧小明~~");
	}
	int result = 13 + 1;//起步价+燃油附加费
	double price = 2.3;
	if (time_go < 5 || time_go >= 23)
	{
		price *= 1.2;
	}
	result += (total_length - 3)*price;
	price = 2.3;
	if (time_go < 5 || time_go >= 23)
	{
		price *= 1.2;
	}
	result += (total_length - 3)*price;
	printf("小明一天的打车费%d", result);
	return result;
}

//练手的函数2(不带声明)
//数组测试
int array_test()
{
	int my_array[3] = { 1,2,3 };
	printf("%d %d\n", my_array[1], sizeof(my_array));

	return 0;
}
//冒泡排序
int paixu_maopao()
{
	double arr[] = { 1.78, 1.77, 1.82, 1.79, 1.85, 1.75, 1.86, 1.77, 1.81, 1.80 };
	int i, j;
	printf("\n************排队前*************\n");
	for (i = 0; i<10; i++)
	{
		if (i != 9)
			printf("%.2f, ", arr[i]);  //%.2f表示小数点后精确到两位
		else
			printf("%.2f", arr[i]);    //%.2f表示小数点后精确到两位
	}
	for (i = 9; i >= 0; i--)
	{
		for (j = 0; j <= i; j++)
		{
			if (arr[j]>arr[j + 1])      //当前面的数比后面的数大时
			{
				double temp;//定义临时变量temp
				temp = arr[j];//将前面的数赋值给temp
				arr[j] = arr[j + 1];//前后之数颠倒位置
				arr[j + 1] = temp;//将较大的数放在后面    
			}
		}
	}
	printf("\n************排队后*************\n");
	for (i = 0; i<10; i++)
	{
		if (i != 9)
			printf("%.2f, ", arr[i]);  //%.2f表示小数点后精确到两位     
		else
			printf("%.2f", arr[i]);    //%.2f表示小数点后精确到两位
	}
	return 0;
}
//计算二维数组的对角线的和
int count_er_wei()
{
	int arr[3][3] = { { 1,2,3 },{ 4,5,6 },{ 7,8,9 } };
	int i, j;
	int sum = 0;
	for (i = 0; i<3; i++)
	{
		for (j = 0; j<3; j++)
		{
			if ((i == 0 && j == 2) || (i == 2 && j == 0))
			{
				sum += arr[i][j];
			}
			else if (i == j)
			{
				sum += arr[i][j];
			}
			else
			{

			}
		}
	}
	printf("对角线元素之和是：%d\n", sum);
	return 0;
}
/*
在一个长度为10的整型数组里面，保存了班级10个学生的考试成绩。
要求编写5个函数，分别实现计算考试的总分，最高分，最低分，平均分和考试成绩降序排序。
*/
int my_test1()
{
	int score[10] = { 67,98,75,63,82,79,81,91,66,84 };
	int i, j, total = 0, max = score[0], min = score[0];

	for (i = 0; i < 10; i++)
	{
		total += score[i];
		if (i > 0 && score[i] > max)max = score[i];
		if (i > 0 && score[i] < min)min = score[i];
	}
	printf("总分:%d\n最高分:%d\n最低分:%d\n平均分:%d\n", total, max, min, total / 10);
	for (i = 0; i < 10; i++)
	{
		for (j = i + 1; j < 10; j++)
		{
			if (score[i] < score[j])
			{
				int temp;
				temp = score[i];
				score[i] = score[j];
				score[j] = temp;
			}
		}
	}
	for (i = 0; i < 10; i++)
	{
		printf("%d ", score[i]);
	}
	return 0;
}

#pragma endregion