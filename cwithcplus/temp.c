#pragma region ��ϰ����
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
	//������λ��num,��λ��sd,ʮλ��td,��λ��hd
	int num, sd, td, hd;
	//ѭ��������λ��
	for (num = 100; num<1000; num++)
	{
		//��ȡ��λ����num��λ�ϵ�����
		hd = num / 100;
		//��ȡ��λ����numʮλ�ϵ�����
		td = (num / 10) % 10;
		//��ȡ��λ����num��λ�ϵ�����
		sd = num % 10;
		//ˮ�ɻ�����������ʲô��
		if (hd*hd*hd + td*td*td + sd*sd*sd == num)
		{
			printf("ˮ�ɻ����֣�%d\n", num);
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
		printf("һ��ֻ��ʮ������Ŷ��~~");
		break;
	}
	//�ж��Ƿ�����
	int is_leap_year(year);
	if (is_leap_year && month > 2)sum++;
	if (day > 0)sum += day;
	printf("%d��%d��%d���Ǹ���ĵ�%d��\n", year, month, day, sum);
	return 0;
}
int factorial(int n)
{
	int result;
	if (n < 0)printf("�Բ���,������������,����������!\n");
	else if (n == 1 || n == 0)result = 1;
	else
		result = factorial(n - 1)*n;
	return result;
}
int monkey_eat_peach(int n)
{
	int num; //������ʣ������
	if (n == 10)
	{
		num = 1;
		printf("��%d����ʣ����%d��\n", n, num);
	}
	else
	{
		num = (monkey_eat_peach(n + 1) + 1) * 2;//�����ǲ�Ӧ���õݹ��أ�
		printf("��%d����ʣ����%d��\n", n, num); //��������ʣ���Ӹ���
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
		printf("�ؼҵ�ʱ�䲻�԰�С��~~");
	}
	int result = 13 + 1;//�𲽼�+ȼ�͸��ӷ�
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
	printf("С��һ��Ĵ򳵷�%d", result);
	return result;
}

//���ֵĺ���2(��������)
//�������
int array_test()
{
	int my_array[3] = { 1,2,3 };
	printf("%d %d\n", my_array[1], sizeof(my_array));

	return 0;
}
//ð������
int paixu_maopao()
{
	double arr[] = { 1.78, 1.77, 1.82, 1.79, 1.85, 1.75, 1.86, 1.77, 1.81, 1.80 };
	int i, j;
	printf("\n************�Ŷ�ǰ*************\n");
	for (i = 0; i<10; i++)
	{
		if (i != 9)
			printf("%.2f, ", arr[i]);  //%.2f��ʾС�����ȷ����λ
		else
			printf("%.2f", arr[i]);    //%.2f��ʾС�����ȷ����λ
	}
	for (i = 9; i >= 0; i--)
	{
		for (j = 0; j <= i; j++)
		{
			if (arr[j]>arr[j + 1])      //��ǰ������Ⱥ��������ʱ
			{
				double temp;//������ʱ����temp
				temp = arr[j];//��ǰ�������ֵ��temp
				arr[j] = arr[j + 1];//ǰ��֮���ߵ�λ��
				arr[j + 1] = temp;//���ϴ�������ں���    
			}
		}
	}
	printf("\n************�ŶӺ�*************\n");
	for (i = 0; i<10; i++)
	{
		if (i != 9)
			printf("%.2f, ", arr[i]);  //%.2f��ʾС�����ȷ����λ     
		else
			printf("%.2f", arr[i]);    //%.2f��ʾС�����ȷ����λ
	}
	return 0;
}
//�����ά����ĶԽ��ߵĺ�
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
	printf("�Խ���Ԫ��֮���ǣ�%d\n", sum);
	return 0;
}
/*
��һ������Ϊ10�������������棬�����˰༶10��ѧ���Ŀ��Գɼ���
Ҫ���д5���������ֱ�ʵ�ּ��㿼�Ե��ܷ֣���߷֣���ͷ֣�ƽ���ֺͿ��Գɼ���������
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
	printf("�ܷ�:%d\n��߷�:%d\n��ͷ�:%d\nƽ����:%d\n", total, max, min, total / 10);
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