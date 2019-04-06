#define _CRT_SECURE_NO_WARNINGS
/*
*����˳��ջ����Թ�����
*/

//ԭʼ����
//http://blog.csdn.net/g15827636417/article/details/52749667

#include<stdio.h>
#include<malloc.h>
#define MAXSIZE 100
#define m 6
#define n 8
//��ջ�е�Ԫ�ؽ��ж���,dΪ����
typedef struct {
	int x, y, d;
}point;
//��ջ�Ľṹ���ж���
typedef struct {
	point data[MAXSIZE];
	int top;
}MazeStack;
//���ƶ�������ж��壬������е���ƶ�
typedef struct {
	int x, y;
}item;
//��ջ����Ϊ��ջ
void setNULL(MazeStack *s) {
	s->top = -1;
}
//�ж�ջ�Ƿ�Ϊ��
bool isEmpty(MazeStack *s) {
	if (s->top >= 0) return false;
	else return true;
}
//��ջ����
MazeStack * push(MazeStack *s, point x) {
	if (s->top>MAXSIZE - 1) {
		printf("ջ�������\n");
		return s;
	}
	else {
		s->top++;
		s->data[s->top] = x;
		return s;

	}
}
//��ջ����
point * pop(MazeStack *s) {
	if (isEmpty(s)) {
		printf("ջΪ�գ�\n");
		return NULL;
	}
	else {
		s->top--;
		return &(s->data[s->top + 1]);
	}
}
//ȡջ��Ԫ��
point * getTop(MazeStack *s) {
	if (isEmpty(s)) {
		printf("ջΪ�գ�\n");
		return NULL;
	}
	else {
		return &(s->data[s->top]);
	}
}
//���ƶ���λ�ý��ж���
void defineMove(item xmove[8]) {
	xmove[0].x = 0, xmove[0].y = 1;
	xmove[1].x = 1, xmove[1].y = 1;
	xmove[2].x = 1, xmove[2].y = 0;
	xmove[3].x = 1, xmove[3].y = -1;
	xmove[4].x = 0, xmove[4].y = -1;
	xmove[5].x = -1, xmove[5].y = -1;
	xmove[6].x = 1, xmove[6].y = 0;
	xmove[7].x = -1, xmove[7].y = 1;
}
//�������в����Ĳ���
int maze_test() {
	//���Թ����ж���
	int maze[m + 2][n + 2], x, y, i, j, d;
	//���ƶ���λ�ý��ж���
	item xmove[8];
	//����ջ����ʼ��
	point start, *p;
	//��ջ���ж���
	MazeStack *s;
	s = (MazeStack*)malloc(sizeof(MazeStack));
	setNULL(s);
	//���ƶ���λ�ý��ж���
	defineMove(xmove);
	//���Թ���������
	printf("�������Թ���\n");
	for (i = 0; i<m + 2; i++)
		for (j = 0; j<n + 2; j++)
			scanf("%d", &maze[i][j]);
	start.x = 1;
	start.y = 1;
	start.d = -1;
	p = (point*)malloc(sizeof(point));
	//�����ѹ��ջ
	s = push(s, start);
	while (!isEmpty(s)) {
		p = pop(s);
		x = p->x;
		y = p->y;
		d = p->d + 1;
		while (d<8) {
			i = xmove[d].x + x;
			j = xmove[d].y + y;
			if (maze[i][j] == 0) {
				p->d = d;
				s = push(s, *p);
				x = i;
				y = j;
				maze[x][y] = -1;
				point nw;
				nw.x = x;
				nw.y = y;
				nw.d = -1;
				s = push(s, nw);
				if (x == m&&y == n) {
					printf("�ҵ����ڣ�\n");
					while (!isEmpty(s)) {
						p = pop(s);
						printf("%d %d %d\n", p->x, p->y, p->d);
					}
					return 1;
				}
				else {
					break;
				}
			}
			else {
				d++;
			}
		}
	}
	return 0;
}
