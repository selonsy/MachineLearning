#define _CRT_SECURE_NO_WARNINGS
/*
*运用顺序栈解决迷宫问题
*/

//原始文章
//http://blog.csdn.net/g15827636417/article/details/52749667

#include<stdio.h>
#include<malloc.h>
#define MAXSIZE 100
#define m 6
#define n 8
//对栈中的元素进行定义,d为方向
typedef struct {
	int x, y, d;
}point;
//对栈的结构进行定义
typedef struct {
	point data[MAXSIZE];
	int top;
}MazeStack;
//对移动数组进行定义，方便进行点的移动
typedef struct {
	int x, y;
}item;
//将栈设置为空栈
void setNULL(MazeStack *s) {
	s->top = -1;
}
//判断栈是否为空
bool isEmpty(MazeStack *s) {
	if (s->top >= 0) return false;
	else return true;
}
//进栈操作
MazeStack * push(MazeStack *s, point x) {
	if (s->top>MAXSIZE - 1) {
		printf("栈上溢出！\n");
		return s;
	}
	else {
		s->top++;
		s->data[s->top] = x;
		return s;

	}
}
//退栈操作
point * pop(MazeStack *s) {
	if (isEmpty(s)) {
		printf("栈为空！\n");
		return NULL;
	}
	else {
		s->top--;
		return &(s->data[s->top + 1]);
	}
}
//取栈顶元素
point * getTop(MazeStack *s) {
	if (isEmpty(s)) {
		printf("栈为空！\n");
		return NULL;
	}
	else {
		return &(s->data[s->top]);
	}
}
//对移动的位置进行定义
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
//进行所有操作的测试
int maze_test() {
	//对迷宫进行定义
	int maze[m + 2][n + 2], x, y, i, j, d;
	//对移动的位置进行定义
	item xmove[8];
	//定义栈的起始点
	point start, *p;
	//对栈进行定义
	MazeStack *s;
	s = (MazeStack*)malloc(sizeof(MazeStack));
	setNULL(s);
	//对移动的位置进行定义
	defineMove(xmove);
	//对迷宫进行输入
	printf("请输入迷宫：\n");
	for (i = 0; i<m + 2; i++)
		for (j = 0; j<n + 2; j++)
			scanf("%d", &maze[i][j]);
	start.x = 1;
	start.y = 1;
	start.d = -1;
	p = (point*)malloc(sizeof(point));
	//将起点压入栈
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
					printf("找到出口！\n");
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
