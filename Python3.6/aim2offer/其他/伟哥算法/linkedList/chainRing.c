#include<stdio.h>
#include<malloc.h>
/*
  问题: 判断一个链表是否有环，有环请找出环的入口
  思路: 1.用两个指针p和g分别指向第一个节点，同时遍历链表；
        2.p的步长为1，g的步长为2，若最终两个指针相等（相遇），则为有环；否则，遇到空指针时为无环；
        3.有环时，固定g指针，用p按步长为1从第一次相遇处继续遍历链表，直到第二次p和g相遇，此时遍历的长度k为环的长度；
        4.重置p和g指向第一个节点，调整p和g的步长都为1，此时p先走k步；
        5.p和g同时遍历链表，在p和g第一次相等（相遇）的地方即为环的入口。
  tricks: 1.对链表使用不同步长的指针同时移动，可用来判断环
          2.对链表使用固定距离的两个指针同时移动，可用来求解位置相关的问题
*/

struct Node{
	int data;
	struct Node *next;
};

int main()
{
	int LEN = 30, POS = 5;
	struct Node *head, *p, *g, *entrance;

	head = (Node *) malloc(sizeof(Node *));	// 头结点
	head->data = -1;
	head->next = NULL;
	g = head;

	// 无环部分，尾插法，顺序插入
	for(int i=0; i<LEN; i++){
		p = (Node *) malloc(sizeof(Node *));
		p->data = i;
		p->next = NULL;
		g->next = p;
		g = p;
		if(i==POS-1)
			entrance = p; // 记录环的入口
		if(i==LEN-1){
			p->next = entrance; // 形成环
			//printf("%d->%d(环)", i, entrance->data);
		}
		//else
		//	printf("%d->", i);
	}

	p = head->next; // 起始第一个节点，p每次走1步
	g = head->next; // 起始第一个节点，g每次走2步

	while(p){
		printf("(%d,%d)",p->data,g->data);
		p = p->next;
		g = g->next->next;
		if(p == g)
			break; // 第一次在环中任意位置相遇
	}

	int k = 0; // k为环的长度
	while(p){
		p = p->next;
		k++;
		printf("(%d,%d)",p->data,g->data);
		if(p==g)
			break; // 只移动p第二次遇到g时，k为一圈的长度
	}

	printf("\np走了k=%d后p,g再次相遇在(%d,%d)\n", k, p->data, g->data);
	printf("链长: %d, 真实入口位置: %d, k(环长): %d", LEN, POS, k);

	if(k>0){
		p = head->next;
		g = head->next;
		for(int i=0; i<k; i++){
			p = p->next; // 先让p走k步，pg一起按一步走直到相遇的地方为环
		}

		int pos = 1;
		while(p){
			if(p == g){
				printf("\n找到入口位置: %d\n", pos);
				return 1;
			}
			p = p->next;
			g = g->next;
			pos++;
		}
	}
	printf("There is no circle in the link!");
	return 1;
}
