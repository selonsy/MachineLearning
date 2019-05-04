#include<stdio.h>
#include<malloc.h>

struct Node{
    int co;
    bool isEnd;	//判断是否是子窜结尾
    Node *next;
};

int main(){
    int n, m, c, num, temp, count=0;
    int *color;
    scanf("%d %d %d", &n, &m, &c);
    color = (int *)malloc(sizeof(int)*(c+1));
    
    Node *head, *p, *tail;
    for(int i=0; i<n; i++){
        if(i==0){
            head = (Node *)malloc(sizeof(Node *));
            head->co = -1;
            head->next = NULL;
            tail = head;
        }
        scanf("%d", &num);
        if(num == 0){
			color[0] = 0;
            p = (Node *)malloc(sizeof(Node *));
            p->co = 0;
			p->isEnd = true;
            p->next = tail->next;
            tail->next = p;
            tail = p;
        }
        for(int j=0; j<num; j++){
            scanf("%d", &temp);
			color[temp] = 0;
            p = (Node *)malloc(sizeof(Node *));
            p->co = temp;
			p->isEnd = (j==num-1)?true:false;
            p->next = tail->next;
            tail->next = p;
            tail = p;
        }
    }
	tail->next = head->next;

	int nofit_num = 0, check_n = 0, confirm_n = -1;

	Node *q = head;
	p = head->next;
	for(int i=0; i<m-1; ){
		// 先让p走m串
		*(color+p->co) += 1;
		if(p->isEnd)
			i++;
		p = p->next;
	}
	
	count = n + m;	// 计算检测色珠的总窜数
	for(int i=0; i<count; ){
		color[p->co] += 1;
		if(check_n!=confirm_n && *(color+p->co)>1 && p->co != 0){
			nofit_num++;
			confirm_n = check_n;	// 同一窜只计算一次
			printf("confirm_n %d color %d: %d\n", confirm_n, p->co, *(color+p->co));
		}
		
		if(p->isEnd){
			check_n++;	// 此串为结尾
			i++;
			q = q->next;
			while(!q->isEnd){
				*(color+q->co) -= 1;
				q = q->next;
			}
			*(color+q->co) -= 1;
		}

		p = p->next;
	}
	printf("%d\n", nofit_num);
	return 0;
}

/*

5 2 3
3 1 2 3
0
2 2 3
1 2
1 3

2
*/

